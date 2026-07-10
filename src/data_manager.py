import pandas as pd
import numpy as np
from joblib import load, dump
from src.common import ORDER, DISTANCE, HIGHWAY_TYPES, SPEED_TYPES, LANES_TYPES, SPATIAL_DENSITY, CONNECTIVITY
from src.common import D_PATH, L_OSM_ID_PATH, CONNECTIVITY_PATH, DISTCITY_PATH, SPATIAL_DENSITY_PATH, L_PATH, U_PATH, PROCESSED_DATASETS


class DataManager:
    unlabelled_df = None
    labelled_df = None
    D = None
    D_prime = None

    def load_D(self):
        self.D = pd.read_parquet(D_PATH, engine='pyarrow')
        self.D_prime = None
    
    """
    ************************
    *                      *
    *  DATA PREPROCESSING   *
    *                      *
    ************************
    """
    def _optimize_memory(self):
        for col in self.D.select_dtypes(include=['float64']).columns:
            self.D[col] = self.D[col].astype('float32')
        for col in self.D.select_dtypes(include=['int64']).columns:
            self.D[col] = self.D[col].astype('int32')
    
    def _filter_records(self):
        self.D = self.D[~self.D["highway"].isin(["tertiary_link", "secondary_link", "primary_link"])]
        self.D = self.D[self.D["lanes"] != 0]

    def _apply_data_preprocessing(self):
        self._optimize_memory()
        self._filter_records()



    """
    ************************
    *                      *
    * FEATURE ENGINEERING   *
    *                      *
    ************************
    """

    def _encode_and_derive_features(self):
        self.D["hour"] = self.D["timestamp"].dt.hour
        self.D["day"] = self.D["timestamp"].dt.day
        self.D["weekday"] = self.D["timestamp"].dt.weekday.astype(int)
        self.D["is_weekend"] = (self.D["timestamp"].dt.weekday >= 5).astype(int)

        self.D = pd.get_dummies(self.D, columns=["highway"], prefix="highway")
    
    def _add_additional_features(self):
        connectivity = pd.read_parquet(CONNECTIVITY_PATH)
        distances = pd.read_csv(DISTCITY_PATH)
        SPATIAL_DENSITY = pd.read_parquet(SPATIAL_DENSITY_PATH)

        D_prime_additional = connectivity.merge(distances, on="osm_id", how="outer")
        D_prime_additional = D_prime_additional.merge(SPATIAL_DENSITY, on="osm_id", how="outer")
        return D_prime_additional
    
    def _apply_aggregation_function(self, D_prime_additional):
        list_df = []
        list_target = []
        for _, road_segment in self.D.groupby("osm_id", sort=False):
           target = self._vectorize_target(road_segment)
           list_df.append(road_segment)
           list_target.append(target)
        self._map_features_to_target(list_df, list_target, D_prime_additional)
        del self.D

    def _vectorize_target(self, df):
        # If same day + same hour, mean together (remove fifteen minutes granularity)
        df_hourly_granuality = (df.groupby(["day", "hour"], as_index=False)
        .agg(target=("sampleSize", "mean"), weekday=("weekday", "first"), is_weekend=("is_weekend", "first")) )
        
        #if same hour, same weekday, concatenate (remove day granularity)
        df_fused_per_weekday = (df_hourly_granuality.groupby(["hour", "weekday", "is_weekend"], as_index=False)
        .agg(median_target=("target", "median")))


        weekday_vec = (df_fused_per_weekday[df_fused_per_weekday["is_weekend"] == False].groupby("hour")["median_target"].mean().values)

        weekend_vec = (df_fused_per_weekday[df_fused_per_weekday["is_weekend"] == True].groupby("hour")["median_target"].mean().values)

        res = np.concatenate([weekday_vec, weekend_vec])
        return res

    
    def _map_features_to_target(self, list_DP, list_target, additional):
        hsl = HIGHWAY_TYPES + SPEED_TYPES + LANES_TYPES
        additional_features = DISTANCE + CONNECTIVITY + SPATIAL_DENSITY

        columns = ["osm_id"] + ['lon', 'lat'] + hsl + additional_features + ['target']
        vec_df = []

        for df_dp, target in zip(list_DP, list_target):
            osm_id = df_dp["osm_id"].iloc[0]
            row_hsl = df_dp[hsl].iloc[0].tolist()
            row_additional = additional.loc[additional["osm_id"] == osm_id, additional_features].iloc[0].tolist()
            row = [osm_id] + [df_dp["lon"].iloc[0], df_dp["lat"].iloc[0]] + row_hsl + row_additional + [target]
            vec_df.append(row)

        df = pd.DataFrame(vec_df, columns=columns)

        numeric_cols = ['lon', 'lat'] + SPEED_TYPES + LANES_TYPES + additional_features
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        self.D_prime = df

    
    def _format_set(self): 
        order =  ["osm_id"] + ORDER
        self.D_prime = self.D_prime[order]
    
    
    def _apply_feature_engineering(self):
        self._encode_and_derive_features()
        D_prime_additional = self._add_additional_features()
        self._apply_aggregation_function(D_prime_additional)
        self._format_set()
    
    """
    ************************
    *                      *
    *    SPLITTING U AND L *
    *                      *
    ************************
    """
    
    def _split_L_and_U(self) -> None:
        sensors_ids_df = pd.read_csv(L_OSM_ID_PATH)
        sensors_ids = sensors_ids_df['osm_id']
        L = self.D_prime[self.D_prime['osm_id'].isin(sensors_ids)]
        U = self.D_prime[~self.D_prime['osm_id'].isin(sensors_ids)]
        L = L.drop(columns=["osm_id"])
        U = U.drop(columns=["osm_id"])
        dump(L, L_PATH)
        dump(U, U_PATH)
        return L, U
    
    """
    ************************
    *                      *
    * VALIDATION AND TEST  *
    *                      *
    ************************
    """

    def _build_subsetdf_from_U(self, U, seed):
        #We want 500 hundred samples
        proportion_highways = [36, 27, 10, 9, 8, 5, 5]
        proportion_highways = [x * 5 for x in proportion_highways]

        sampled_parts = []
        for proportion, highway in zip(proportion_highways, HIGHWAY_TYPES):
            sampled = U[U[highway] == 1].sample(proportion,random_state=seed)
            sampled_parts.append(sampled)
            U = U.drop(sampled.index)
        subset_df = pd.concat(sampled_parts, ignore_index=True)
        return subset_df, U
    

    
    """
    ************************
    *                      *
    *    PUBLIC METHODS   *
    *                      *
    ************************
    """
    def get_D(self):
        return self.D
    
    def build_L_and_U(self):
        self._apply_data_preprocessing()
        self._apply_feature_engineering()
        L, U = self._split_L_and_U()
        return L, U
    
    def build_validation_test_sets(self, seeds):
        D_U = load(U_PATH)
        for seed in seeds:
            val, D_U = self._build_subsetdf_from_U(D_U, seed)
            test, D_U = self._build_subsetdf_from_U(D_U, seed)
            dump(val, f"{PROCESSED_DATASETS}/val_{seed}.joblib")
            dump(test, f"{PROCESSED_DATASETS}/test_{seed}.joblib")
            dump(D_U, f"{PROCESSED_DATASETS}/U_{seed}.joblib")


