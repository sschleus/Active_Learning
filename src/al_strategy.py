import random
import pandas as pd
from src.model import Model
import numpy as np
from math import floor
from joblib import dump

from src.common import HIGHWAY_TYPES
from src.common import HYBRID_ANALYSIS
from src.common import SINGLE, NO_MODEL, UNIFORM_RANDOM_SAMPLING, STRATIFIED_RANDOM_SAMPLING, RANDOM_SAMPLING_HIGH_TRAFFIC_ONLY, UNCERTAINTY_ONLY, FURTHEST_SAMPLING, HYBRID

class Strategy:
    def __init__(self, al_type, D_L, D_U, batch_size, current_spent, seed):
        self.type = al_type
        self.D_L = D_L
        self.D_U = D_U
        self.current_spent = current_spent
        self.batch_size = batch_size
        self.seed = seed
        self.model = None
        self.saved = None
    
    """
    ************************
    *                      *
    *  MODEL AND SET HANDLING       *
    *                      *
    ************************
    """
    
    def _categorize(self):
        if self.type == UNCERTAINTY_ONLY or self.type == HYBRID:
            return SINGLE
        else:
            return NO_MODEL
    
    def add_batch_to_D_L(self, datapoints):
        al_category = self._categorize()
        if al_category == SINGLE:
            self.model.update_training(datapoints)
    
    def init_model(self):
        al_category = self._categorize()
        if al_category == NO_MODEL:
           pass
        elif al_category == SINGLE:
            self.model = Model()
            self.model.init_sets(self.D_L)
    
    def update_informations(self,  D_L_updated, D_U_updated, current_spent):
        self.D_U = D_U_updated
        self.D_L = D_L_updated
        self.current_spent = current_spent

    """
    ************************
    *                      *
    *  STRATEGIES          *
    *                      *
    ************************
    """
    def _save_scores(self, diversity_vector, uncertainty_vector, hybrid_vector, pool_U, pool_L):
        if self.saved:
            self.saved = False
            current_batch = (self.current_spent) // self.batch_size
            dump(diversity_vector, f"{HYBRID_ANALYSIS}/diversity_batch_{current_batch}.joblib")
            dump(uncertainty_vector, f"{HYBRID_ANALYSIS}/uncertainty_batch_{current_batch}.joblib")
            dump(hybrid_vector, f"{HYBRID_ANALYSIS}/hybrid_batch_{current_batch}.joblib")
            dump(pool_U, f"{HYBRID_ANALYSIS}/U_batch_{current_batch}.joblib")
            dump(pool_L, f"{HYBRID_ANALYSIS}/L_batch_{current_batch}.joblib")
        
    def _get_uncertainty_score_normalised(self, u_pool):
        X, Y = self.model.get_training_set()
        model_trained = self.model.learn_random_forest(X, Y, self.seed)
        tree_preds = np.array([tree.predict(u_pool) for tree in model_trained.estimators_])
        tree_variance = np.var(tree_preds, axis=0)
        uncertainty_scores = np.mean(tree_variance, axis=1)
        s_min = np.min(uncertainty_scores)
        s_max = np.max(uncertainty_scores)

        if s_max == s_min:
            norm_scores = np.ones_like(uncertainty_scores, dtype=float)
        else:
            norm_scores = (uncertainty_scores - s_min) / (s_max - s_min)
        return norm_scores

    def _compute_hybrid_score(self, diversity_vector, uncertainty_vector):
        hybrid_scores = diversity_vector * uncertainty_vector
        hybrid_idx = np.argsort(-hybrid_scores)
        return hybrid_idx, hybrid_scores

    def _select_hybrid(self):
        res = []
        pool_U = self.D_U.copy()
        pool_L = self.D_L.copy()

        uncertainty_vector= self._get_uncertainty_score_normalised(pool_U.iloc[:, :-1])

        for _ in range(self.batch_size):
            diversity_vector = self._get_diversity_rankings_fps(pool_U, pool_L)
            ranked_idx, hybrid_vector = self._compute_hybrid_score(diversity_vector,uncertainty_vector)
            self._save_scores(diversity_vector, uncertainty_vector, hybrid_vector, pool_U, pool_L)

            best_pos = ranked_idx[0]
            best_label = pool_U.index[best_pos]

            df = pool_U.loc[[best_label]]
            pool_L = pd.concat([pool_L, df])
            pool_U = pool_U.drop(best_label)
            uncertainty_vector = np.delete(uncertainty_vector, best_pos)
            res.append(df)
        return pd.concat(res, ignore_index=False)
    
    def _get_top_values_idx(self, vector, n):
        size = self.batch_size if n is None else n
        return np.argsort(vector)[-size:]
    
    def _get_uncertainty_ranking(self, pool_U):
        X, Y = self.model.get_training_set()
        model_trained = self.model.learn_random_forest(X, Y, self.seed)
        tree_preds = np.array([tree.predict(pool_U) for tree in model_trained.estimators_])
        tree_variance = np.var(tree_preds, axis=0) 
        uncertainty_scores = np.mean(tree_variance, axis=1) 
        score = self._get_top_values_idx(uncertainty_scores, len(uncertainty_scores))
        return score


    def _select_uncertainty(self):
        pool_U = self.D_U.copy()
        df_features = pool_U.iloc[:, :-1]
        idx_uncertainty = self._get_uncertainty_ranking(df_features)
        pool_selected = pool_U.iloc[idx_uncertainty[:self.batch_size]]
        return pd.concat([pool_selected], ignore_index=False)

    def _get_diversity_rankings_fps(self, pool_u, pool_l):
        coords_u = pool_u[["lon", "lat"]].to_numpy()
        coords_l = pool_l[["lon", "lat"]].to_numpy()
        diff_lon = coords_u[:, None, 0] - coords_l[None, :, 0]
        diff_lat = coords_u[:, None, 1] - coords_l[None, :, 1]
        distances = np.sqrt(diff_lon**2 + diff_lat**2)
        raw_scores = distances.min(axis=1)
        s_max = raw_scores.max()
        s_min = raw_scores.min()
        s_max = raw_scores.max()
        s_min = raw_scores.min()
        norm_scores = (raw_scores - s_min) / (s_max - s_min)
        return norm_scores

    def _select_furthest_point(self):
        selected_parts = []
        pool_U = self.D_U.copy()
        pool_L = self.D_L.copy()

        for _ in range(self.batch_size):
            scores = self._get_diversity_rankings_fps(pool_U, pool_L)
            best_idx = scores.argmax()
            selected_point = pool_U.iloc[[best_idx]]
            selected_parts.append(selected_point)
            pool_L = pd.concat([pool_L, selected_point], ignore_index=False)
            pool_U = pool_U.drop(selected_point.index)
        selected = pd.concat(selected_parts, ignore_index=False)
        return selected
    
    def _count_by_highway(self, df):
        counts = []
        for highway in HIGHWAY_TYPES:
            count = (df[highway] == 1).sum()
            counts.append(count)
        return counts
    
    def _allocate_additions(self):
        roads_to_select_from = self._count_by_highway(self.D_U)
        roads_already_labelled = self._count_by_highway(self.D_L)
        total_roads_to_select_from = sum(roads_to_select_from)
        total_roads_already_labelled = sum(roads_already_labelled)
        total_roads_after_selection = total_roads_already_labelled + self.batch_size

        if total_roads_to_select_from == 0:
            wanted_distribution = [1 / len(roads_to_select_from)] * len(roads_to_select_from)
        else:
            wanted_distribution = [count / total_roads_to_select_from for count in roads_to_select_from]
        
        missing_roads = []
        for wanted_part, already_labelled in zip(wanted_distribution, roads_already_labelled):
            wanted_total = wanted_part * total_roads_after_selection
            missing = wanted_total - already_labelled
            missing_roads.append(max(0, missing))
        total_missing = sum(missing_roads)

        roads_to_add = []
        if total_missing == 0:
            for wanted_part in wanted_distribution:
                roads_to_add.append(self.batch_size * wanted_part)
        else:
            for missing in missing_roads:
                roads_to_add.append(self.batch_size * missing / total_missing)
        final_roads_to_add = [int(floor(value)) for value in roads_to_add]
        remaining_roads = self.batch_size - sum(final_roads_to_add)

        priority = []
        for i in range(len(roads_to_add)):
            decimal_part = roads_to_add[i] - final_roads_to_add[i]
            priority.append((decimal_part, i))
        priority.sort(reverse=True)
        for _, i in priority[:remaining_roads]:
            final_roads_to_add[i] += 1
        return final_roads_to_add


    def _select_random_motorway(self):
        random.seed(self.seed)
        D_U_high_traffic = self.D_U[(self.D_U['highway_motorway'] == 1) |(self.D_U['highway_trunk'] == 1)]
        selected = D_U_high_traffic.sample(n=self.batch_size, random_state=self.seed)
        return selected
    
    def _select_random_pool(self, highway_type, n):
        random.seed(self.seed)
        D_U_filtered = self.D_U[self.D_U[highway_type]==1]
        selected = D_U_filtered.sample(n=n, random_state=self.seed)
        return selected


    def _select_stratified_random_sampling(self) -> pd.DataFrame:
        number_to_select_per_type = self._allocate_additions()
        df = pd.DataFrame()
        for highway_type, n in zip(HIGHWAY_TYPES, number_to_select_per_type):
            selected = self._select_random_pool(highway_type, n)
            df = pd.concat([df, selected], ignore_index=False)
        return df

    
    def _select_uniform_random(self) -> pd.DataFrame:
        random.seed(self.seed)
        selected = self.D_U.sample(n=self.batch_size, random_state=self.seed)
        return selected
    
    """
    ************************
    *                      *
    *  Main method         *
    *                      *
    ************************
    """

    
    def run(self):
        if self.type == UNIFORM_RANDOM_SAMPLING:
            return self._select_uniform_random()
        if self.type == RANDOM_SAMPLING_HIGH_TRAFFIC_ONLY:
            return self._select_random_motorway()
        if self.type == STRATIFIED_RANDOM_SAMPLING:
            return self._select_stratified_random_sampling()
        if self.type == UNCERTAINTY_ONLY:
            return self._select_uncertainty()
        if self.type == FURTHEST_SAMPLING:
            return self._select_furthest_point()
        if self.type == HYBRID:
            self.saved = True
            return self._select_hybrid()
        else:
            raise ValueError