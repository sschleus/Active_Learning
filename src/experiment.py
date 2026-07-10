from joblib import load
from src.common import PROCESSED_DATASETS, L_PATH
from src.common import BASE, CONNECTIVITY, DISTANCE, SPATIAL_DENSITY
from src.model import Model
import pandas as pd
import numpy as np

class Experiment:
    def __init__(self, test_phase=False):
        self.is_test_phase = test_phase
        self.model = Model()

    """
    ************************
    *                      *
    *    PRIVATE METHODS   *
    *                      *
    ************************
    """

    def _load_evaluation_sets(self, seed):
        val = load(f"{PROCESSED_DATASETS}/val_{seed}.joblib")
        test = load(f"{PROCESSED_DATASETS}/test_{seed}.joblib")
        return val, test
    
    def _load_correct_evaluation_sets(self, seed):
        validation, test = self._load_evaluation_sets(seed)
        if self.is_test_phase:
            del validation
            return test
        else:
            del test
            return validation
        
    def _get_mean_and_std_per_set(self, results, name):
        values = []
        for result in results:
            values.extend(result[name])
        return np.mean(values), np.std(values)
    
    def _get_parameters_from_result_tuning(self, result, parameter_names):
        parameters = {}
        for name in parameter_names:
            parameters[name] = result[name]
        return parameters
    
    def _find_best_rf_parameters(self, results, parameter_names):
        best_result = None
        best_mean_mse = float("inf")
        for res_a in results:
            parameters_a = self._get_parameters_from_result_tuning(res_a,parameter_names)
            mse_values = []
            for res_b in results:
                parameters_b = self._get_parameters_from_result_tuning(res_b,parameter_names)
                if parameters_a == parameters_b:
                    mse_values.append(res_b["mse"])
            mean_mse = sum(mse_values) / len(mse_values)
            if mean_mse < best_mean_mse:
                best_mean_mse = mean_mse
                best_result = res_a.copy()
                best_result["mean_mse"] = mean_mse
        return best_result
    
    def _compute_accuracy(self, datapoints, evaluation, seed):
        D_L = load(L_PATH)
        self.model.init_sets(D_L, evaluation)
        if datapoints:
            dps_total = pd.concat([df for df in datapoints], ignore_index=True)  
            self.model.update_training(dps_total)
        predicts = self.model.fit_predict_random_forest(seed)
        r2 = self.model.analyze_r2(predicts)
        mse  = self.model.analyze_mse(predicts)
        mae_mr = self.model.analyze_medium_traffic_road_mae(predicts)
        return r2, mse, mae_mr
    
    def _compute_max_accuracy(self, D_U, eval_set, seed):
        D_L = load(L_PATH)
        self.model.init_sets(D_L, eval_set)
        self.model.update_training(D_U)
        predicts = self.model.fit_predict_random_forest(seed)
        r2 = self.model.analyze_r2(predicts)
        mse  = self.model.analyze_mse(predicts)
        mae_mr = self.model.analyze_medium_traffic_road_mae(predicts)
        mse_per_category = self.model.analyze_mse_per_category(predicts)
        return r2, mse, mae_mr, mse_per_category

    
    def _take_first_segments(self, dps, threshold):
        selected = []
        remaining = threshold
        for df in dps:
            if remaining <= 0:
                break
            selected_df = df.iloc[:remaining]
            selected.append(selected_df)
            remaining -= len(selected_df)
        return selected




    """
    ************************
    *                      *
    *    PUBLIC METHODS   *
    *                      *
    ************************
    """
    def compute_max_accuracy(self, D_U, seed):
        eval_set = self._load_correct_evaluation_sets(seed)
        r2, mse, mae_mr, mse_per_category = self._compute_max_accuracy(D_U, eval_set, seed)
        return r2, mse, mae_mr, mse_per_category


    def analyse_datapoints_selection(self, datapoints, seed):
        evaluation = self._load_correct_evaluation_sets(seed)
        r2, mse, mae_mr = self._compute_accuracy(datapoints, evaluation, seed)
        return r2, mse, mae_mr
    
    def assess_accuracy_per_budget_threshold(self, dps, seed, granularity=False):
        eval_set = self._load_correct_evaluation_sets(seed)
        thresholds = [0, 100, 200, 300, 400, 500]
        if granularity:
            thresholds = [x for x in range(0,501,25)]
        results = []
        for threshold in thresholds:
            dps_to_evaluate = None
            if threshold != 0:
                dps_to_evaluate = self._take_first_segments(dps, threshold)
            r2, mse, mae_mr = self._compute_accuracy(dps_to_evaluate, eval_set, seed)
            results.append({"budget": threshold,"r2": r2,"mse": mse,"mae_mr": mae_mr, "seed": seed})
        return pd.DataFrame(results)



    def study_additional_features_accuracy(self, seeds):
        rmse_results = []
        r2_results = []
        D_L = load(L_PATH)
        model = Model()
        model.init_sets(D_L)
        X, y = model.get_training_set()
        base = [c for c in BASE[:-1]]
        features_sets = {"base": base, "base_connect":base + CONNECTIVITY, "base_dist": base + DISTANCE, "base_density": base + SPATIAL_DENSITY,  "total":  base + DISTANCE + CONNECTIVITY + SPATIAL_DENSITY}
        for seed in seeds:
            rmse, r2 = model.cross_validation(X, y, features_sets, seed)
            rmse_results.append(rmse)
            r2_results.append(r2)
        for set in features_sets:
            print(f"Name:{set}")
            rmse_mean, rmse_std = self._get_mean_and_std_per_set(rmse_results, set)
            r2_mean, r2_std = self._get_mean_and_std_per_set(r2_results, set)
            print(f"RMSE: {rmse_mean}  RMSE std: {rmse_std}")
            print(f"R2: {r2_mean}  R2 std: {r2_std}")
            print("#################")


    def study_parameters_for_rf(self, seeds):
        D_L = load(L_PATH)
        results = []
        for seed in seeds:
            eval_set = self._load_correct_evaluation_sets(seed)
            self.model.init_sets(D_L, eval_set)
            seed_results, tested_parameters = self.model.tune_rf(seed)
            results.extend(seed_results)
        parameter_names = tested_parameters.keys()
        return self._find_best_rf_parameters(results, parameter_names)


        
       
        