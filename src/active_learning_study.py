from src.data_manager import DataManager
from src.analyzer import Analyzer
from src.experiment import Experiment
import numpy as np
from src.al_runner import ALRunner
from src.common import L_PATH, U_PATH, PROCESSED_DATASETS, EXPERIMENTS
from src.common import HYBRID, HIGHWAY_TYPES
from joblib import dump, load
import pandas as pd

class ALStudy :  

    def __init__(self):
        self.analyzer = Analyzer()
        self.experimenter = Experiment()
        self.dataManager = None
        self.D = None
    

    """
    ************************
    *                      *
    *    DATASET ANALYSIS   *
    *                      *
    ************************
    """
    
    def load_D(self):
        self.dataManager = DataManager()
        self.dataManager.load_D()
        D = self.dataManager.get_D()
        self.D = D

    def build_split_labelled_unlabelled(self):
        self.dataManager.build_L_and_U()
    
    def get_basic_features_analysis_from_D(self):
        highway_distribution_table = self.analyzer.get_distribution(self.D, "highway")
        speed_distribution_table = self.analyzer.get_speed_distribution(self.D)
        lanes_distribution_table = self.analyzer.get_distribution(self.D, "lanes")
        return highway_distribution_table, speed_distribution_table, lanes_distribution_table
    
    def print_map_D(self):
        self.analyzer.plot_highway_distribution(self.D)
    
    def get_target_analysis_from_D(self):
        traffic_table = self.analyzer.get_traffic_variation_table(self.D)
        self.analyzer.plot_week_traffic_variation(self.D)
        self.analyzer.plot_month_records_by_week(self.D)
        return traffic_table
    
    def analyze_U_and_L_distribution(self):
        L = load(L_PATH)
        U = load(U_PATH)
        table_L = self.analyzer.get_highway_distribution_table(L)
        table_U = self.analyzer.get_highway_distribution_table(U)
        return table_L, table_U
    
    def print_L_map(self):
        L = load(L_PATH)
        self.analyzer.plot_highway_distribution_L(L)
    
    """
    ************************
    *                      *
    * ADDITIONAL FEATURES   *
    *                      *
    ************************
    """
    def set_seeds_experiment(self, seeds):
        self.dataManager = DataManager()
        self.dataManager.build_validation_test_sets(seeds)

    def apply_additional_features_experiment(self, seeds):
        self.experimenter.study_additional_features_accuracy(seeds)

    """
    ************************
    *                      *
    * METHODOLOGY   *
    *                      *
    ************************
    """

    def analyse_batch_parameter(self, seeds):
        self._set_experiment_for_evaluation(False)
        D_L = load(L_PATH)
        batchs = [25, 50, 100]
        for batch in batchs:
            runner = ALRunner(300, batch, D_L, False)
            mse_scores = []
            for seed in seeds:
                D_U = self._load_D_U(seed)
                runner.set_D_U(D_U)
                dps = runner.start_al_loop(HYBRID, seed)
                runner.reset_loop(D_L, D_U)
                _, mse, _ = self.experimenter.analyse_datapoints_selection(dps, seed)
                mse_scores.append(mse)
            mean_mse = np.mean(mse_scores)
            std_mse = np.std(mse_scores)
            print(f"Batch {batch}: mean={mean_mse}, std={std_mse}")
    
    def tune_models_parameters(self, seeds):
        self._set_experiment_for_evaluation(False)
        best_tuned = self.experimenter.study_parameters_for_rf(seeds)
        print("Best parameters:")
        for key, value in best_tuned.items():
            if key not in ("seed", "mse"):
                print(f" {key}: {value}")


    """
    ************************************
    *                                  *
    * PERFORMANCE ASSESSEMENT          *
    *                                   *
    *************************************
    """

    def analyse_max_accuracy(self, seeds):
        self._set_experiment_for_evaluation(True)
        r2_scores, mse_scores, mae_mr_scores, mse_per_category_scores = [], [], [], []
        for seed in seeds:
            D_U = self._load_D_U(seed)
            r2, mse, mae_mr, mse_per_category = self.experimenter.compute_max_accuracy(D_U, seed)
            r2_scores.append(r2)
            mse_scores.append(mse)
            mae_mr_scores.append(mae_mr)
            mse_per_category_scores.append(mse_per_category)
        print("Average results:")
        print("R2:", np.mean(r2_scores))
        print("MSE:", np.mean(mse_scores))
        print("MAE MR:", np.mean(mae_mr_scores))
        print("-------------------")
        print("Std R2:", np.std(r2_scores))
        print("Std MSE:", np.std(mse_scores))
        print("Std MAE MR:", np.std(mae_mr_scores))
        mean_mse_category = np.mean(mse_per_category_scores, axis=0)
        std_mse_category = np.std(mse_per_category_scores, axis=0)
        self.analyzer.plot_per_category(mean_mse_category, std_mse_category)




    def analyse_performance_strategy_per_budget_threshold(self, type, seeds):
        self._set_experiment_for_evaluation(True)
        D_L = load(L_PATH)
        all_results = []
        runner = ALRunner(500, 25, D_L, False)
        for seed in seeds:
            D_U = self._load_D_U(seed)
            runner.set_D_U(D_U)
            dps = runner.start_al_loop(type, seed)
            runner.reset_loop(D_L, D_U)
            dump(dps, EXPERIMENTS / f"dps_{type}_{seed}.joblib")
            print("   dps selection done")
            seed_results = self.experimenter.assess_accuracy_per_budget_threshold(dps, seed)
            all_results.append(seed_results)
        full_results = pd.concat(all_results, ignore_index=True)
        dump(full_results, EXPERIMENTS / f"dps_full_{type}.joblib")
        summary = full_results.groupby("budget").agg(avg_r2=("r2", "mean"),std_r2=("r2", "std"),avg_mse=("mse", "mean"),std_mse=("mse", "std"),avg_mae_mr=("mae_mr", "mean"),std_mae_mr=("mae_mr", "std"))
        summary = summary.reset_index()
        self.analyzer.plot_evolution_accuracy_per_budget(summary)
        self.analyzer.print_evolution_accuracy_per_budget(summary)
    
    def analyse_hybrid_per_budget_threshold(self, seeds):
        self._set_experiment_for_evaluation(True)
        D_L = load(L_PATH)
        all_results = []
        runner = ALRunner(500, 25, D_L, False)
        for seed in seeds:
            D_U = self._load_D_U(seed)
            runner.set_D_U(D_U)
            dps = runner.start_al_loop(HYBRID, seed)
            runner.reset_loop(D_L, D_U)
            dump(dps, EXPERIMENTS / f"dps_{HYBRID}_{seed}.joblib")
            print("   dps selection done")
            seed_results = self.experimenter.assess_accuracy_per_budget_threshold(dps, seed, True)
            all_results.append(seed_results)
        full_results = pd.concat(all_results, ignore_index=True)
        dump(full_results, EXPERIMENTS / f"dps_full_{HYBRID}.joblib")
        summary = full_results.groupby("budget").agg(avg_r2=("r2", "mean"),std_r2=("r2", "std"),avg_mse=("mse", "mean"),std_mse=("mse", "std"),avg_mae_mr=("mae_mr", "mean"),std_mae_mr=("mae_mr", "std"))
        summary = summary.reset_index()
        self.analyzer.plot_evolution_accuracy_per_budget(summary)
        self.analyzer.print_evolution_accuracy_per_budget(summary)


    def _load_D_U(self, seed):
        D_U = load(f"{PROCESSED_DATASETS}/U_{seed}.joblib")
        return D_U

    def _set_experiment_for_evaluation(self, test_phase=True):
        self.experimenter = Experiment(test_phase)
    

    def analyse_performance_strategy(self, type, batch_size, budget, seeds, test_phase=True):
        r2_scores, mse_scores, mae_mr_scores = [], [], []
        self._set_experiment_for_evaluation(test_phase)
        D_L = load(L_PATH)
        runner = ALRunner(budget, batch_size, D_L, False)
        for seed in seeds:
            D_U = self._load_D_U(seed)
            runner.set_D_U(D_U)
            dps = runner.start_al_loop(type, seed)
            runner.reset_loop(D_L, D_U)
            r2, mse, mae_mr = self.experimenter.analyse_datapoints_selection(dps, seed)
            r2_scores.append(r2)
            mse_scores.append(mse)
            mae_mr_scores.append(mae_mr)
        print("Average results:")
        print("R2:", round(np.mean(r2_scores), 4))
        print("MSE:", round(np.mean(mse_scores), 4))
        print("MAE MR:", round(np.mean(mae_mr_scores), 4))

    def compare_different_strategy(self):
        self.analyzer.plot_strategy_comparison()
    
    def analyse_road_type_composition(self, seeds):
        res = self._load_hybrid_data(seeds)
        self.analyzer.plot_road_type_by_batch(res, seeds)
        self.analyzer.compare_with_full_dataset(res)
        self.analyzer.compare_seed_stability(res)

    def analyse_geography_hybrid(self, seed):
        D_L = load(L_PATH)
        batch = load(f"{EXPERIMENTS}/dps_9_{seed}.joblib")
        self.analyzer.plot_seed_geography(batch, seed)
        self.analyzer.plot_seed_geography_with_labelled(batch, D_L, seed)
        self.analyzer.plot_seed_geography_first_batches(batch, D_L, seed)
    
    def analyse_recurrence_hybrid(self, seeds):
        batches = self._load_hybrid_data(seeds)
        self.analyzer.analyze_recurrent_points(batches)
        self.analyzer.analyze_batch_recurrence(batches)
        self.analyzer.plot_points_present_in_all_seeds(batches, seeds)
    
    def analyse_uncertainty_diversity_scores(self, seed):
        self.analyzer.plot_batch_zero_scores(seed)
    

    def _load_hybrid_data(self, seeds):
        batch_concatained = []
        base_columns = ["lon", "lat"] + HIGHWAY_TYPES
        for seed in seeds:
            batches = load(f"{EXPERIMENTS}/dps_9_{seed}.joblib")
            for batch_id, df in enumerate(batches):
                batch = df[base_columns].copy()
                batch["batch"] = batch_id
                batch["seed"] = seed
                roads = batch[HIGHWAY_TYPES].idxmax(axis=1)
                batch["road_type"] = roads.str.removeprefix("highway_")
                batch_concatained.append(batch)
        res = pd.concat(batch_concatained, ignore_index=True)
        return res



        