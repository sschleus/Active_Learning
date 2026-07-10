import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from src.common import HIGHWAY_MEDIUM, HIGHWAY_TYPES
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import warnings

class Model:
    def init_sets(self, training, evaluation=None):
        self.training_features = training.drop(columns=["target"])
        self.training_labels = training["target"]
        if evaluation is not None:
            self.evaluation_features = evaluation.drop(columns=["target"])
            self.evaluation_labels = evaluation["target"]
    
    def get_training_set(self):
        x =  np.asarray(self.training_features, float)
        y = np.vstack(self.training_labels.to_numpy())
        return x, y
    
    def get_evaluation_set(self):
        x =  np.asarray(self.evaluation_features, float)
        y = np.vstack(self.evaluation_labels.to_numpy())
        return x, y
    
    def cross_validation(self, X, y, feature_sets, seed):
        X = pd.DataFrame(X, columns=feature_sets["total"])
        model = RandomForestRegressor(n_estimators=500,random_state=seed,n_jobs=-1)
        kfold = KFold(n_splits=5,shuffle=True,random_state=seed)
        rmse_scores = {}
        r2_scores = {}
        for set_name, features in feature_sets.items():
            rmse = -cross_val_score(model,X[features],y,cv=kfold,scoring="neg_root_mean_squared_error")
            r2 = cross_val_score(model,X[features],y,cv=kfold,scoring="r2")
            rmse_scores[set_name] = rmse
            r2_scores[set_name] = r2
        return rmse_scores, r2_scores
    
    def get_mse(self, result):
            return result["mse"]
    
    def tune_rf(self, seed):
        results = []
        X_train, Y_train = self.get_training_set()
        X_val, Y_val = self.get_evaluation_set()
        parameters_tested = {"n_estimators": [100, 300], "max_depth": [None, 20], "min_samples_split": [2, 10], "min_samples_leaf": [1, 5], "max_features": ["sqrt", 1.0]}
        for n_estimators in parameters_tested["n_estimators"]:
            for max_depth in parameters_tested["max_depth"]:
                for min_samples_split in parameters_tested["min_samples_split"]:
                    for min_samples_leaf in parameters_tested["min_samples_leaf"]:
                        for max_features in parameters_tested["max_features"]:
                            model = RandomForestRegressor(bootstrap=True, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=seed, n_jobs=-1)
                            model.fit(X_train, Y_train)
                            preds = model.predict(X_val)
                            mse = mean_squared_error(Y_val, preds)
                            results.append({"seed": seed, "n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf, "max_features": max_features, "mse": mse})
        results.sort(key=self.get_mse)
        return results, parameters_tested
    
    def update_training(self, df: pd.DataFrame):
        new_labelled_features = df.drop(columns=["target"])
        new_labelled_labels = df["target"]
        self.training_features = pd.concat([self.training_features, new_labelled_features], ignore_index=True)
        self.training_labels = pd.concat([self.training_labels, new_labelled_labels], ignore_index=True)
    
    def fit_predict_random_forest(self, seed):
        random_forest = RandomForestRegressor(bootstrap=True,n_estimators=100, min_samples_split=2, min_samples_leaf=5,
        max_depth=None, max_features=1.0,n_jobs=-1,random_state=seed)
        X, Y = self.get_training_set()
        X_eval, _ = self.get_evaluation_set()
        random_forest.fit(X, Y)
        predictions = random_forest.predict(X_eval)
        return predictions
    
    def learn_random_forest(self, X, Y, seed):
        random_forest = RandomForestRegressor(bootstrap=True,n_estimators=300, min_samples_split=2, min_samples_leaf=1,
        max_depth=None, max_features=1.0,n_jobs=-1,random_state=seed)
        warnings.simplefilter(action='ignore', category=UserWarning)
        random_forest.fit(X, Y) 
        return random_forest
    
    def analyze_r2(self, predictions:np.ndarray) -> float:
        _, Y = self.get_evaluation_set()
        r2 = r2_score(Y, predictions)
        return r2
    
    def analyze_mse(self, predictions):
        _, Y = self.get_evaluation_set()
        mse = mean_squared_error(Y, predictions)
        return mse
    
    def analyze_medium_traffic_road_mae(self, predictions):
        _, Y = self.get_evaluation_set()
        mask = np.zeros(len(self.evaluation_features), dtype=bool)
        for highway_type in HIGHWAY_MEDIUM:
            mask |= (self.evaluation_features[highway_type] == 1)
        mae = np.mean(np.abs(Y[mask] - predictions[mask]))
        return mae
    
    def analyze_mse_per_category(self, predictions):
        X_features = self.evaluation_features
        _, Y = self.get_evaluation_set()
        res = []
        for highway_type in HIGHWAY_TYPES:
            mask = X_features[highway_type] == 1
            mse = mean_squared_error(Y[mask], predictions[mask])
            res.append(mse)
        return res