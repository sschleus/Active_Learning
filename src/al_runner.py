import pandas as pd
from src.common import HIGHWAY_TYPES
from src.al_strategy import Strategy

class ALRunner:
    def __init__(self, budget, batch_size, D_L, verbose=False):
        self.init_budget = budget
        self.remaining_budget = budget
        self.batch_size = batch_size
        self.verbose = verbose
        self.D_L = D_L
        self.D_U = None
    

    """
    ************************
    *                      *
    *    PRIVATE METHODS   *
    *                      *
    ************************
    """
    
    def _update_components(self, n, selected):
        self.remaining_budget -= n
        self.D_L = pd.concat([self.D_L, selected])
        self.D_U = self.D_U.drop(selected.index)
    
    def _display_selection(self, df):
        df = pd.concat(df, ignore_index=True)
        for highway_type in HIGHWAY_TYPES:
            print(f"{highway_type}: {df[highway_type].sum()}")
    
    def _run_strategy(self, strategy):
        if self.init_budget != self.remaining_budget:
            strategy.update_informations(self.D_L, self.D_U, self.init_budget - self.remaining_budget)
        selected = strategy.run()
        self._update_components(self.batch_size, selected)
        return selected
    
    """
    ************************
    *                      *
    *    PUBLIC METHODS   *
    *                      *
    ************************
    """

    def set_D_U(self, D_U):
        self.D_U = D_U
    
    def reset_loop(self, D_L, D_U):
        self.remaining_budget = self.init_budget
        self.D_L = D_L
        self.D_U = D_U
    
    def start_al_loop(self, active_learning_type, seed):
        labelised = []
        strategy = Strategy(active_learning_type, self.D_L, self.D_U, self.batch_size, self.init_budget - self.remaining_budget, seed)
        strategy.init_model()

        while self.remaining_budget > 0:
            dataPoints_batch = self._run_strategy(strategy)
            strategy.add_batch_to_D_L(dataPoints_batch)
            if self.verbose :
                print("Budget: " + str(self.remaining_budget))
            labelised.append(dataPoints_batch)
        if self.verbose:
            self._display_selection(labelised)
        return labelised
    

    
