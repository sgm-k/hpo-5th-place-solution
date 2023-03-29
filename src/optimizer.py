import numpy as np
import pandas as pd
import scipy.stats as ss
import skopt
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from copy import deepcopy
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from turbo import Turbo1
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube

def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)  # Need to do it this way due to ties
    o_stats = obs[idx]
    return o_stats

def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))  # Replace inf by something large
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss

class MyOptimizer(AbstractOptimizer):

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.space=[]
        self.hebo=None
        self.rec={}
        self.rec_list={}
        self.parameter_pool = {}
        self.parameter_list = []
        self.trial_pool = {}
        self.randseed = self.config.randseed.get()
        self.optuna_distributions =[]
        self.distributions = None
        self.skopt=None
        self.skopt_distributions =[]
        self.turbo=None
        self.turbo_lb=np.zeros(0)
        self.turbo_ub=np.zeros(0)

    def restart(self):
        self.turbo._restart()
        self.turbo._X = np.zeros((0, self.turbo.dim))
        self.turbo._fX = np.zeros((0, 1))
        X_init = latin_hypercube(self.turbo.n_init, len(self.turbo_lb))
        self.X_init = from_unit_cube(X_init, self.turbo_lb, self.turbo_ub)

    def generate_parameter(self) -> None:
        self.check_result()
        new_params = []
        hp_list = self.params.get_parameter_list()
        trial_id = self.trial_id.get()
        n_suggestions=1
        X_next = np.zeros((n_suggestions, len(self.turbo_ub)))
        
        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])
            self.X_init = self.X_init[n_init:, :]  # Remove these pending points
        
        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        if n_adapt > 0:
            if len(self.turbo._X) > 0:  # Use random points if we can't fit a GP
                X = to_unit_cube(np.array(deepcopy(self.turbo._X)), np.array(self.turbo_lb), np.array(self.turbo_ub))
                fX = copula_standardize(deepcopy(self.turbo._fX).ravel())  # Use Copula
                X_cand, y_cand, _ = self.turbo._create_candidates(
                    X, fX, length=self.turbo.length, n_training_steps=100, hypers={}
                )
                X_next[-n_adapt:, :] = self.turbo._select_candidates(X_cand, y_cand)[:n_adapt, :]
                X_next[-n_adapt:, :] = from_unit_cube(X_next[-n_adapt:, :], self.turbo_lb, self.turbo_ub)
        print("Trial ID:",trial_id)
        self.rec = self.hebo.suggest(n_suggestions = 1)
        
        # Skopt
        if trial_id % 2 == 0 and trial_id < 51:
            trial_skopt = self.skopt.ask()
            itr=0

            for hp in hp_list:
                new_param = {
                    'parameter_name': hp.name,
                    'type': hp.type,
                    'value': float(trial_skopt[itr])
                }
                new_params.append(new_param)
                self.rec[hp.name]=new_param["value"]
                itr+=1
            print("Skopt Param",new_params)

        # TuRBO
        elif trial_id % 2 == 1 and trial_id < 51:
            itr=0
            for hp in hp_list:

                new_param = {
                    'parameter_name': hp.name,
                    'type': hp.type,
                    'value': float(X_next[0][itr])
                }
                new_params.append(new_param)
                self.rec[hp.name]=new_param["value"]
                itr+=1
            print("TuRBO Param:",new_params)
        
        # HEBO
        else:
            for hp in hp_list:
                new_param = {
                    'parameter_name': hp.name,
                    'type': hp.type,
                    'value': (float(self.rec[hp.name]))
                }
                new_params.append(new_param)
                self.rec[hp.name]=new_param["value"]
            print("HEBO Param:",new_params)  
            
        self.parameter_pool[trial_id] = new_params
        self.rec_list[trial_id]=self.rec   
        return new_params

    def pre_process(self) -> None:
        super().pre_process()
        self.parameter_list = self.params.get_parameter_list()
        distributions = []
        for p in self.parameter_list:
            para={}
            if p.type == 'FLOAT':
                para["name"]=p.name
                para["type"]="num"
                para["lb"]=p.lower
                para["ub"]=p.upper
                distributions.append(para) 
            elif p.type == 'INT':
                para["name"]=p.name
                para["type"]="int"
                para["lb"]=p.lower
                para["ub"]=p.upper
                distributions.append(para) 
        self.space=DesignSpace().parse(distributions)
        for p in self.parameter_list:
            if p.type == 'FLOAT':
                skopt_space=skopt.space.Real(p.lower,p.upper,name=p.name)
                self.skopt_distributions.append(skopt_space) 
            elif p.type == 'INT':
                skopt_space=skopt.space.Integer(p.lower,p.upper,name=p.name)
                self.skopt_distributions.append(skopt_space) 
        for p in self.parameter_list:
            self.turbo_lb=np.append(self.turbo_lb,p.lower)
            self.turbo_ub=np.append(self.turbo_ub,p.upper) 
        self.create_study()
    
    def post_process(self) -> None:
        self.check_result()
        super().post_process()
    
    def check_result(self) -> None:
        trial_id = self.trial_id.get()
        objective = self.storage.result.get_any_trial_objective(trial_id)
        del_keys = []
        for trial_id, param in self.parameter_pool.items():
            objective = self.storage.result.get_any_trial_objective(trial_id)
            pd.DataFrame()
            if objective is not None:
                # HEBO
                self.hebo.observe(self.rec_list[trial_id],np.array(objective).reshape(-1, 1)) 
                # Skopt
                value_list=self.rec_list[trial_id].iloc[0].values.tolist()
                rounded_list = [round(value, 5) for value in value_list]
                self.skopt.tell(rounded_list,objective)
                # TuRBO
                if len(self.turbo._fX) >= self.turbo.n_init:
                    self.turbo._adjust_length(objective)
                self.turbo.n_evals += 1
                try:
                    self.turbo._X = np.vstack((self.turbo._X, deepcopy(rounded_list)))
                except:
                    self.turbo._X = (rounded_list)
                self.turbo._fX = np.vstack((self.turbo._fX, deepcopy(objective)))
                try:
                    self.turbo.X = np.vstack((self.turbo.X, deepcopy(rounded_list)))
                except:
                    self.turbo.X = (rounded_list)
                self.turbo.fX = np.vstack((self.turbo.fX, deepcopy(objective)))
                del_keys.append(trial_id)
        for key in del_keys:
            self.parameter_pool.pop(key)
            self.logger.info(f'trial_id {key} is deleted from parameter_pool')

    def create_study(self) -> None:       
        if self.opt is None:
            # HEBO
            self.hebo = HEBO(self.space)
            # Skopt            
            self.skopt = skopt.Optimizer(self.skopt_distributions)
            # TuRBO
            dim_ = len(self.turbo_lb)
            max_evals_ = np.iinfo(np.int32).max  
            self.turbo = Turbo1(
                f=None,
                lb=self.turbo_lb,
                ub=self.turbo_ub,
                n_init=2 * dim_ + 1,
                max_evals=max_evals_,
                batch_size=1,  # We need to update this later
                verbose=False,
            )
            self.restart()