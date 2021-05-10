from sklearn import ensemble
from sklearn import model_selection
from sklearn import linear_model
from sklearn.experimental import enable_halving_search_cv
from sklearn import metrics 
from functools import partial
import pandas as pd
import numpy as np
import config
import json
import os
import dispatcher
import logging

#-------------------------------------------------------------------
#-------------------------------CLASSES-----------------------------
#-------------------------------------------------------------------

class Tune:

    def __init__(self, estimator, train_X, train_y, n_folds = 5):
        self.model = estimator
        self.train_X = train_X
        self.train_y = train_y.values.ravel()
        self.folds = model_selection.StratifiedKFold(n_splits = n_folds)

    
    def half_searchCV(self,params, metric):
        
        search = model_selection.HalvingGridSearchCV(self.model, params, scoring = metric, n_jobs = 10, cv = self.folds)
        print("Grid Search Begins")
        search.fit(self.train_X, self.train_y)
        self.best_params = search.best_params_
        self.best_score = search.best_score_


if __name__ == "__main__":

    train_X = pd.read_csv(config.train_X, index_col = config.index)
    train_y = pd.read_csv(config.train_y, index_col = config.index)

    folds_col = train_X.pop("kfold")

    # # hyperparameter tuning
    estimator_name = "logistic_regression"
    estimator = dispatcher.MODELS[estimator_name]
    # estimator = partial(estimator, n_jobs = 10)
    metric = "f1"
    #Parameter values to perform grid_search over
    params_grid = {
        "C" : np.logspace(-3,3,10),
        "class_weight" :["balanced", {0:1,1:1}, {0:0.25, 1:0.75}],
        "max_iter": [2000]
    }
    
    #Tuning the hyperparameters
    tune = Tune(estimator(), train_X, train_y)
    tune.half_searchCV(params_grid, metric)
    print(f"The best f1_score is {tune.best_score}")
    print(f"The best parameters are {tune.best_params}")

    best_params = tune.best_params
    print("Tuning has been successfully completed!")
    
    avg_f1_train, avg_f1_val = [], []
    folds = folds_col.nunique()
    
    logging.basicConfig(filename= "Tuning_results.log", level = logging.INFO,
                        format = "%(asctime)s:%(name)s:%(message)s")
    logging.info(f"-------------MODEL INFO---------------")
    logging.info(f"Estimator: {str(estimator)}")
    
    train_X['kfold'] = folds_col


    #Storing the parameters
    try:
        with open(config.params_path, "r+") as file:
            data = json.load(file)

        data[estimator_name] = best_params

        with open(config.params_path, "w") as file:
            json.dump(data, file, indent=2)

    except:
        with open(config.params_path, 'w') as file:
            data = dict(estimator_name, best_params)
            json.dump(data, file, indent=2)
            file.close()
    print("Parameters dumped successfully!")    

    #Performing Cross validation on the final model to receive the best f1 score
    for fold in range(folds):
        x_train = train_X[train_X.kfold != fold]
        y_train = train_y.loc[x_train.index,:].values.ravel()
        x_val = train_X[train_X.kfold == fold]
        y_val = train_y.loc[x_val.index,:].values.ravel()
        
        x_train.pop("kfold")
        x_val.pop("kfold")

        model = estimator(**best_params)    
        model.fit(x_train, y_train)

        train_pred = model.predict(x_train)
        val_pred  = model.predict(x_val)
        
        #Computing the F1 score    
        f1_train = metrics.f1_score(y_train, train_pred)
        f1_val = metrics.f1_score(y_val, val_pred)
        
        avg_f1_train.append(f1_train)
        avg_f1_val.append(f1_val)
        
        
        # logging.info("Estimators: {}".format(model.estimators_))
        logging.info(f"-"*50)
        logging.info(f"Fold: {fold}, F1_score train : {f1_train}, F1_score val :{f1_val}")
        logging.info(f"-"*50)
    
    
    #Average f1 score for training and validation sets
    avg_f1_train = sum(avg_f1_train)/folds
    avg_f1_val = sum(avg_f1_val)/folds
    
    logging.info(f"Average F1_score train : {avg_f1_train}, Average F1_score val :{avg_f1_val}")
    logging.info("Best Parameters: {} , Number of columns in df: {}".format(model.get_params, x_train.shape[1]))
    logging.info(f"-"*50)
    logging.info(f"-"*50)