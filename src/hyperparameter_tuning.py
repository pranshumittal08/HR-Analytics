from sklearn import ensemble
from sklearn import model_selection
from sklearn import linear_model
from sklearn.experimental import enable_halving_search_cv
from sklearn import metrics 
import pandas as pd
import numpy as np
import config
import os
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

    # train_X.pop("kfold")

    #Random Forest hyperparameter tuning
    # estimator = ensemble.RandomForestClassifier(n_jobs = 10, verbose = 1)
    # params_grid = {
    #     "class_weight" : ["balanced", "balanced_subsample"],
    #     "max_depth" : np.arange(4, 18,3),
    #     "n_estimators": np.arange(200,1000,200),
    #     "criterion" : ["gini", "entropy"],
    #     "min_samples_leaf": np.arange(50,301,50),
    #     "max_features": np.arange(5, len(train_X.columns)-5, 5),
    #     "min_impurity_decrease":[ 0.001],
    #     "max_samples": np.arange(5,15,3)
    # }
    # metric = "f1"
    # tune = Tune(estimator, train_X, train_y)
    # tune.half_searchCV(params_grid, metric)
    # print(f"The best f1_score is {tune.best_score}")
    # print(f"The best parameters are {tune.best_params}")

    best_params = {'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 10, 'max_features': 20, 'max_samples': 11, 'min_impurity_decrease': 0.001, 'min_samples_leaf': 100, 'n_estimators': 400}
    avg_f1_train, avg_f1_val = [], []
    folds = train_X.kfold.nunique()
    
    logging.basicConfig(filename= "Tuning_results.log", level = logging.INFO,
                        format = "%(asctime)s:%(name)s:%(message)s")
    
    
    
    logging.info(f"-------------MODEL INFO---------------")
    logging.info(f"Estimator: {'Random Forest'}")
    
    for fold in range(folds):
        x_train = train_X[train_X.kfold != fold]
        y_train = train_y.loc[x_train.index,:].values.ravel()
        x_val = train_X[train_X.kfold == fold]
        y_val = train_y.loc[x_val.index,:].values.ravel()
        
        x_train.pop("kfold")
        x_val.pop("kfold")

        model = ensemble.RandomForestClassifier(n_jobs = 10, verbose = 1, class_weight= "balanced_subsample", max_depth = 40, max_samples = 5000)    
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