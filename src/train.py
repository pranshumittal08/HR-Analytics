import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import config
import dispatcher
from sklearn import metrics
import logging
import os
import joblib

if __name__ == "__main__":
    train_X = pd.read_csv(config.train_X, index_col = "employee_id")
    train_y = pd.read_csv(config.train_y, index_col = "employee_id")
    
    avg_f1_train, avg_f1_val = [], []
    folds = train_X.kfold.nunique()
    
    logging.basicConfig(filename= "Training_results.log", level = logging.INFO,
                        format = "%(asctime)s:%(name)s:%(message)s")
    logging.info(f"-------------MODEL INFO---------------")
    
    for fold in range(folds):
        x_train = train_X[train_X.kfold != fold]
        y_train = train_y.loc[x_train.index,:].values.ravel()
        x_val = train_X[train_X.kfold == fold]
        y_val = train_y.loc[x_val.index,:].values.ravel()
        
        #Dropping the kfold column from the dataframes
        x_train.pop("kfold")
        x_val.pop("kfold")
        
        #Creating the model
        est_1 = ("rf",dispatcher.MODELS["random_forest"])
        est_2 = ("svm",dispatcher.MODELS["SVM"])
        est_3 = ("lr",dispatcher.MODELS["logistic_regression"])
        est_4 = ("catboost",dispatcher.MODELS["catBoost"])
        
        model = ensemble.StackingClassifier(estimators = [est_1,est_2, est_4], 
                                            final_estimator = CatBoostClassifier(early_stopping_rounds=5,  
                                                                                 class_weights = {0 : 0.25, 1: 0.75}), 
                                            n_jobs = 8, 
                                            verbose = 2)
        
        # model = dispatcher.MODELS["random_forest"]
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred  = model.predict(x_val)
        
        #Computing the F1 score    
        f1_train = metrics.f1_score(y_train, train_pred)
        f1_val = metrics.f1_score(y_val, val_pred)
        
        avg_f1_train.append(f1_train)
        avg_f1_val.append(f1_val)
        
        #Perform Logging
        logging.info("Estimators: {}, Final Estimator: {}".format(model.estimators_, model.final_estimator_))
        # logging.info("Estimators: {}".format(model.estimators_))
        logging.info(f"-"*50)
        logging.info(f"Fold: {fold}, F1_score train : {f1_train}, F1_score val :{f1_val}")
        logging.info(f"-"*50)
    
    
    #Average f1 score for training and validation sets
    avg_f1_train = sum(avg_f1_train)/folds
    avg_f1_val = sum(avg_f1_val)/folds
    
    logging.info(f"Average F1_score train : {avg_f1_train}, Average F1_score val :{avg_f1_val}")
    logging.info(f"-"*50)
    logging.info(f"-"*50)
     
    #Save the model
    joblib.dump(model, filename = os.path.join(config.model_path, "stack_model_catb_1.pkl"))
    
    
    
    
    
    
    
    