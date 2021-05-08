from sklearn import linear_model
from sklearn import ensemble
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


MODELS = {"random_forest": ensemble.RandomForestClassifier(n_estimators = 200,
                                                           max_depth = 12,
                                                           min_samples_leaf=100,
                                                           max_features = 15,
                                                           class_weight = {0:1,1:3}, ), 
       "logistic_regression": linear_model.LogisticRegression(max_iter = 2000, 
                                                              class_weight={0:1,1:3}),
       "AdaBoost": ensemble.AdaBoostClassifier(n_estimators =200, learning_rate = 0.1), 
       "SVM" : SVC(C = 8, class_weight= {0: 1, 1: 3}, gamma = 0.002), 
       "xgBoost"  : XGBClassifier(class_weights = {0 : 0.25, 1: 0.75}),
       "catBoost" : CatBoostClassifier(early_stopping_rounds=5, 
                                       class_weights = {0 : 0.25, 1: 0.75})}


