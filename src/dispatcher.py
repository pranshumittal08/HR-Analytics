from sklearn import linear_model
from sklearn import ensemble
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


MODELS = {"random_forest": ensemble.RandomForestClassifier, 
       "logistic_regression": linear_model.LogisticRegression,
       "ada_boost": ensemble.AdaBoostClassifier, 
       "svm" : SVC, 
       "xg_boost"  : XGBClassifier,
       "cat_boost" : CatBoostClassifier}


