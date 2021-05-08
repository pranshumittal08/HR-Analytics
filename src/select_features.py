from sklearn import feature_selection
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn import linear_model
import pandas as pd
import numpy as np
import config


#########################################################################
#--------------------------------CLASSES------------------------------
#########################################################################

class RecursiveFeatureElimination:

    def __init__(self, train_X, train_y, test_df, n_features = 40, estimator = linear_model.LogisticRegression(max_iter = 2000)):
        self.train_X = train_X
        self.train_y = train_y
        self.test_df = test_df
        self.n_features = n_features
        self.estimator = estimator
    
    def transform(self, selector):
        self.important_features = self.train_X.columns[selector.get_support()]
        self.feature_ranking = selector.ranking_
        self.train_X.loc[:, self.important_features] = self.train_X[self.important_features]
        self.test_df.loc[:, self.important_features] = self.test_df[self.important_features]
    
    def rfe(self):
        rfe_obj = feature_selection.RFE(self.estimator, n_features_to_select = self.n_features)
        rfe_obj.fit(self.train_X, self.train_y)
        self.transform(rfe_obj)

    def rfeCV(self, n_folds = 5):
        self.important_features = [True] * (len(self.train_X.columns) - 1)
        for fold in range(n_folds):
            rfe_obj = feature_selection.RFE(self.estimator, n_features_to_select = self.n_features)
            fold_index = self.train_X[self.train_X.kfold != fold].index
            df = self.train_X[self.train_X.kfold != fold].copy()
            df.pop('kfold')
            rfe_obj.fit(df, self.train_y.loc[fold_index,:].values.ravel())
            
            assert(len(self.important_features) == len(rfe_obj.get_support()))
            tup = zip(self.important_features,rfe_obj.get_support())
            self.important_features = [i[0] and i[1] for i in tup]
            print("Completed for fold {}".format(fold))

        self.important_features = df.columns[self.important_features]
        
        self.train_X = self.train_X[self.important_features]
        self.test_df = self.test_df[self.important_features]
        

#########################################################################
#--------------------------------FUNCTIONS------------------------------
#########################################################################


def drop_low_var_columns(train_X, test_df, threshold):

    select_features = feature_selection.VarianceThreshold(threshold = threshold)
    select_features.fit(train_X)
    cols_to_drop = train_X.columns[select_features.get_support() != True]
    train_X.drop(columns = cols_to_drop, inplace = True)
    test_df.drop(columns = cols_to_drop, inplace = True)


#########################################################################
#--------------------------------MAIN------------------------------
#########################################################################
if __name__ == "__main__":
    train_X = pd.read_csv(config.train_X, index_col = 'employee_id')
    train_y = pd.read_csv(config.train_y, index_col = 'employee_id')
    test_df = pd.read_csv(config.test_df, index_col = "employee_id")
    drop_low_var_columns(train_X, test_df, threshold = 0.01)
    feature_selector = RecursiveFeatureElimination(train_X,train_y, test_df )
    feature_selector.rfeCV()
    folds_col = train_X.kfold
    train_X = feature_selector.train_X.copy()
    train_X["kfold"] = folds_col
    test_df = feature_selector.test_df.copy()
    train_X.to_csv(config.train_X)
    test_df.to_csv(config.test_df)
