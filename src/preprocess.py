import numpy as np
import pandas as pd
from sklearn import impute
from sklearn.experimental import enable_iterative_imputer
import config

#########################################################################
#--------------------------------CLASSES------------------------------
#########################################################################
     
class ImputeMissingValues():
    """
    ------------------------------------------------------------
    Class to impute missing values using different techniques
    ------------------------------------------------------------
    """

    def __init__(self, train_df, test_df = None):
        self.train_df = train_df
        self.test_df = test_df
        self.strategies = ["mean", "median", "most_frequent"]
    

    def impute(self, col, imputer):
        self.train_df.loc[:,col] = imputer.fit_transform(self.train_df[col].values.reshape(-1,1))
        if self.test_df.shape[0] > 0:
            self.test_df.loc[:, col] = imputer.transform(self.test_df[col].values.reshape(-1,1))

    def simple_imputer(self, col, how = "mean"):
        if how in self.strategies:
            imputer = impute.SimpleImputer(strategy=how)
            self.impute(col, imputer)
    

    def iterative_imputer(self, col, how = "mean", ):
        if how in self.strategies:
            imputer = impute.IterativeImputer(initial_strategy=how)
            self.impute(col, imputer)
    
    def knn_imputer(self, col, neighbors = 5):
        imputer = impute.KNNImputer(n_neighbors=neighbors)
        self.impute(col, imputer)



#########################################################################
#--------------------------------FUNCTIONS------------------------------
#########################################################################

def set_index(df, column_name):
    try:
        df.set_index(column_name, inplace = True)
    except NameError:
        print("The specified column is not present in the dataframe.")

def create_y_train(train_df, target_label):
    y_train = train_df.pop(target_label)
    y_train.to_csv(config.train_y)


#########################################################################
#--------------------------------MAIN------------------------------
#########################################################################


if __name__ == "__main__":
    train_df = pd.read_csv(config.train_folds)
    test_df = pd.read_csv(config.test_set)
    
    set_index(train_df, "employee_id")
    set_index(test_df, "employee_id")
    create_y_train(train_df, "is_promoted")
    
    imputeMissing = ImputeMissingValues(train_df, test_df)
    imputeMissing.iterative_imputer("previous_year_rating", "median")
    imputeMissing.simple_imputer("education", "most_frequent")
    train_df.to_csv(config.train_folds)
    test_df.to_csv(config.test_df)
 
