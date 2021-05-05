import numpy as np
import pandas as pd
from sklearn import impute
from sklearn.experimental import enable_iterative_imputer

def set_index(df, column_name):
    try:
        df.set_index(column_name, inplace = True)
    except NameError:
        print("The specified column is not present in the dataframe.")
     
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
        if self.test_df != None:
            self.test_df.loc[:, col] = imputer.transform(self.test_df[col].values.reshape(-1,1))

    def simple_impute(self, col, how = "mean"):
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

    
if __name__ == "__main__":
    train_df = pd.read_csv("input/train_folds.csv")
    print(train_df.isna().sum())
    
    set_index(train_df, "employee_id")

    imputeMissing = ImputeMissingValues(train_df)
    imputeMissing.iterative_imputer("previous_year_rating")
    print(train_df.isna().sum())
