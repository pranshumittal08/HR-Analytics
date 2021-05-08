from sklearn import preprocessing
from sklearn import feature_selection
import pandas as pd
import numpy as np
import config

#########################################################################
#--------------------------------CLASSES------------------------------
#########################################################################

#4. Create label encodings
class Encodings:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
    
    def label_encode(self, cols):
        
        for col in cols:
            labels = preprocessing.LabelEncoder()
            labels.fit(self.train_df[col].values)
            self.train_df.loc[:,col] = labels.transform(self.train_df[col].values)
            self.test_df.loc[:,col] = labels.transform(self.test_df[col].values)
    
    def one_hot_encode(self):
        
        encoder = preprocessing.OneHotEncoder(drop = "first", sparse = True)

        for i, df in enumerate([self.train_df, self.test_df]):
            if i == 0:
                one_hot_matrix = encoder.fit_transform(df.select_dtypes(exclude= "number").values).toarray()

            else:
                one_hot_matrix = encoder.transform(df.select_dtypes(exclude= "number").values).toarray()
            encoder_features = encoder.get_feature_names()
            one_hot_df = pd.DataFrame(one_hot_matrix, index = df.index, columns = encoder_features)
            df  = df.join(one_hot_df)
            if i == 0:
                self.train_df.loc[:, df.columns] = df
                self.train_df.drop(columns = self.train_df.select_dtypes(include = "object").columns, inplace = True)
            else:
                self.test_df.loc[:, df.columns] = df
                self.test_df.drop(columns = self.test_df.select_dtypes(include = "object").columns, inplace = True)
            

#Create Categorical features using already existring cat features
class Categorical:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.cat_cols = list(self.train_df.select_dtypes(exclude = "number").columns)
    
    def create_features(self):
        for col in self.cat_cols:
            for col_2 in self.cat_cols[self.cat_cols.index(col)+1:]:
                new_col = "_".join([col, col_2])
                self.train_df[new_col] = self.train_df[col]+"_" +self.train_df[col_2]
                self.test_df[new_col]  = self.test_df[col] + "_" + self.test_df[col_2]

    

class Aggregate:

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

    def create_features(self, metric, cols):
        new_df = pd.DataFrame(index= self.train_df.index)
        for col in cols:
            grouped = self.train_df.groupby(col)
            agg_df = grouped.aggregate(metric)
            agg_cols = agg_df.columns
            # Fix error in below line
            transformed_df = grouped[agg_cols].transform(metric).rename(columns = lambda x : "_".join([metric, col, x]))
            
            # new_df = pd.DataFrame(index= self.train_df.index)
            new_df.loc[:, transformed_df.columns] = transformed_df.values
            
            for col_2 in agg_cols:
                mapping = dict(agg_df[col_2])
                new_col_name = "_".join([metric, col, col_2])
                
                self.test_df.loc[:,new_col_name] = self.test_df[col].map(mapping)

        self.train_df.loc[:, new_df.columns] = new_df.values

class Polynomial:

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        
    
    def create_features(self):
        poly = preprocessing.PolynomialFeatures(include_bias = False)
        
        poly.fit(self.train_df.select_dtypes(include = "number"))
        
        self.train_df.loc[:, poly.get_feature_names()] = poly.transform(self.train_df.select_dtypes(include = "number"))
        
        self.test_df.loc[:, poly.get_feature_names()] = poly.transform(self.test_df.select_dtypes(include = "number"))

class Scale:

    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
    
    def fit_transform(self, scaler):
        cols = self.train_df.select_dtypes(include = "number").columns
        self.train_df.loc[:, cols] = scaler.fit_transform(self.train_df[cols])
        self.test_df.loc[:, cols] = scaler.transform(self.test_df[cols])
    
    def min_max(self):
        scaler = preprocessing.MinMaxScaler()
        self.fit_transform( scaler)

    def standard(self):
        scaler = preprocessing.StandardScaler()
        self.fit_transform( scaler)

    def quantile(self):
        scaler = preprocessing.QuantileTransformer()
        self.fit_transform( scaler)

    def robust(self):
        scaler = preprocessing.RobustScaler()
        self.fit_transform( scaler)


#########################################################################
#--------------------------------FUNCTIONS------------------------------
#########################################################################





#########################################################################
#--------------------------------MAIN------------------------------
#########################################################################

if __name__ == "__main__":
    
    train_df = pd.read_csv(config.train_folds, index_col = "employee_id")
    test_df = pd.read_csv(config.test_df, index_col = "employee_id")
    
    folds_col = train_df.pop("kfold")
    #Steps
    # 1. Create cat features
    # cat_obj = Categorical(train_df, test_df)
    # cat_obj.create_features()

    # 2. Create aggregates
    agg_object = Aggregate(train_df, test_df)
    cols_for_agg = train_df.columns[train_df.nunique() < 10]
    agg_object.create_features("mean", cols_for_agg)

    #3. Scaling the features
    scale_obj = Scale(train_df, test_df)
    scale_obj.min_max()

    # 4. Create polynomial features
    #poly_obj = Polynomial(train_df, test_df)
    #poly_obj.create_features()

    # 3. Create encodings
    encode_obj = Encodings(train_df, test_df)
    encode_obj.label_encode(['education'])
    encode_obj.one_hot_encode()

    

    train_df.loc[:, "kfold"] = folds_col

    train_df.to_csv(config.train_X)
    test_df.to_csv(config.test_df)
 