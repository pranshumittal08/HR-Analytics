from sklearn import preprocessing
import pandas as pd

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
                self.train_df = df
            else:
                self.test_df = df

#Create Categorical features using already existring cat features
class CreateCategoricalFeatures():
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.cols = self.train_df.select_dtypes(exclude = "number")
    
    def create_features(self):
        for col in self.cols:
            for col in self.cols[self.cols.index(col)+1:]:
                new_
                self.train_df[]


if __name__ == "__main__":
    train_df = pd.read_csv("input/train_set.csv")
    test_df = pd.read_csv("input/test_set.csv")
    encode_obj = Encodings(train_df, test_df)
    print(train_df.shape)
    encode_obj.one_hot_encode()
    train_df = encode_obj.train_df
    print(train_df.head)


# def create_cat_features(self):

#1. Create polynomial features

#2. Create aggregate features using groupby
## How to deal with the features for test dataset ???

#3. Create new categorical features by combining multiple features
## After creating cat features, encode them using one hot or binary encoding
## Convert to sparse matrix using the scipy library



#5. Remove features with low variance or standard deviation

#6. Drop highly correlated features

#7. Then drop features using recursive feature elimination
"""
order of -
 create new cat vars
"""
 