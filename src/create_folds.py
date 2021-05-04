import pandas as pd
import numpy as np
from sklearn import model_selection
import config
import os

if __name__ == '__main__':
    print(os.getcwd())
    #Reading data from csv files
    df = pd.read_csv(config.train_path)

    df['kfold'] = -1
    
    df = df.sample(frac = 1).reset_index(drop = True)

    kf = model_selection.StratifiedKFold(n_splits = 5)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, df.is_promoted.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold

    df.to_csv("input/train_folds.csv", index = False)

