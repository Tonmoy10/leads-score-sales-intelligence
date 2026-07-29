#Script containing all the function definitions required
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

import config

#For loading the data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=config.COLS_TO_DROP)
    return df

#For one hot encoding so models can make sense of string values
def encoded_feat(df):
    X_val = df.drop(columns=config.TARGET_COL,errors="ignore")
    y_val = df[config.TARGET_COL]

    X_encoded = pd.get_dummies(X_val,drop_first=True)
    return X_encoded,y_val

#For defining and standardizing the train and test data on same scale
def split_and_standardize(X_val,y):
    scalar = StandardScaler()
    X_train,X_test,y_train,y_test = train_test_split(X_val,y,test_size=0.20,random_state=config.RANDOM_SEED)

    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    return X_train,X_test,y_train,y_test,scalar