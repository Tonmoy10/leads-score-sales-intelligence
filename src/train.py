#Script to train the model on Random Forest Classifier with tuned hyperparameters
import sys
import os
import preprocess
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.cluster import KMeans


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

import config

if __name__=="__main__":
    print("Stage 1: Dataset Loading")
    df = preprocess.load_data(config.RAW_DATA_PATH)

    print("Stage 2: One Hot Encoding")
    X_val,y_val = preprocess.encoded_feat(df)
    X_val = X_val.fillna(0)
    

    print("Stage 3: Splitting train & test data and standardizing it to a single scale")
    X_train,X_test,y_train,y_test,scalar = preprocess.split_and_standardize(X_val,y_val)

    print("Stage 4: Training data on tuned Random Forest Classifier")
    rf = RandomForestClassifier(n_estimators=100,random_state=config.RANDOM_SEED,**config.RF_PARAMS).fit(X_train,y_train)

    print("Stage 5: KMeans Clustering")
    kmeans = KMeans(n_clusters=5,random_state=config.RANDOM_SEED)
    kmeans.fit(X_train)

    print("Stage 6: Saving the random forest model, KMeans model and scalar as .pkl file")
    #Making sure the directory exists
    os.makedirs(config.MODELS_DIR,exist_ok=True)

    #Saving the Random Forest model
    with open(config.MODEL_PATH,"wb") as r:
        pickle.dump(rf,r)

    #Saving the scalar
    with open(config.SCALAR_PATH,"wb") as s:
        pickle.dump(scalar,s)

    #Saving the KMeans model
    with open(config.KMEANS_PATH,"wb") as k:
        pickle.dump(kmeans,k)


    print("Execution Successful!")
