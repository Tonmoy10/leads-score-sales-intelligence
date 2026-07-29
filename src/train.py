#Script to train the model on Random Forest Classifier with tuned hyperparameters
import sys
import os
import preprocess
from sklearn.ensemble import RandomForestClassifier
import pickle


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

import config

if __name__=="__main__":
    print("Stage 1: Dataset Loading")
    df = preprocess.load_data(config.RAW_DATA_PATH)

    print("Stage 2: One Hot Encoding")
    X_val,y_val = preprocess.encoded_feat(df)

    print("Stage 3: Splitting train & test data and standardizing it to a single scale")
    X_train,X_test,y_train,y_test,scalar = preprocess.split_and_standardize(X_val,y_val)

    print("Stage 4: Training data on tuned Random Forest Classifier")
    rf = RandomForestClassifier(n_estimators=100,random_state=config.RANDOM_SEED,**config.RF_PARAMS).fit(X_train,y_train)

    print("Stage 5: Saving the model and scalar as .pkl file")
    #Making sure the directory exists
    os.makedirs(config.MODELS_DIR,exist_ok=True)
    #Saving the model
    with open(config.MODEL_PATH,"wb") as f:
        pickle.dump(rf,f)
    #Saving the scalar
    with open(config.SCALAR_PATH,"wb") as f:
        pickle.dump(scalar,f)

    print("Execution Successful!")
