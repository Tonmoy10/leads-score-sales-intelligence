#Script to score the leads on a scale of 1-100
import sys
import os
import pandas as pd
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

import config

if __name__=="__main__":
    #Fetching the dataset
    print("Stage 1: Loading the dataset")
    df = pd.read_csv(config.RAW_DATA_PATH)

    #Using Lead Number as ID to identify leads
    lead_ids = df[["Lead Number"]].copy() if "Lead Number" in df.columns else pd.DataFrame({"Index": df.index})

    #Dropping unwanted columns along with target column
    print("Stage 2: Dropping unwanted columns") 
    df_clean = df.drop(columns=config.COLS_TO_DROP+[config.TARGET_COL],errors="ignore")
    df_enc = pd.get_dummies(data=df_clean,drop_first=True)

    #Loading the Random Forest model
    print("Stage 3: Loading the model and scaler")
    with open(config.MODEL_PATH,"rb") as m:
        rf = pickle.load(m)

    #Loading the StandardScaler
    with open(config.SCALAR_PATH,"rb") as s:
        scalar = pickle.load(s)

    #Re indexing the data according to models original set of features
    print("Stage 4: Indexing the dataset according to the original set of features")
    df_indexed = df_enc.reindex(columns=scalar.feature_names_in_,fill_value=0)
    df_std = scalar.transform(df_indexed)

    #Using the probability to calculate a lead score on a scale of 1-100
    print("Stage 5: Calculating and adding lead scores to the dataset")
    probability = rf.predict_proba(df_std)[:,1]

    lead_ids["Lead Score"] = (probability*100).round(2)
    lead_ids = lead_ids.sort_values(by="Lead Score",ascending=False)

    #Saving the final dataset
    print("Saving the final dataset")
    os.makedirs(config.FINAL_DIR,exist_ok=True)
    lead_ids.to_csv(config.FINAL_DATA_PATH,index=False)



