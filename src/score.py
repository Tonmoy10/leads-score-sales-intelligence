#Script to score the leads on a scale of 1-100
import sys
import os
import pandas as pd
import pickle
import shap
import numpy as np

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
    df_enc = df_enc.fillna(0)

    #Loading the Random Forest model
    print("Stage 3: Loading the model and scaler")
    with open(config.MODEL_PATH,"rb") as m:
        rf = pickle.load(m)

    #Loading the StandardScaler
    with open(config.SCALAR_PATH,"rb") as s:
        scalar = pickle.load(s)

    #Loading the KMeans model
    with open(config.KMEANS_PATH,"rb") as k:
        kmeans = pickle.load(k)

    #Re indexing the data according to models original set of features
    print("Stage 4: Indexing the dataset according to the original set of features")
    df_indexed = df_enc.reindex(columns=scalar.feature_names_in_,fill_value=0)
    df_std = scalar.transform(df_indexed)

    print("Stage 5: Segemnting and adding lead scores to the dataset")
    #Using the probability to calculate a lead score on a scale of 1-100
    probability = rf.predict_proba(df_std)[:,1]

    lead_ids["Lead Score"] = (probability*100).round(2)

    #Segmenting the data
    clusters = kmeans.predict(df_std)
    cluster_names = {
        0:"Normal Traffic",
        1:"Passive Chat Inquiries",
        2:"Marketing Niche Professionals",
        3:"Window Shoppers",
        4:"High Velocity Conversions"
        }

    lead_ids["Cluster Name"] = [cluster_names[x] for x in clusters]

    #SHAP Explanations
    print("Stage 6: Generating SHAP Explanations")
    high_scores = lead_ids["Lead Score"]>50
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(df_std[high_scores])

    #Different versions of SHAP store shap values differenly so this block checks is its a list and acts accordingly
    if isinstance(shap_values,list):
        pos_val = shap_values[1]
    else:
        pos_val = shap_values[:,:,1]

    #Storing the top driving feature
    shap_df = pd.DataFrame(pos_val, columns=scalar.feature_names_in_)
    lead_ids["Top Driving Feature"] = "N/A - Low Score"
    lead_ids.loc[high_scores,"Top Driving Feature"] = shap_df.idxmax(axis=1).values


    #Saving the final dataset
    print("Saving the final dataset")
    lead_ids = lead_ids.sort_values(by="Lead Score", ascending=False)
    os.makedirs(config.FINAL_DIR,exist_ok=True)
    lead_ids.to_csv(config.FINAL_DATA_PATH,index=False)

    print("Clustering and Scoring Complete!")



