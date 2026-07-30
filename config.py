import os

#Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Main Directory
DATA_DIR = os.path.join(BASE_DIR,"data")
MODELS_DIR = os.path.join(BASE_DIR,"models")

#Sub-Directory
RAW_DIR = os.path.join(DATA_DIR,"raw") #For raw dataset
INTERIM_DIR = os.path.join(DATA_DIR,"interim") #For cleaned dataset
PROCESSED_DIR = os.path.join(DATA_DIR,"processed") #For ML processed dataset
SEGMENTED_DIR = os.path.join(DATA_DIR,"segmented") #For clustered dataset
FINAL_DIR = os.path.join(DATA_DIR,"final") #For final dataset

#Data Path
RAW_DATA_PATH = os.path.join(RAW_DIR,"Leads.csv")
INTERIM_DATA_PATH = os.path.join(INTERIM_DIR,"Cleaned_Leads.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR,"Processed_Leads.csv")
SEGMENTED_DATA_PATH = os.path.join(SEGMENTED_DIR,"Segmented_Leads.csv")
FINAL_DATA_PATH = os.path.join(FINAL_DIR,"Final_Leads.csv")

#Model Path
MODEL_PATH = os.path.join(MODELS_DIR,"tuned_rf_model.pkl")

#Scalar Path
SCALAR_PATH = os.path.join(MODELS_DIR,"standard_scaler.pkl")

#KMeans Path
KMEANS_PATH = os.path.join(MODELS_DIR,"kmeans_model.pkl")

#Defining random seed to keep scripts deterministic
RANDOM_SEED = 42

#Defining target column
TARGET_COL = "Converted"

#Defining columns to drop to clean dataset
COLS_TO_DROP = ["Lead Quality","Lead Profile","How did you hear about X Education","Asymmetrique Activity Index", "Last Notable Activity",
                "Asymmetrique Profile Index","Asymmetrique Activity Score","Asymmetrique Profile Score","Prospect ID","City"]

#Parameter tuning for Random Forest
RF_PARAMS = {
    "max_depth": None,
    "min_samples_leaf": 2,
    "min_samples_split": 5
}