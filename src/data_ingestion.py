#Script to import and store the dataset from Kaggle
import kagglehub
import os
import shutil
import sys

#Pointing to root directory to find config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

import config

# Download latest version
path = kagglehub.dataset_download("ashydv/leads-dataset")

print("Path to dataset files:", path)
os.makedirs(config.RAW_DIR,exist_ok=True)

#Finding and storing the csv file name
filename = [x for x in os.listdir(path) if x.endswith(".csv")][0]

#Copying the dataset to a separate directory named "data"
shutil.copy(src=os.path.join(path,filename),dst=config.RAW_DATA_PATH)

print("Data Fetching SUccessful!")