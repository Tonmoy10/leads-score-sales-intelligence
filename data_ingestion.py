import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("ashydv/leads-dataset")

print("Path to dataset files:", path)

os.makedirs("data/raw",exist_ok=True)

#Finding and storing the csv file name
filename = [x for x in os.listdir(path) if x.endswith(".csv")][0]

#Copying the dataset to a separate directory names "data"
shutil.copy(src=os.path.join(path,filename),dst=os.path.join("data/raw",filename))