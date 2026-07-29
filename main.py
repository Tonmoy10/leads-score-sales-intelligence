#Run python main.py train to train the model and run python main.py to score the model
import subprocess
import sys

#To train the model
if len(sys.argv) > 1 and sys.argv[1]=="train":
    print("Start Training!")
    subprocess.run([sys.executable,"src/data_ingestion.py"])
    subprocess.run([sys.executable,"src/train.py"])
    print("Training Complete")
#To score the model
else:
    print("Start Scoring!")
    subprocess.run([sys.executable,"src/data_ingestion.py"])
    subprocess.run([sys.executable,"src/score.py"])
    print("Scoring Complete")
