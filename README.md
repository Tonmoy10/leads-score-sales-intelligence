# Lead Scoring and Sales Intelligence Pipeline

## Overview
An automated pipeline that scores inbound leads by conversion probability and groups them into behavioural segments. It runs as a daily batch job. New leads come in, get cleaned and scored, and the output feeds a Power BI dashboard that sales reps use to decide who to call first.

A Random Forest model predicts conversion probability and converts it into a 1 to 100 lead score. A K-Means model runs alongside it to group leads into five behavioural personas. SHAP explains which feature drove each high-scoring lead's prediction, so a rep has a reason to open the call with, not just a number.

## Stack
- Data processing: Python, Pandas, NumPy
- Modelling: Scikit-learn (Random Forest, K-Means)
- Explainability: SHAP
- Dashboard: Power BI

## How it works

**Preprocessing.** Raw lead data is cleaned, missing values are handled, categorical fields are one-hot encoded, and numerical fields are scaled. Column alignment is enforced with reindex() so new data always matches the shape the model was trained on.

**Lead scoring.** A Random Forest classifier estimates conversion probability from historical lead behaviour. That probability becomes a 1 to 100 score.

**Persona segmentation.** A K-Means model with five clusters groups leads by behavioural pattern. One example is a low-intent "window shopper" segment against a high-conversion segment. This runs independently of the scoring model, on the same underlying data.

**Explainability.** For leads scoring above 50, SHAP identifies the single feature contributing most to that score. Leads below the threshold are skipped to keep the daily run fast. Explaining a lead nobody's going to call isn't worth the compute.

## Repository structure
notebooks/ exploratory work behind the pipeline

**01_data_cleaning.ipynb:** initial data cleaning and inspection

**02_data_preprocessing.ipynb:** encoding, scaling, feature prep for modelling

**03_clustering.ipynb:** K-Means exploration and cluster definition

**04_lead_scoring.ipynb:** Random Forest and logistic regression training and scoring

src/

**data_ingestion.py:** pulls the raw data from Kaggle

**preprocess.py:** shared cleaning and feature engineering functions

**train.py:** trains the tuned Random Forest

**score.py:** runs clustering and scores new leads

**main.py:** entry point, routes to train.py or score.py

**config.py:** paths and parameters

The notebooks were where the actual modelling decisions got made: which features mattered, how many clusters made sense, what the tuned Random Forest parameters should be. The scripts in src/ are the production version of those same decisions, rebuilt to run without anyone opening a notebook.

## Dashboard
The scored output feeds a two-page Power BI dashboard.

The SDR view is a sorted list of leads with score, persona, and top driving feature, so reps know who to call and what to say. The leadership view shows lead volume and average score by persona, plus conversion rate by lead source, to show where marketing spend is and isn't working.

## Running it

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the models. This cleans the data, fits the Random Forest and K-Means models, fits the scaler, and saves everything to models/:
```bash
python main.py train
```

Run the daily pipeline. This scores new leads, assigns personas, runs SHAP on leads above the threshold, and exports the result for Power BI:
```bash
python main.py
```

## Results
The Random Forest reached 93% overall accuracy. On the converted class specifically, precision was 0.95 and recall was 0.88, giving an F1 of 0.91. High precision matters more than raw accuracy here, since a false positive costs a rep's time on a call that was never going to convert.

Of all inbound leads, only 36.44% scored above the actionable threshold of 50. That number alone is a useful read on how much of the top of the funnel is worth sales attention versus how much needs a marketing fix instead.

The clustering showed a segment nicknamed "window shoppers," high in volume but low in conversion intent, as the largest single group of inbound traffic. That's a specific, actionable finding for whoever owns lead generation, not just a modelling exercise.