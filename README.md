
## Overview

This project aims to build a sales prediction system using machine learning. The primary focus of this code is on automated feature enrichment via the Upgini service, which seamlessly adds external features to the existing sales records.

---

## 1. Technology Highlights

This project leverages several key technologies and libraries within the Python ecosystem:

* **Pandas**: Extensively used for loading CSV datasets, data type manipulation (especially converting strings to datetime objects), managing random data sampling, and sorting dataframes chronologically.
* **Scikit-Learn (sklearn)**: Imported for data splitting management (although in this specific file, the split between train and test data is done manually using date filtering).
* **Upgini**: An AutoFE (Automated Feature Engineering) tool. This library algorithmically searches for public and external data relevant to the foundational data of this project.
* **CatBoost**: A machine learning library based on the Gradient Boosting algorithm that excels with tabular data. It is set up from the start (via pip installation) as the target model for training the data.

---

## 2. Impact of the Technologies Used

* **Effortless Information Expansion**: Upgini makes a massive impact by transforming basic attributes (relying solely on dates) into a highly enriched dataset. It automatically pulls financial market data (such as average crude oil prices, gold, and stock market indices) as well as weather/climate data to serve as additional feature references for sales patterns.
* **Time Efficiency**: Conventional workflows require data scientists to manually find, clean, and merge external data from various APIs. Implementing this script radically shortens the Feature Engineering phase to just a few lines of code.
* **Mathematical Performance Explainability**: The script generates "SHAP values" for each discovered data source (for example, weather data is shown to have a high SHAP value influence in Upgini's breakdown). This provides a significant advantage for the explainability of the model's predictive performance.

---

## 3. How to Use / Run Guide

Below are the step-by-step instructions to run this project in environments like Jupyter Notebook or Google Colab:

**Step 1: Library Setup**
First, run the following command to install and load all the essential Python packages into your environment:

```python
import pandas as pd
%pip install -Uq upgini catboost

```

**Step 2: Loading the Dataset and Preprocessing**
Ensure that the source data file named `train.csv` is located in the same directory. The code will extract a random sample of 10,000 rows and convert the date strings into datetime objects so that the transaction sequence is handled correctly:

```python
df = pd.read_csv('train.csv')
df = df.sample(n=10_000, random_state=0)
df["store"] = df["store"].astype(str)
df["item"] = df["item"].astype(str)
df["date"] = pd.to_datetime(df["date"])
df.sort_values("date", inplace=True)
df.reset_index(inplace=True, drop=True)

```

**Step 3: Train-Test Split**
The system uses dates to separate the training phase from the actual testing/prediction phase. The data is split starting from `2017-01-01` for training and from `2017-01-10` for testing:

```python
train = df[df['date'] < "2017-01-01"]
test = df[df['date'] >= '2017-01-10']

train_features = train.drop('sales', axis=1)
train_target = train['sales']

test_features = test.drop("sales", axis=1)
test_target = test['sales']

```

**Step 4: Running the "Feature Enricher"**
The primary search key used here is the date. You need to initialize the Upgini function to merge external features for training and evaluate it using the test set by running this code:

```python
from upgini import FeaturesEnricher, SearchKey
from upgini.metadata import CVType

enricher = FeaturesEnricher(
    search_keys={
        "date" : SearchKey.DATE,
    },
    cv = CVType.time_series
)
enricher.fit(train_features, train_target, eval_set=[(test_features, test_target)])

```

Wait for the progress bar to reach 100%. An output table will then be displayed, providing insights into the new set of features that will be incorporated into the AI's learning process.
