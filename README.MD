# README: Backend Execution Guide for Anomaly Detection Model

---

## Decision Tree Overview
This repository contains a Decision Tree model for detecting anomalies in station data. The model processes station datasets, extracts features, and predicts anomalies (binary values: 0 for normal, 1 for anomaly). This guide explains how to set up, execute, and utilize the backend model.

---

## Prerequisites
### Software Requirements
- Python 3.8 or higher
- Required Python packages:
  - `pandas`
  - `numpy`
  - `scikit-learn`

### Installation
Install the necessary dependencies using `pip`:


### Load and Initialize the Model
Use the following code to initialize the model and load the pre-trained weights
```python
from model import Model

# Initialize the model
model = Model()
```

### Feature Extraction
The extract_map_features() function processes the Map column in the dataset to extract features like mean_sla, max_sla, min_sla, and std_sla.
```python
# Example usage
import pandas as pd
from model import extract_map_features

# Load a sample dataset
df = pd.read_csv('cleaned_dataset/sample_station.csv')

# Extract features
df['Map_Features'] = df['Map'].apply(extract_map_features)
features_df = pd.DataFrame(df['Map_Features'].tolist())

# Combine features with the main DataFrame
df = pd.concat([df, features_df], axis=1).drop(columns=['Map'])
```

### Predict Anamolies
```python
# Define features for prediction
features = df[['mean_sla', 'max_sla', 'min_sla', 'std_sla']]

# Get predictions
predictions = model.predict(features)

# Add predictions to the DataFrame
df['Prediction'] = predictions
```