# Credit Card Fraud Detection

This project is aimed at detecting fraudulent credit card transactions using supervised machine learning models. The dataset includes various attributes related to credit card transactions, and we apply Exploratory Data Analysis (EDA), data preprocessing, feature engineering, and model building using two different classifiers: Random Forest and XGBoost.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Saving & Loading the Model](#saving--loading-the-model)
- [Acknowledgments](#acknowledgments)

## Project Overview

Credit card fraud detection is crucial for financial institutions to minimize the losses that occur due to fraudulent activities. This project uses machine learning models to classify transactions as fraudulent or legitimate based on historical transaction data.

### Key Features:
- **Data Preprocessing**: Time-based feature extraction, handling missing values, one-hot encoding.
- **Modeling**: Random Forest and XGBoost classifiers for predicting fraudulent transactions.
- **Model Evaluation**: Standard metrics such as Accuracy, Precision, Recall, F1 Score, and ROC AUC.
- **Model Persistence**: Saving the trained model using `pickle` for future use.

## Project Structure
```
.
├── fraudTrain.csv               # Training dataset
├── fraudTest.csv                # Test dataset
├── appreciatewealth.py          # Main Python script
├── README.md                    # Project documentation
├── fraud_detection_model.pkl    # Saved model (after running the code)
└── requirements.txt             # Dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/fraud_detection.git
   cd fraud_detection
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

   *Dependencies include*:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - xgboost
   - pickle

3. **Prepare the Dataset**:
   - The project uses two datasets, `fraudTrain.csv` and `fraudTest.csv`. Ensure both are present in the project directory.

## Usage

### Running the Project

1. **Execute the script**:
   ```bash
   python appreciatewealth.py
   ```

   This will perform the following steps:
   - Load the training and testing datasets.
   - Perform Exploratory Data Analysis (EDA) to visualize and understand the data.
   - Preprocess the data, including feature extraction and encoding.
   - Train two different models: Random Forest and XGBoost.
   - Evaluate the models using standard metrics.
   - Save the best-performing model using `pickle`.

### Evaluating the Models

The script trains both the Random Forest and XGBoost models and evaluates their performance based on the following metrics:
- **Accuracy**: Proportion of correctly predicted fraud and legitimate transactions.
- **Precision**: Number of true frauds detected out of all predicted frauds.
- **Recall**: Ability of the model to detect actual fraudulent transactions.
- **F1 Score**: Harmonic mean of precision and recall.
- **ROC AUC Score**: Area under the ROC curve, indicating the model's performance.

## Data Preprocessing

- **Handling Missing Values**: Missing values are replaced with `0` (can be adjusted as per data).
- **Feature Engineering**: Time-based features (hour, day, month) are extracted from transaction dates.
- **One-Hot Encoding**: Categorical columns are converted to numerical representations using one-hot encoding.

## Model Training

Two machine learning models are trained using the preprocessed data:
1. **Random Forest Classifier**: A popular ensemble method based on decision trees.
2. **XGBoost Classifier**: An efficient implementation of gradient-boosted decision trees designed for speed and performance.

## Model Evaluation

The model is evaluated on the test dataset using several metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

After evaluating both models, the XGBoost classifier's trained model is saved as a `.pkl` file using the `pickle` module for future use.

## Saving & Loading the Model

To save the model:
```python
with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)
```

To load the model in the future:
```python
with open('fraud_detection_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
```

## Acknowledgments

This project was developed using open-source datasets and libraries. Thanks to the contributors of `pandas`, `scikit-learn`, `seaborn`, and `xgboost` for providing powerful tools for data science and machine learning.

---
