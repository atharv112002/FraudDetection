import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import pickle

# Load the datasets
fraud_train = pd.read_csv('fraudTrain.csv')
fraud_test = pd.read_csv('fraudTest.csv')


# EDA - Exploratory Data Analysis
def perform_eda(data):
    # Basic info
    print(data.info())
    print(data.describe())

    # Checking class imbalance
    print('Class distribution:')
    print(data['is_fraud'].value_counts())

    # Visualization: Fraud distribution
    sns.countplot(x='is_fraud', data=data)
    plt.title('Fraud vs Legitimate Transaction Distribution')
    plt.show()

    # Visualization: Transaction Amount Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data['transactionAmount'], bins=50)
    plt.title('Transaction Amount Distribution')
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


# Perform EDA on the training data
perform_eda(fraud_train)


# Data Preprocessing
def preprocess_data(data):
    # Handle missing values
    data = data.fillna(0)  # Filling missing values (if any) with 0

    # Feature Engineering: Example time-based features from transaction timestamps
    data['transactionDate'] = pd.to_datetime(data['transactionDate'])
    data['hour'] = data['transactionDate'].dt.hour
    data['day'] = data['transactionDate'].dt.day
    data['month'] = data['transactionDate'].dt.month

    # Drop irrelevant columns
    drop_cols = ['transactionID', 'transactionDate', 'customerID', 'merchantID']  # Adjust based on your dataset
    data = data.drop(columns=drop_cols, errors='ignore')

    # One-hot encoding for categorical features (if needed)
    data = pd.get_dummies(data, drop_first=True)

    return data


# Preprocess the train and test data
train_data = preprocess_data(fraud_train)
test_data = preprocess_data(fraud_test)

# Splitting the data into features and target
X_train = train_data.drop('isFraud', axis=1)
y_train = train_data['isFraud']
X_test = test_data.drop('isFraud', axis=1)
y_test = test_data['isFraud']

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building: Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Model Evaluation on Test Data
y_pred_rf = rf_model.predict(X_test_scaled)


# Evaluation Metrics
def evaluate_model(y_true, y_pred):
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
    print(f'Precision: {precision_score(y_true, y_pred)}')
    print(f'Recall: {recall_score(y_true, y_pred)}')
    print(f'F1 Score: {f1_score(y_true, y_pred)}')
    print(f'ROC AUC Score: {roc_auc_score(y_true, y_pred)}')


# Evaluate Random Forest
print("Random Forest Performance:")
evaluate_model(y_test, y_pred_rf)

# Model Building: XGBoost Classifier
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predictions with XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate XGBoost
print("\nXGBoost Performance:")
evaluate_model(y_test, y_pred_xgb)

# Saving the best model using pickle
with open('fraud_detection_model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

# Load the model (for future use)
# with open('fraud_detection_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)
