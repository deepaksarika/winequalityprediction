import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import io

import warnings
warnings.filterwarnings('ignore')

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv('winequality.csv')
    return df

# Load dataset
df = load_data()

# Sidebar options
st.sidebar.title("Wine Quality Analysis")
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Wine Quality Dataset")
    st.write(df.head())

# Display dataset information
st.sidebar.subheader("Dataset Info")
if st.sidebar.checkbox("Show data info"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

if st.sidebar.checkbox("Show data description"):
    st.subheader("Dataset Description")
    st.write(df.describe().T)

if st.sidebar.checkbox("Show missing values"):
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# Fill missing values with column mean
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

# Encode categorical variables before correlation matrix
df.replace({'white': 1, 'red': 0}, inplace=True)

# Data visualization
st.sidebar.subheader("Visualizations")
if st.sidebar.checkbox("Show data distributions"):
    st.subheader("Data Distributions")
    fig, ax = plt.subplots(figsize=(10, 10))
    df.hist(bins=20, ax=ax)
    st.pyplot(fig)

if st.sidebar.checkbox("Show alcohol content by quality"):
    st.subheader("Alcohol Content by Quality")
    fig, ax = plt.subplots()
    ax.bar(df['quality'], df['alcohol'])
    ax.set_xlabel('Quality')
    ax.set_ylabel('Alcohol')
    st.pyplot(fig)

if st.sidebar.checkbox("Show heatmap of correlations"):
    st.subheader("Heatmap of Correlations")
    fig, ax = plt.subplots(figsize=(12, 12))
    sb.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Drop highly correlated features
df = df.drop('total sulfur dioxide', axis=1)

# Create binary target variable
df['best quality'] = df['quality'].apply(lambda x: 1 if x > 5 else 0)

# Define features and target
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

# Split data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40
)

# Normalize features
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Initialize models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

# Train and evaluate models
model_names = ["Logistic Regression", "XGBClassifier", "SVC"]
for model_name, model in zip(model_names, models):
    model.fit(xtrain, ytrain)
    train_pred = model.predict(xtrain)
    test_pred = model.predict(xtest)

    st.write(f'{model_name} :')
    st.write('Training ROC AUC:', metrics.roc_auc_score(ytrain, train_pred))
    st.write('Validation ROC AUC:', metrics.roc_auc_score(ytest, test_pred))
    st.write()

# Plot confusion matrix for best model (XGBClassifier)
st.subheader('Confusion Matrix - XGBClassifier')
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(models[1], xtest, ytest, ax=ax)
st.pyplot(fig)

# Classification report for best model
st.subheader('Classification Report - XGBClassifier')
st.text(metrics.classification_report(ytest, models[1].predict(xtest)))
