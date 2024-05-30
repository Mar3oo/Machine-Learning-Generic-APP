import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

def perform_eda(data):
    st.header("Exploratory Data Analysis (EDA)")
    analyze_data = st.checkbox("Perform EDA?")
    if analyze_data:
        columns_to_analyze = st.multiselect("Select columns for analysis:", options=data.columns)
        if columns_to_analyze:
            st.subheader("Histograms")
            for col in columns_to_analyze:
                plt.figure(figsize=(8, 6))
                sns.histplot(data[col], kde=True)
                plt.title(f"Histogram for {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                st.pyplot()
            st.subheader("Correlation Matrix")
            corr = data[columns_to_analyze].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Correlation Matrix")
            st.pyplot()

def encode_categorical(data):
    categorical_features = data.select_dtypes(include=['object']).columns
    encoding_method = st.radio("Select encoding method for categorical data:", ("Label Encoding", "One-Hot Encoding"))
    if encoding_method == "Label Encoding":
        label_encoders = {}
        for col in categorical_features:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])
    elif encoding_method == "One-Hot Encoding":
        data = pd.get_dummies(data, columns=categorical_features)
    return data

def choose_variables(data):
    st.header("Choose X and Y variables")
    X_variables = st.multiselect("Select independent variables (X):", options=data.columns)
    Y_variable = st.selectbox("Select dependent variable (Y):", options=data.columns)
    return X_variables, Y_variable

def main():
    st.sidebar.header("Steps to get the algorithms prediction accuracy")
    st.sidebar.text("1- Upload CSV or Excel file")
    st.sidebar.text("2- Choose target feature")
    st.sidebar.text("3- Remove unimportant features")

    data = pd.DataFrame()
    target = ""

    dataset = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
    if dataset is not None:
        if "csv" in dataset.name:
            data = pd.read_csv(dataset)
        elif "xlsx" in dataset.name:
            data = pd.read_excel(dataset)
        st.write(data.head())
        st.write(data.shape)

        target = st.selectbox("Choose the target variable:", options=data.columns)

        perform_eda(data)

        data = encode_categorical(data)

        select_columns = st.multiselect("Select features to remove from the dataframe:", options=data.columns)
        if select_columns:
            data.drop(select_columns, axis=1, inplace=True)

        X_variables, Y_variable = choose_variables(data)

        numerical_features = data.select_dtypes(['int64', 'float64']).columns
        categorical_feature = data.select_dtypes(['object']).columns
        missing_value_num = st.radio("Set missing value for numerical value 👇", ["mean", "median"])
        missing_value_cat = st.radio("Set missing value for categorical value 👇", ['most frequent', "put additional class"])

        for col in numerical_features:
            data[col] = SimpleImputer(strategy=missing_value_num, missing_values=np.nan).fit_transform(
                data[col].values.reshape(-1, 1))
        for col in categorical_feature:
            if data[col].nunique() > 7:
                data[col] = SimpleImputer(strategy='most_frequent', missing_values=np.nan).fit_transform(
                    data[col].values.reshape(-1, 1))
            else:
                data[col] = LabelEncoder().fit_transform(data[col])

        if (len(numerical_features) != 0):
            st.header("Numerical Columns")
            st.write(numerical_features)
        if (len(categorical_feature) != 0):
            st.header("Categorical columns")
            st.write(categorical_feature)
        if (len(categorical_feature) != 0 or len(numerical_features) != 0):
            st.header("Number of null values")
            st.write(data.isna().sum())


        def normalize_features(X):
          scaler = StandardScaler()
          X_normalized = scaler.fit_transform(X)
          return X_normalized 

        X_normalized_classification = normalize_features(data[X_variables])
        X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(X_normalized_classification, data[Y_variable], test_size=0.2, random_state=42)

        X_normalized_regression = normalize_features(data[X_variables])
        X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_normalized_regression, data[Y_variable], test_size=0.2, random_state=42)
        st.write("Training and testing sets created successfully!")

        def train_and_evaluate_classification(X_train_classification, X_test_classification, y_train_classification, y_test_classification, selected_model):
            st.header("Training Model for Classification: " + selected_model)
            model = classification_models[selected_model]
            st.write(f"Training {selected_model}...")
            model.fit(X_train_classification, y_train_classification)
            st.write(f"{selected_model} trained successfully!")

            st.header("Model Evaluation for Classification: " + selected_model)
            st.write(f"Evaluating {selected_model}...")
            y_pred = model.predict(X_test_classification)
            accuracy = accuracy_score(y_test_classification, y_pred)
            report = classification_report(y_test_classification, y_pred)
            st.write(f"Accuracy of {selected_model}: {accuracy}")
            st.header(f"Classification Report of {selected_model}:\n{report}")

        def train_and_evaluate_regression(X_train_regression, X_test_regression, y_train_regression, y_test_regression, selected_model):
            st.header("Training Model for Regression: " + selected_model)
            model = regression_models[selected_model]
            st.write(f"Training {selected_model}...")
            model.fit(X_train_regression, y_train_regression)
            st.write(f"{selected_model} trained successfully!")

            st.header("Model Evaluation for Regression: " + selected_model)
            st.write(f"Evaluating {selected_model}...")
            y_pred = model.predict(X_test_regression)
            r2=r2_score(y_test_regression, y_pred)
            mse = mean_squared_error(y_test_regression, y_pred)
            st.write(f"r2 of {selected_model}: {r2}")
            st.write(f"MSE of {selected_model}: {mse}")

        st.title("Machine Learning Model Training and Evaluation")

        task = st.selectbox("Select Task", ["Classification", "Regression"])

        X_variables = []

        classification_models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier()
        }

        regression_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Lasso Regression": Lasso(),
            "Ridge Regression": Ridge()
        }

        if task == "Classification":
            st.header("Classification Task")
            selected_model = st.radio("Select Classification Model", list(classification_models.keys()))
            if st.button("Train and Evaluate"):
                train_and_evaluate_classification(X_train_classification, X_test_classification, y_train_classification, y_test_classification, selected_model)

        elif task == "Regression":
            st.header("Regression Task")
            selected_model = st.radio("Select Regression Model", list(regression_models.keys()))
            if st.button("Train and Evaluate"):
                train_and_evaluate_regression(X_train_regression, X_test_regression, y_train_regression, y_test_regression, selected_model)
if __name__ == "__main__":
    main()