import pandas as pd
import numpy as np
from preprocess import remove_id_columns
from sklearn.model_selection import train_test_split
from models import (
    train_linear_regression,
    train_random_forest,
    train_svm,
    train_decision_tree,
    train_knn,
    train_gradient_boosting,
    evaluate_classification_model,
    evaluate_regression_model
)


# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = remove_id_columns(data)
    return data

# Function to train and evaluate models


def train_and_evaluate_model(independent_var, model_choice):
    path = "studentPerformance.csv"
    data = load_data(path)

    X = data.drop(columns=[independent_var])
    y = data[independent_var]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123)

    # Assuming classification if unique values <= 5
    is_classification = y.nunique() <= 5

    model = None
    results = None
    feature_importance = None

    if model_choice == "Linear Regression":
        model = train_linear_regression(X_train, y_train)
        mse, r_squared, results = evaluate_regression_model(
            model, X_test, y_test)
    elif model_choice == "Random Forest":
        model = train_random_forest(X_train, y_train, is_classification)
        if is_classification:
            conf_matrix, class_report = evaluate_classification_model(
                model, X_test, y_test)
            results = (conf_matrix, class_report)
        else:
            mse, r_squared, results = evaluate_regression_model(
                model, X_test, y_test)
        feature_importance = model.feature_importances_
    elif model_choice == "Support Vector Machine":
        model = train_svm(X_train, y_train, is_classification)
        if is_classification:
            conf_matrix, class_report = evaluate_classification_model(
                model, X_test, y_test)
            results = (conf_matrix, class_report)
        else:
            mse, r_squared, results = evaluate_regression_model(
                model, X_test, y_test)
    elif model_choice == "Decision Tree":
        model = train_decision_tree(X_train, y_train, is_classification)
        if is_classification:
            conf_matrix, class_report = evaluate_classification_model(
                model, X_test, y_test)
            results = (conf_matrix, class_report)
        else:
            mse, r_squared, results = evaluate_regression_model(
                model, X_test, y_test)
        feature_importance = model.feature_importances_
    elif model_choice == "K-Nearest Neighbors":
        model = train_knn(X_train, y_train, is_classification)
        if is_classification:
            conf_matrix, class_report = evaluate_classification_model(
                model, X_test, y_test)
            results = (conf_matrix, class_report)
        else:
            mse, r_squared, results = evaluate_regression_model(
                model, X_test, y_test)
    elif model_choice == "Gradient Boosting":
        model = train_gradient_boosting(X_train, y_train, is_classification)
        if is_classification:
            conf_matrix, class_report = evaluate_classification_model(
                model, X_test, y_test)
            results = (conf_matrix, class_report)
        else:
            mse, r_squared, results = evaluate_regression_model(
                model, X_test, y_test)
        feature_importance = model.feature_importances_

    if feature_importance is not None:
        feature_importance = pd.Series(
            feature_importance, index=X.columns).sort_values(ascending=False)

    return results, feature_importance


# results, importance = train_and_evaluate_model("GradeClass", "Random Forest")
# # print("Model Result:", results)
# # if importance is not None:
# #     print("Feature Importance:\n", importance)
