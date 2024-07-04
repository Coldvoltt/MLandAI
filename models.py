import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import statsmodels.api as sm


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, is_classification):
    if is_classification:
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, is_classification):
    if is_classification:
        model = SVC()
    else:
        model = SVR()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, is_classification):
    if is_classification:
        model = DecisionTreeClassifier()
    else:
        model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, is_classification):
    if is_classification:
        model = KNeighborsClassifier()
    else:
        model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, is_classification):
    if is_classification:
        model = GradientBoostingClassifier()
    else:
        model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model


def evaluate_classification_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    return conf_matrix, class_report


def evaluate_regression_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    # Add a constant to the model for statsmodels
    X_test_const = sm.add_constant(X_test)
    sm_model = sm.OLS(y_test, X_test_const)
    results = sm_model.fit()

    # Create a regression results table
    regression_table = pd.DataFrame({
        "Metric": ["Mean Squared Error", "R Squared", "F-statistic", "P-value"],
        "Value": [mse, r_squared, results.fvalue, results.f_pvalue]
    })

    return mse, r_squared, regression_table
