# for the class DataFrameSelector I used the following website https://stackoverflow.com/questions/48491566/name-dataframeselector-is-not-defined

import csv
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


class DataFrameSelector(BaseEstimator, TransformerMixin):
    # I used the following website for this class: https://stackoverflow.com/questions/48491566/name-dataframeselector-is-not-defined
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def preprocess_data(df):
    """
    Preprocess the data.
    Categorical features will go to a OneHotEncoder.
    Numerical features will go to a StandardScaler
    :param df: pandas dataframe
    :return: the fitted pipline for preprocessing
    """
    num_attribs = ['kitchens', 'bathrooms', 'rooms']
    cat_attribs = ['type', 'condition', 'elevator', 'subway', 'district', 'recentOwner']
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('one_hot_encoder', OneHotEncoder(categories="auto", handle_unknown="ignore")),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    prepared_data = full_pipeline.fit_transform(df)
    return prepared_data, full_pipeline


def train_data(modeltype, data, solution):
    """
    Train model with given data.
    :param modeltype: which sklearn model
    :param data: training data
    :param solution: solution for the data
    :return: model
    """
    model = modeltype()
    model.fit(data, solution)
    return model


def split_data(df, solution_feature=None):
    """
    Remove features that are not needed and split to data and solutions.
    :param df: pandas dataframe
    :param solution_feature: bool: if true there is no solution in the dataframe
    :return: list [ data, solutions
    """
    if solution_feature is None:
        data = df.drop(["identifier", "latitude", "longitude"], axis=1).copy()
        return data
    data = df.drop(["identifier", "latitude", "longitude", 'prediction', "highValue"], axis=1).copy()
    solutions = df[solution_feature].copy()
    return data, solutions


def choose_algortihm(data, solutions):
    """
    For test purposes print results of different models.
    :param data: dataframe
    :param solutions: solutions for data
    :return:
    """
    algorithms = {}
    for algo in [DecisionTreeClassifier, GradientBoostingClassifier, RandomForestClassifier, SGDClassifier,
                 RidgeClassifier, BernoulliNB]:
        model = algo()
        algorithms[model.__class__.__name__] = model
        print(model.__class__.__name__)
        print(cross_val_score(model, data, solutions, cv=5).mean())
        predictions = cross_val_predict(model, data, solutions, cv=5)
        conf_mat = confusion_matrix(solutions, predictions)
        print(conf_mat)
        print(metrics.classification_report(solutions, predictions))
        print(metrics.precision_recall_fscore_support(solutions, predictions))


def task3(training_df, current_df, modeltype, solutiontype, accuracy):
    """
    Solution for task 3.
    :param training_df: pandas dataframe
    :param current_df: pandas dataframe
    :param modeltype: sklearn class
    :param solutiontype: string (prediction or highValue)
    :param accuracy: accuracy of existing company
    :return: result df and amount of recall deviation
    """
    # split df in data and solutions with the required features only
    data, solutions = split_data(training_df, solutiontype)
    # preproces data with onhotencoding and a standard scaler
    prepared_data, pipleine = preprocess_data(data)
    # compare scores diffrent algorithms
    # choose_algortihm(prepared_data, solutions)
    # get and train a model with the training data
    model = train_data(modeltype, prepared_data, solutions)
    # score the model with a cross_value scorer
    score = cross_val_score(model, prepared_data, solutions, cv=5).mean()
    print(f"The score is {score}")
    # make predictions with the cross value predictor for test purposes
    test_predictions = cross_val_predict(model, data, solutions, cv=5)
    # get the recall and precision of the model
    recall = metrics.recall_score(solutions, test_predictions)
    precision = metrics.precision_score(solutions, test_predictions)
    # print info about the model
    print(model.__class__.__name__)
    print(f"f1-score = {metrics.f1_score(solutions, test_predictions)}")
    print(f"recall = {metrics.recall_score(solutions, test_predictions)}")
    print(f"accuracy = {metrics.accuracy_score(solutions, test_predictions)}")
    print(f"precision = {metrics.precision_score(solutions, test_predictions)}")
    print(metrics.confusion_matrix(solutions, test_predictions))
    print(metrics.classification_report(solutions, test_predictions))

    # prepare current data for prediction
    current_data = split_data(current_df)
    prepared_current_data = pipleine.transform(current_data)
    # make predictions
    predictions = model.predict(prepared_current_data)
    current_df['prediction'] = predictions.tolist()
    # filter only predicted data for company
    predicted_df = current_df[current_df.prediction == True]
    # calculate all the info for the solution and print it
    amount = predicted_df.shape[0]
    right = round(amount * accuracy)
    wrong = amount - right
    total = right * 600 + wrong * 100 - amount * 450
    print(
        f"Total profit of {total} by selling {amount} houses from which: {right} high values and {wrong} low value properties.")

    extra = round(amount * (1 - recall))
    wrong_pred = round(amount * (1 - precision))
    real_amount = amount + extra - wrong_pred
    real_right = round(real_amount * accuracy)
    real_wrong = real_amount - real_right
    real_total = real_right * 600 + real_wrong * 100 - real_amount * 450
    print(f"extra ={extra}", f"   wrong predicted = {wrong_pred}")
    print(
        f"total profit, with recall and precision, of {real_total} by selling {real_amount} houses from which: {real_right} high values and {real_wrong} low value properties.")

    return current_df, extra


def task4(training_df, current_df, modeltype, solutiontype, recall_amount):
    """
    Solution for task 4.
    :param training_df: pandas dataframe
    :param current_df: pandas dataframe
    :param modeltype: sklearn class
    :param solutiontype: string (prediction or highValue)
    :param recall_amount: amount of recall deviation
    :return:
    """
    # filter predictions of other company
    current_df = current_df[current_df.prediction == False]
    current_df = current_df.drop(['prediction'], axis=1).copy()
    # split df in data and solutions with the required features only
    data, solutions = split_data(training_df, solutiontype)
    # preproces data with onhotencoding and a standard scaler
    prepared_data, pipleine = preprocess_data(data)
    # get and train a model with the training data
    model = train_data(modeltype, prepared_data, solutions)
    # score the model with a cross_value scorer
    # score = cross_val_score(model, prepared_data, solutions, cv=5).mean()
    # make predictions with the cross value predictor for test purposes
    test_predictions = cross_val_predict(model, data, solutions, cv=5)
    # get the precision of the model
    precision = metrics.precision_score(solutions, test_predictions)
    # print info about the model
    print(f"f1-score = {metrics.f1_score(solutions, test_predictions)}")
    print(f"recall = {metrics.recall_score(solutions, test_predictions)}")
    print(f"accuracy = {metrics.accuracy_score(solutions, test_predictions)}")
    print(f"precision = {metrics.precision_score(solutions, test_predictions)}")
    print(metrics.confusion_matrix(solutions, test_predictions))
    print(metrics.classification_report(solutions, test_predictions))
    # prepare current data for prediction
    current_data = split_data(current_df)
    prepared_current_data = pipleine.transform(current_data)

    # make predictions
    predictions = model.predict(prepared_current_data)
    current_df['highValue'] = predictions.tolist()
    # filter only predicted high value data
    predicted_df = current_df[current_df.highValue == True]
    # Calculate the solution for best case
    amount = predicted_df.shape[0]
    total = amount * 600 - amount * 450
    print(
        f"total profit of {total} by selling {amount} is bast case")
    # Calculate the solution for worst case precision

    precision_worst = round(amount * (1 - precision))
    precision_worst_right = amount - precision_worst

    worstcase_total = precision_worst_right * 600 + precision_worst * 100 - (amount) * 450
    print(
        f"Total profit precision worst case of {worstcase_total} by selling {amount} houses from which: {precision_worst_right} high values and {precision_worst} low value.")
    # Calculate the solution for worst case recall
    recall_worst_right = amount - recall_amount
    recall_worst_total = recall_worst_right * 600 + recall_amount * 50 - (amount) * 450
    print(
        f"Total profit recall worst case of {recall_worst_total} by selling {amount} houses from which: {recall_worst_right} high values and {recall_amount} high value together with the other company.")
    # Calculate the solution for worst case precision and recall
    worst_right = precision_worst_right - recall_amount
    recall_worst_total = worst_right * 600 + recall_amount * 300 + precision_worst * 100 - (amount) * 450
    print(
        f"Total profit recall and precision worst case of {recall_worst_total} by selling {amount} houses from which: {worst_right} high values and {recall_amount} high value together with the other company and {precision_worst} low value.")
    # write results
    predicted_df[['identifier']].to_csv("selection.csv", header=False, index=False)


def task1(df):
    """
    Solution for task 1.
    :param df: pandas dataframe
    :return: the accuracy of the existing company
    """
    predicted = df[df.prediction == True]
    correct = predicted[predicted.highValue == True]
    wrong = predicted[predicted.highValue == False]
    total_amount = predicted.shape[0]
    correct_amount = correct.shape[0]
    wrong_amount = wrong.shape[0]
    income = correct_amount * 600 + wrong_amount * 100
    expenses = total_amount * 450
    profit = income - expenses

    print(
        f"The company bought {total_amount} houses, from which {correct_amount} high value and {wrong_amount} low value.")

    print(f"The company had an income of {income} euro and {expenses} euro expenses.")
    print(f"This totals to a profit of {profit}")
    accuracy = correct_amount / total_amount
    print(f"This means that the company had an accuracy of {accuracy}.")
    return accuracy


def task2(df):
    """
    Solution for task 2.
    :param df: pandas dataframe
    :return:
    """
    highvalue = df[df.highValue == True]
    predicted = highvalue[highvalue.prediction == True]
    not_predicted = highvalue[highvalue.prediction == False]
    highvalue_amount = highvalue.shape[0]
    predicted_amount = predicted.shape[0]
    not_predicted_amount = not_predicted.shape[0]
    percentage = 100 * not_predicted_amount / highvalue_amount
    print(
        f"Of the {highvalue_amount} high  value houses, the company bought {predicted_amount} and didn't by {not_predicted_amount}.")
    print(f"This means that the company bought {percentage}% of the high value houses in the market.")
    print(df.corr())


def readDB(filename):
    """
    read the database
    :param filename: string
    :return: pandas framework
    """
    dataset = pd.read_csv(filename)
    return dataset


if __name__ == '__main__':
    # Load the dataset as a pandas DataFrame
    filename = "historical.csv"
    current_filename = "current.csv"
    df = readDB(filename)
    current_df = readDB(current_filename)

    # change printoptions
    dataset = np.genfromtxt(filename, delimiter=',', skip_header=1)
    desired_width = 320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', desired_width)
    np.set_printoptions(linewidth=desired_width)
    # switch feature names year and type
    df.rename({"type": "temp"}, axis=1, inplace=True)
    df.rename({"year": "type"}, axis=1, inplace=True)
    df.rename({"temp": "year"}, axis=1, inplace=True)

    current_df.rename({"type": "temp"}, axis=1, inplace=True)
    current_df.rename({"year": "type"}, axis=1, inplace=True)
    current_df.rename({"temp": "year"}, axis=1, inplace=True)

    # switch feature names bathrooms and floor
    df.rename({"bathrooms": "temp"}, axis=1, inplace=True)
    df.rename({"floor": "bathrooms"}, axis=1, inplace=True)
    df.rename({"temp": "floor"}, axis=1, inplace=True)

    current_df.rename({"bathrooms": "temp"}, axis=1, inplace=True)
    current_df.rename({"floor": "bathrooms"}, axis=1, inplace=True)
    current_df.rename({"temp": "floor"}, axis=1, inplace=True)

    # task 1
    print("##### Task 1 #####")
    accuracy = task1(df)

    # task 2
    print("\n\n##### Task 2 #####")
    task2(df)
    # task3
    print("\n\n##### Task 3 #####")
    current_df, recall_amount = task3(df, current_df, DecisionTreeClassifier, "prediction", accuracy)

    # task4
    print("\n\n##### Task 4 #####")
    task4(df, current_df, RidgeClassifier, "highValue", recall_amount)
