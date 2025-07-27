import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt

def load_data(path):
    data = pd.read_csv(path)
    return data
def locate_null(data):
    null_columns = []
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            null_columns.append(col)
        #print(f"{col} has {data[col].isnull().sum()} missing values") if data[col].isnull().sum() > 0 else print(f"{col} has no missing values")
    return null_columns

def one_hot_code_dataset(data, columns):
    for col in columns:
        one_hot = pd.get_dummies(data[col], prefix=col)
        data = pd.concat([data, one_hot], axis=1)
        data.drop(col, axis=1, inplace=True)

    bool_columns = data.select_dtypes(include=['bool']).columns
    data[bool_columns] = data[bool_columns].astype(int)

    return data

def plot_histogram(data):
    for col in data.columns:
        data.loc[:, col].hist()
        plt.title(col)
        plt.show()

def standardize_dataset(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data

def impute_missing_values(raw_data):
    reg_imputer = IterativeImputer()
    data = reg_imputer.fit_transform(raw_data)
    data = pd.DataFrame(data, columns=data.columns)
    return data

