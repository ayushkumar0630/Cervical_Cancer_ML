"""
Use this file to train the ML algorithm
"""


import numpy as np
import sklearn as sk
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# use this function to open up the excel document
def open_data(PATH): 
	csv = pd.read_csv("Data/kag_risk_factors_cervical_cancer.csv")
	return csv

# use this function to get rid of missing data and
# perform more data preprocessing 
def data_preprocessing(csv):
	return csv

def define_x_variables(csv):
	x_var = csv.iloc[:,0:27].values
	return x_var

def define_y_variables(csv):
	y_var = csv.iloc[:,28:31].values
	return y_var

def missing_values(): 
	imputer = sk.imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

def replace_questions_marks(csv):
	return csv.replace('?', np.NaN)

def split_dataset(X, Y, test_size):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)
	return X_train, X_test, Y_train, Y_test

def train_multiple_linear_regression():
	#import linear regression model
	from sklearn.linear_model import LinearRegression

	# open dataset
	cervical_url = "cervical-cancer-risk-classification"
	cervical_csv = open_data(cervical_url)

	# alot of '?' values so we need to replace
	cervical_csv = replace_questions_marks(cervical_csv)

	preprocessed_csv = data_preprocessing(cervical_csv)

	# define X and Y variables	
	X = define_x_variables(cervical_csv)
	Y = define_y_variables(cervical_csv)

	X_train, X_test, Y_train, Y_test = split_dataset(X, Y, test_size = .20)

	# run regression 
	multi_regression = LinearRegression()
	multi_model = multi_regression.fit(X_train, Y_train)

	Y_predict = multi_regression.predict(X_test)

	print "Prediction: " + Y_predict
	print "Actual: " + Y_test

	# debug code
	#print "X : ", X
	#for x in range(len(X)):
	#	print X[x]
	#print "Y : ", Y
	#for y in range(len(Y)):
	#	print Y[y]
	#print "preprocessed_data: ", preprocessed_data


if __name__ == "__main__":
	#running the multiple linear regression model
	train_multiple_linear_regression()