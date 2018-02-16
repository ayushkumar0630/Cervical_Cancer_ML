"""
Use this file to train the ML algorithm
"""


import numpy as np
import sklearn as sk
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import os

# use this function to open up the excel document
def open_data(PATH): 
	csv = pd.read_csv("/Vitrix_Health/Cervical_Cancer_ML/Data/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv")
	return csv

# use this function to get rid of missing data and
# perform more data preprocessing 
def data_preprocessing(csv):
	#convert to numeric type
	csv = csv.convert_objects(convert_numeric=True)

	# for continuous data
	csv['Number of sexual partners'] = csv['Number of sexual partners'].fillna(csv['Number of sexual partners'].median())
	csv['First sexual intercourse'] = csv['First sexual intercourse'].fillna(csv['First sexual intercourse'].median())
	csv['Num of pregnancies'] = csv['Num of pregnancies'].fillna(csv['Num of pregnancies'].median())
	csv['Smokes'] = csv['Smokes'].fillna(1)
	csv['Smokes (years)'] = csv['Smokes (years)'].fillna(csv['Smokes (years)'].median())
	csv['Smokes (packs/year)'] = csv['Smokes (packs/year)'].fillna(csv['Smokes (packs/year)'].median())
	csv['Hormonal Contraceptives'] = csv['Hormonal Contraceptives'].fillna(1)
	csv['Hormonal Contraceptives (years)'] = csv['Hormonal Contraceptives (years)'].fillna(csv['Hormonal Contraceptives (years)'].median())
	csv['IUD'] = csv['IUD'].fillna(0) # Under suggestion
	csv['IUD (years)'] = csv['IUD (years)'].fillna(0) #Under suggestion
	csv['STDs'] = csv['STDs'].fillna(1)
	csv['STDs (number)'] = csv['STDs (number)'].fillna(csv['STDs (number)'].median())
	csv['STDs:condylomatosis'] = csv['STDs:condylomatosis'].fillna(csv['STDs:condylomatosis'].median())
	csv['STDs:cervical condylomatosis'] = csv['STDs:cervical condylomatosis'].fillna(csv['STDs:cervical condylomatosis'].median())
	csv['STDs:vaginal condylomatosis'] = csv['STDs:vaginal condylomatosis'].fillna(csv['STDs:vaginal condylomatosis'].median())
	csv['STDs:vulvo-perineal condylomatosis'] = csv['STDs:vulvo-perineal condylomatosis'].fillna(csv['STDs:vulvo-perineal condylomatosis'].median())
	csv['STDs:syphilis'] = csv['STDs:syphilis'].fillna(csv['STDs:syphilis'].median())
	csv['STDs:pelvic inflammatory disease'] = csv['STDs:pelvic inflammatory disease'].fillna(csv['STDs:pelvic inflammatory disease'].median())
	csv['STDs:genital herpes'] = csv['STDs:genital herpes'].fillna(csv['STDs:genital herpes'].median())
	csv['STDs:molluscum contagiosum'] = csv['STDs:molluscum contagiosum'].fillna(csv['STDs:molluscum contagiosum'].median())
	csv['STDs:AIDS'] = csv['STDs:AIDS'].fillna(csv['STDs:AIDS'].median())
	csv['STDs:HIV'] = csv['STDs:HIV'].fillna(csv['STDs:HIV'].median())
	csv['STDs:Hepatitis B'] = csv['STDs:Hepatitis B'].fillna(csv['STDs:Hepatitis B'].median())
	csv['STDs:HPV'] = csv['STDs:HPV'].fillna(csv['STDs:HPV'].median())
	csv['STDs: Time since first diagnosis'] = csv['STDs: Time since first diagnosis'].fillna(csv['STDs: Time since first diagnosis'].median())
	csv['STDs: Time since last diagnosis'] = csv['STDs: Time since last diagnosis'].fillna(csv['STDs: Time since last diagnosis'].median())

	#for catagorical data
	csv = pd.get_dummies(data=csv, columns=['Smokes','Hormonal Contraceptives','IUD','STDs', 'Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology','Schiller'])
	return csv

def define_x_variables(csv):
	x_var = csv.iloc[:,0:27].values
	return x_var

def define_y_variables(csv):
	y_var = csv.iloc[:,28:].values
	return y_var

def replace_questions_marks(csv):
	return csv.replace('?', np.NaN)

def split_dataset(X, Y, test_size):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)
	return X_train, X_test, Y_train, Y_test

def test_accuracy(Y_predict, Y_actual):
	num_matches = 0

	for x in range(len(Y_predict)):
		if Y_predict[x] == Y_actual[x]: 
			num_matches = num_matches + 1

	return float(num_matches)/float(len(Y_predict)+1)

def random_forest_classification(X_train, Y_train, X_test):
	forest_classifier = RandomForestClassifier(n_estimators = 1000)
	forest_classifier.fit(X_train, Y_train)
	Y_predict = forest_classifier.predict(X_test)
	return forest_classifier, Y_predict

def plot_feature_importance(rfc_model, features):
	importances = rfc_model.feature_importances_
	indices = np.argsort(importances)
	
	plt.figure(1)
	plt.title('Feature Importances')
	plt.barh(range(len(indices)), importances[indices], color='b', align='center')
	plt.yticks(range(len(indices)), features[indices])	
	plt.xlabel('Relative Importance')
	plt.show()

def train_random_forest_classifier():
	# open dataset
	cervical_url = "cervical-cancer-risk-classification"
	cervical_csv = open_data(cervical_url)

	# alot of '?' values so we need to replace
	cervical_csv = replace_questions_marks(cervical_csv)
	preprocessed_csv = data_preprocessing(cervical_csv)

	# define X and Y variables	
	X = define_x_variables(preprocessed_csv)
	Y = define_y_variables(preprocessed_csv)

	X_train, X_test, Y_train, Y_test = split_dataset(X, Y, test_size = .25)

	# run classifier
	rfc_model, Y_predict_forest = random_forest_classification(X_train, Y_train, X_test)

	"""	
	print "Prediction (RFC): "
	print (Y_predict_forest)
	print "Actual: " 
	print (Y_test)

	print "Accuracy(RFC): "
	print (accuracy_score(Y_test, Y_predict_forest) * 100.)
	"""
	return accuracy_score(Y_test, Y_predict_forest) * 100.
	#plot_feature_importance(rfc_model, X)
	#export_graphviz(rfc_model.fit(X_train, Y_train), filled=True, rounded=True)
	#os.system('dot -Tpng tree.dot -o tree.png')

if __name__ == "__main__":
	#running the multiple linear regression model
	train_random_forest_classifier()
