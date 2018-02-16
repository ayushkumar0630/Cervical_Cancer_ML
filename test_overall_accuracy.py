# Using this file to test the total mean with standard deviation

import numpy as np
import training
import sys


def test_random_forest_classifier_accuracy(number_of_times):
	list_of_accuracies = []	
	return_time = number_of_times
	print (number_of_times)	
	while (number_of_times > 0):
		list_of_accuracies.append(training.train_random_forest_classifier())
		number_of_times = number_of_times - 1
		#print (number_of_times)
	
	#print (list_of_accuracies)
	np_accuracies = np.array(list_of_accuracies)
	
	print ("Mean of " + str(return_time) + " times:")
	print (np.mean(np_accuracies))
	print ("Standard Deviation of " + str(return_time) + " times:")	
	print (np.std(np_accuracies))	

	return np.std(np_accuracies), np.mean(np_accuracies)

if __name__ == "__main__":
	#running the multiple linear regression model
	#if (len(sys.argv) <= 1):
	#	print ("USAGE: script + int(amount of times you want to run)")
	#	return
	#else:		
	number_of_times = sys.argv[1]
	int_number_of_times = int(number_of_times)
	test_random_forest_classifier_accuracy(int_number_of_times)
