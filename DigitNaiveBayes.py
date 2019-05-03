import sys
import re
import json
from sklearn import metrics as jeff
import math
import numpy as np
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.neighbors

""" Convert a digit to an array of 0's and a 1.  E.g. 3 converts to 0, 0, 0, 1, 0, 0, 0, ...."""
def convert(digit, num_digits):
	d = [0]*num_digits
	d[digit] = 1
	return np.array(d)

class DigitNaiveBayes:
	def __init__(self, data, base_path, model_type):
		self.data = data

		print("DEBUG: DNB.  Begin")
		path_beg = base_path

		sum_matrices = None
		for i in range(len(self.data)):
			x_test = self.data.pop(i)
			x_train = self.data

			x_train_data = [[self.read_data(path_beg + i) for i in j] for j in x_train]
			y_train_arr = [[re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_train[j]] for j in range(len(x_train))]
			y_train_data = [[int(i.group(1)) for i in y_train_arr[j] if i] for j in range(len(y_train_arr))]
	
			x_train_data = np.array(x_train_data).reshape((-1,1024))
			y_train_data = np.array(y_train_data).reshape((-1))
			#print(y_train_data)
			#print("Length={}".format(len(y_train_data)))
			#print("Length[0]={}".format(len(y_train_data[0])))
			#print("Length[0][0]={}".format(len(x_train_data[0][0])))
			#sys.exit(1)
			num_digits = 10
			print("DEBUG: DNB Creating GaussianNB")
			model = self.create_model(1024,[512,512,10], 'sigmoid', model_type)
			print("DEBUG: DNB Fitting GaussianNB")
			if model_type == 'neural':
				print("Before: {}".format(y_train_data))
				y_train_data_adj = np.array([convert(digit, num_digits) for digit in y_train_data])
				print("After: {}".format(y_train_data_adj))
				model.fit(x_train_data, y_train_data_adj, epochs=10, batch_size=8)
			else:
				model.fit(x_train_data, y_train_data)
			print("DEBUG: DNB Fitted GaussianNB")
	
			x_test_data = [self.read_data(path_beg + i) for i in x_test]   # Random input data
			y_test_arr = [re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_test]
			y_test_data = [int(i.group(1)) for i in y_test_arr if i]



			x_test_data = np.array(x_test_data).reshape((-1,1024))
			y_test_data = np.array(y_test_data).reshape((-1))
			#print(x_test_data)
			#print("Length={}".format(len(x_test_data)))
	
			print("DEDNG Calling predict")
			# Evaluate the model from a sample test data set
			y_predict = model.predict(x_test_data)
			print(y_predict)
			if model_type == 'neural':
				y_predict = np.array([np.argmax(p) for p in y_predict])
				print(y_predict)

	
			xd = jeff.confusion_matrix(y_test_data, y_predict)
			if i == 0:
				sum_matrices = np.zeros(xd.shape)
			else:
				sum_matrices = np.add(sum_matrices, xd)
			
			self.data.insert(len(self.data),x_test)

		tp = [sum_matrices[i][i] for i in range(0, len(sum_matrices))]
		fp = [0]*len(sum_matrices)
		fn = [0]*len(sum_matrices)
		for d in range(0, len(sum_matrices)):
			for d2 in range(0, len(sum_matrices)):
				if d != d2:
					fp[d] += sum_matrices[d2][d]
					fn[d] += sum_matrices[d][d2]
		sum = np.sum(sum_matrices)
		tn = [sum - tp[i] - fp[i] - fn[i] for i in range(0, len(sum_matrices))]
		mcc = [((tp[i]*tn[i])-fp[i]*fn[i])/math.sqrt((tp[i]+fp[i])*(tp[i]+fn[i])*(tn[i]+fp[i])*(tn[i]+fn[i])) for i in range(0, len(sum_matrices))]
		print(sum_matrices)
		print(mcc)
		#print("Score was {}.".format(score))
		#print("Labels were {}.".format(model.metrics_names))

		# Make a few predictions
		#x_input = np.array([[0,0], [0,1], [1,0], [1,1]])
		#y_output = model.predict(x_input)
		#print("Result of {} is {}.".format(x_input, y_output))
	""" 
Reads in the given JSON file as outlined in the README.txt file.
"""
	def read_data(self, file):
		try:
			with open(file, 'r') as inf:
				bitmap = json.load(inf)

			return bitmap
		except FileNotFoundError as err:
			print("File Not Found: {0}.".format(err))

	def create_model(self, in_dim, units, activation, model_type):
		assert model_type in ['neural','svm','bayes', 'kNN','LDA']
		
		if model_type == 'neural':
			model = Sequential()
			#model = self.create_model(1024,[1024,50,10], 'sigmoid', model_type)
			model.add(Dense(units=units[0], input_dim=in_dim)) # First (hidden) layer
			model.add(Activation(activation))
			for i in units[1:]:
				model.add(Dense(units=i)) 
				model.add(Activation(activation))
			model.compile(loss='mean_squared_error',
		    	          optimizer='sgd',
		    	          metrics=['accuracy'])
		elif model_type == 'svm':
			model = svm.LinearSVC()
		elif model_type == 'bayes':
			model = GaussianNB()
		elif model_type == 'kNN':
			model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
		elif model_type == 'LDA':
			model = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
		
		return model