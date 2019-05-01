"""A very simple 30-second Keras example.

Author: Christian A. Duncan (based off of the example given on Keras site)
Course: CSC350: Intelligent Systems
Term: Spring 2017

This one trains on the function "1-bit add with carry".
Examples are 
   ((0, 0),(0, 0),
    (0, 1),(0, 1),
    (1, 0),(0, 1),
    (1, 1),(1, 0))
   This requires two inputs and two outputs with 2 nodes in the hidden layer.
"""
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import sys
import re
import json
from sklearn import metrics as jeff
import math

class DigitClassifier:
	def __init__(self, data, model):
		self.data = data
		self.model = model

	
		model = Sequential()
		model.add(Dense(units=1024, input_dim=1024)) # First (hidden) layer
		model.add(Activation('sigmoid'))
		model.add(Dense(units=50)) # First (hidden) layer
		model.add(Activation('sigmoid'))
		model.add(Dense(units=1))   # Second, final (output) layer
		model.add(Activation('sigmoid'))
		
		model.compile(loss='mean_squared_error',
		              optimizer='sgd',
		              metrics=['accuracy'])
		path_beg = "C:/DigitProject/DigitDetector/digit_data/"
		x_train = self.data[:len(self.data)-1]
		x_train_data = [[self.read_data(path_beg + i) for i in j] for j in self.data[:len(self.data)-1]]
		y_train_arr = [[re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_train[j]] for j in range(len(x_train))]
		y_train_data = [[int(i.group(1)) for i in y_train_arr[j] if i] for j in range(len(y_train_arr))]

		x_train_data = np.array(x_train_data).reshape((-1,1024))
		y_train_data = np.array(y_train_data).reshape((-1))

		#print(y_train_data)
		#print("Length={}".format(len(y_train_data)))
		#print("Length[0]={}".format(len(y_train_data[0])))
		#print("Length[0][0]={}".format(len(x_train_data[0][0])))
		#sys.exit(1)
		model.fit(x_train_data, y_train_data, epochs=10, batch_size=8)
		

		

	
		# Train the model, iterating on the data in batches of 32 samples (try batch_size=1)
		
		x_test = self.data[-1:]
		x_test_data = [[self.read_data(path_beg + i) for i in j] for j in self.data[-1:]]   # Random input data
		y_test_arr = [[re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_test[j]] for j in range(len(x_test))]
		y_test_data = [[int(i.group(1)) for i in y_test_arr[j] if i] for j in range(len(y_test_arr))]



		x_test_data = np.array(x_test_data).reshape((-1,1024))
		y_test_data = np.array(y_test_data).reshape((-1))

		#print(x_test_data)
		#print("Length={}".format(len(x_test_data)))
		
		# Evaluate the model from a sample test data set
		y_predict = model.predict(x_test_data)
		y_predict = np.array([ round(p[0]) for p in y_predict])

		xd = jeff.confusion_matrix(y_test_data, y_predict)
		tp = xd[0][0]
		fp = xd[0][1]
		fn = xd[1][0]
		tn = xd[1][1]

		print(xd)
		mcc = ((tp*tn)-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
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

	def create_model(self, in_dim, units, activation):
		model = Sequential()
		model.add(Dense(units=units[0], input_dim=in_dim)) # First (hidden) layer
		model.add(Activation(activation))
		for i in units[1:]:
			model.add(Dense(units=i)) 
			model.add(Activation(activation))
		model.compile(loss='mean_squared_error',
		              optimizer='sgd',
		              metrics=['accuracy'])
		return model