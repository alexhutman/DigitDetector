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

class DigitClassifier:
	def __init__(self, data, model):
		self.data = data
		self.model = model

	
		model = Sequential()
		model.add(Dense(units=100, input_dim=1024)) # First (hidden) layer
		model.add(Activation('sigmoid'))
		model.add(Dense(units=50)) # First (hidden) layer
		model.add(Activation('sigmoid'))
		model.add(Dense(units=1))   # Second, final (output) layer
		model.add(Activation('sigmoid'))
		
		model.compile(loss='mean_squared_error',
		              optimizer='sgd',
		              metrics=['accuracy'])
		path_beg = "C:/Users/Alex/Dropbox/School Snoot/Spring 2019/AI/DigitDetector/digit_data/"
		x_train = self.data[:len(self.data)-1]
		x_train_data = [[self.read_data(path_beg + i) for i in j] for j in self.data[:len(self.data)-1]]
		y_train_arr = [[re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_train[j]] for j in range(len(x_train))]
		y_train = [[i.group(1) for i in y_train_arr[j] if i] for j in range(len(y_train_arr))]

		for i in x_train_data:
			for j in i:
				model.fit(np.array(j), np.array(y_train), epochs=10, batch_size=8)
		

		

	
		# Train the model, iterating on the data in batches of 32 samples (try batch_size=1)
		### model.fit(np.array(x_train_data), np.array(y_train), epochs=10, batch_size=8)
		
		### x_test = self.data[len(self.data)-1:][0]   # Random input data
		### y_test = np.array([[x and y, x^y] for (x,y) in x_test]) # The results (Carry,Add)
		### 
		### # Evaluate the model from a sample test data set
		### score = model.evaluate(x_test, y_test)
		### print()
		### print("Score was {}.".format(score))
		### print("Labels were {}.".format(model.metrics_names))
		### 
		### # Make a few predictions
		### x_input = np.array([[0,0], [0,1], [1,0], [1,1]])
		### y_output = model.predict(x_input)
		### print("Result of {} is {}.".format(x_input, y_output))

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
