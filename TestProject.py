
""""
Author: Christian Duncan
Date: Spring 2019
Course: CSC350 (Intelligent Systems)

This sample code shows how one can read in our JSON image files in Python3.  It reads in the file and then outputs the two-dimensional array.
It applies a simple threshold test - if value is > threshold then it outputs a '.' otherwise it outputs an 'X'.  More refinement is probably better.
But this is for displaying images not processing or recognizing images."""

import json
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import sys
""" 
Reads in the given JSON file as outlined in the README.txt file.
"""
def read_data(file):
    try:
        with open(file, 'r') as inf:
            bitmap = json.load(inf)

        return bitmap
    except FileNotFoundError as err:
        print("File Not Found: {0}.".format(err))

""" 
Prints out the given "image" (a 2-dimensional array).
This just replaces any values greater than a threshold with . and otherwise with an X.
"""
def print_image(img, threshold):
    for row in img:
        for pixel in row:
            print('.' if pixel > threshold else 'X', end='')
        print()  # Newline at end of the row

""" 
Main entry point.  Assumes all the arguments passed to it are file names.
For each argument, reads in the file and the prints it out.
"""
def main():
    x_data = []
    y_data = []

    # for file_name in sys.argv[1:]:
    # Input file structure: input_User_Digit_Sample.json
    for user in range(1,24):
        for digit in range(0, 2):
            for sample in range(0, 2):
                file_name = "input_" + str(user) + "_" + str(digit) + "_" + str(sample) + ".json"
                img = read_data(file_name)
                data = np.array(img).flatten()
                data = [ p / 255 for p in data ]
                print("Data = {}".format(data))
                x_data.append(data)
                y_data.append([1, 0] if digit == 0 else [0, 1])
                # print("Data: {}".format(data))
                # print("Displaying image: {0}.".format(file_name))
                # print_image(img, 200)   # Different thresholds will change what shows up as X and what as a .


    x_data = np.array(x_data)
    y_data = np.array(y_data)
    print("X Train = {}".format(x_data))
    print("Y Train = {}".format(y_data))
    print("X Train length = {}".format(len(x_data)))

    model= Sequential()
    model.add(Dense(units=1000, input_dim=1024)) # First (hidden) layer
    model.add(Activation('sigmoid'))
    model.add(Dense(units=500)) # First (hidden) layer
    model.add(Activation('sigmoid'))
    model.add(Dense(units=2))   # Second, final (output) layer
    model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples (try batch_size=1)
    model.fit(x_data, y_data, epochs=10, batch_size=8)



# Evaluate the model from a sample test data set
    score = model.evaluate(x_data, y_data)
    print()
    print("Score was {}.".format(score))
    print("Labels were {}.".format(model.metrics_names))

# Make a few predictions
    x_input = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_output = model.predict(x_input)
    print("Result of {} is {}.".format(x_input, y_output))

# This is just used to trigger the Python to run the main() method.  The if statement is used so that if this code
# were imported as a module then everything would load but main() would not be executed.
if __name__ == "__main__":
    main()

''' 

""""
Author: Christian Duncan
Date: Spring 2019
Course: CSC350 (Intelligent Systems)

This sample code shows how one can read in our JSON image files in Python3.  It reads in the file and then outputs the two-dimensional array.
It applies a simple threshold test - if value is > threshold then it outputs a '.' otherwise it outputs an 'X'.  More refinement is probably better.
But this is for displaying images not processing or recognizing images."""

from os import listdir
from os.path import isfile, join
import re
import json
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import sys
""" 
Reads in the given JSON file as outlined in the README.txt file.
"""
def read_data(file):
    try:
        with open(file, 'r') as inf:
            bitmap = json.load(inf)

        return bitmap
    except FileNotFoundError as err:
        print("File Not Found: {0}.".format(err))

""" 
Prints out the given "image" (a 2-dimensional array).
This just replaces any values greater than a threshold with . and otherwise with an X.
"""
def print_image(img, threshold):
    for row in img:
        for pixel in row:
            print('.' if pixel > threshold else 'X', end='')
        print()  # Newline at end of the row

""" 
Main entry point.  Assumes all the arguments passed to it are file names.
For each argument, reads in the file and the prints it out.
"""
def main():


    mypath = "C:/MiniDataSet"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    x = [re.search("^input_{0-9}+_{0-9}+_{0-9}+\.json$", i) for i in files]
    print(x)


  

# This is just used to trigger the Python to run the main() method.  The if statement is used so that if this code
# were imported as a module then everything would load but main() would not be executed.
if __name__ == "__main__":
    main()

'''