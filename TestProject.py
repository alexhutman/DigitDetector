
'''
Author: Christian Duncan
Date: Spring 2019
Course: CSC350 (Intelligent Systems)

This sample code shows how one can read in our JSON image files in Python3.  It reads in the file and then outputs the two-dimensional array.
It applies a simple threshold test - if value is > threshold then it outputs a '.' otherwise it outputs an 'X'.  More refinement is probably better.
But this is for displaying images not processing or recognizing images.
'''

from os import listdir
from os.path import isfile, join
import re
import json
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import sys
import random
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

def kCross(data, percentage):
    assert percentage >= 0 and percentage <= 1

    train = random.sample(data,round(percentage*len(data)))
    test = list(set(data)-set(train))
    return [train, test]

""" 
Main entry point.  Assumes all the arguments passed to it are file names.
For each argument, reads in the file and the prints it out.
"""
def main():
    mypath = "C:/DigitProject/DigitDetector/digit_data"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    x = [re.search("^input_[0-9]+_[0-9]+_[0-9]+\.json$", i) for i in files]
    y = [i.group(0) for i in x if i]

    partitions = kCross(y, 0.8)
    train = partitions[0]
    test = partitions[1]
    print(len(train), len(test))


  

# This is just used to trigger the Python to run the main() method.  The if statement is used so that if this code
# were imported as a module then everything would load but main() would not be executed.
if __name__ == "__main__":
    main()


