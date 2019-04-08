from keras.models import Sequential
from keras.layers import Dense
import sample_read.py
import numpy

#fix random seed for reproducibility 
numpy.random.seed(7)
#load pima indians dataset
datset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter-",")
X = dataset[:,0:8]
Y = dataset[:,8]

#create model
model= Sequential()
model.add(Dense(12, input_dim=8; activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit or train the model
model.fit(X,Y, epochs=150, batch_size=10)

#evaluate or test the model
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)