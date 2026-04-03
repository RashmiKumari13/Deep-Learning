
!pip install keras
!pip install TensorFlow
!pip install theano

import keras as k
from keras.models import Sequential 
from keras.layers import Dense, Activation 

model = Sequential()  
model.add(Dense(512, activation = 'relu', input_shape = (784,)))

from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout model = Sequential() 

model.add(Dense(512, activation = 'relu', input_shape = (784,))) 
model.add(Dropout(0.2)) 
model.add(Dense(512, activation = 'relu')) model.add(Dropout(0.2)) 
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(
   loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy']
)
model.summary()
