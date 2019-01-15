# Working with IMDB data to classify reviews as positive or negative
# From 'Deep Learning with Python' by Francois Chollet, Manning Press


#%%
# Downloading/loading the built-in imdb data
from keras.datasets import imdb

#Setting up train and test data
(trainData, trainLabels), (testData, testLabels) = imdb.load_data(
    num_words = 10000 #Only keep top 10K words
)

#%%
#Decoding one of the reviews back to English, just to see how it's done
wordIndex = imdb.get_word_index()
reverseWordIndex = dict(
    [(value, key) for (key, value) in wordIndex.items()])
decodedReview = ' '.join(
    [reverseWordIndex.get(i - 3, '?') for i in trainData[0]])
print(decodedReview)

#%% [markdown]
## Preparing the data
#"You can't feed a list of integers into a neural network. You have to turn them
# into into tensors. There are two ways to do that:<br/><br/>
#       1. Pad your lists so they all have the same length, turn them into n integer
#          tensor of shape (samples, word_indices), and then use as the first layer in
#          your network-- a layer capable of handling such integer tensors (the 'embedding'
#          layer, covered later in the book).<br/><br/>
#       2. One-hot encode your lists to turn them into vectors of 0s and 1s. This would mean,
#          for instance, turning the sequence [3, 5] into a 10K-dimensional vector
#          that would all be zeroes except for indices 3 and 5, which would be ones. Then you
#          could use as the first layer in your network a 'Dense' layer, capable of handling
#          floating-point vector data
# 
# Let's go with the latter solution to vectorize the data, which you;ll do manually for
# maximum clarity"


#%%
import numpy as np 

# Create an all-zero matrix of shape(len(sequences), dimension)
def vectorizeSequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        #Set specific indices of results[i] to 1s
        results[i, sequence] = 1.
    return results

xTrain = vectorizeSequences(trainData) #Vectorized training data
xTest = vectorizeSequences(testData) #Vectorized test data

#Vectorize the labels, as well
yTrain = np.asarray(trainLabels).astype('float32')
yTest = np.asarray(testLabels).astype('float32')


    
#%% [markdown]
## Building Your Network
# "The input data are vectors, and the labels are scalars."
# This type of data works well with a simple stack of fully connected ('Dense')
# layers with 'relu' activations : Dense(16, activation = 'relu')<br/>

# The above line passes 16 to the Dense layer because that's the
# number of 'hidden units' in the layer. <br/><br/>

#      Hidden Units = dimension in the representation space
#      of the layer. A way of thinking about hidden units is that they
#      represent "how much freedom you're allowing the representation to
#      have when learning internal representations." As hidden units (dimensional
#      representation space) increases, your model can handle higher-complexity
#      problems, but computational complexity goes up, as does the potential
#      for overfitting.

#### Two Key Architecture Decisions - How many layers to use and
#### how many hidden units per layer
# (We'll cover how to do this in the next chapter. For now, he chooses for us)<br/><br/>
# For this chapter, we'll use this architecture: 
# * Two intermediate layers with 16 hidden units each
# * A third, output layer that will return the scalar prediction
# Relu activation will be used on the intermediate layers, and we'll use signmoid 
# activation on the output layer so that we get probability scores<br/><br/>

# *Note: Relu (rectified linear unit) zeroes out negative numbers*

## The Model Definition in Keras

#%%
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))



#%% [markdown]
#### What are activation functions and why are they necessary?

# Activation functions like 'relu' provide the ability to deal with non-linearity.
# Without them, the 'Dense' layer would only consist of linear operations-- dot product 
# and addition (output = dot(W, input) + b)<br/><br/>

# If this were the case, we could only handle linear transformations: "The hypthesis 
# space of the layer would be the set of all possible linear transformations of the 
# input data into a 16-dimensional space." Therefore, adding extra layers would not add
# any extra benefit, as each successive stack would still just be implementing linear 
# operations.<br/><br/>

# relu is the most common activation function, but there are many others.



#%% [markdown]
#### Choosing a loss function and an optimizer

# In this problem, we are performing a binary classification with probability 
# as the output, so we'l be using *binary_crossentropy* as our loss function.
# (We could also use something like *mean_squared_error*, but binary_crossentropy
# is a better choice when we're dealing with output probabilities.)<br/><br/>

##### *Crossentropy* measures the distance between probability distributions. In this example, it measures the distance between the actual and predicted values.


#%%
# We configure the model in this step
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# This is not the only option. We're passing the optimizer, loss, and metrics as strings
# because they are packaged in keras. If we wanted to either configure the parameters of 
# the optimizer, we could pass it as a class instance, seen here:

# from keras import optimizers
# model.compile(optimizer = optimizers.RMSprop(lr = 0.001),)
# <br/>

# If we wanted to pass custom loss functions or metrics, we could create them as a 
# function, then pass them as the loss or metric arguments:<br/>

# loss = losses.binary_crossentropy,
# metrics = [metrics.binary_accuracy])



#%% [markdown]
#### 3.4.4 Validation your approach


#%%
# First, we'll split off a validation set from the training data
xVal = xTrain[:10000]
partialXtrain = xTrain[10000:]
yVal = yTrain[:10000]
partialYtrain = yTrain[10000:]

#%%
# Next, we'll train the model for 20 *epochs* (which just means we'll iterate
# over the xTrain and yTrain tensors 20 times). We'll use *mini-batches* of 512
# samples. Loss and accuracy will be monitored on our validation set.
# This is achieved by passing xVal and yVal to the 'validation_data' argument

history = model.fit(partialXtrain,
                    partialYtrain,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (xVal, yVal))

#%%
# The *model.fit* call above returns a history object. This contains a *member history*
# which is a dict containing data about every event in the training of the model.<br/>

# Examine the history:
historyDict = history.history
historyDict.keys()

#%% [markdown]
#### Examining the learning history
# There are four entries in our example-- one per metric in the training and validation.
# We can use Matplotlib to plot the loss and accuracy


#%%
# Plotting the training and validation loss
import matplotlib.pyplot as plt 

historyDict = history.history
lossValues = historyDict['loss']
valLossValues = historyDict['val_loss']

epochs = range(1, len(lossValues) + 1)

plt.plot(epochs, lossValues, 'bo', label = 'Training Loss')
plt.plot(epochs, valLossValues, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#%%
# Plotting the training and validation accuracy
plt.clf()
accValues = historyDict['acc']
valAccValues = historyDict['val_acc']

plt.plot(epochs, accValues, 'bo', label = 'Training Accuracy')
plt.plot(epochs, valAccValues, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#%% [markdown]
# In the above charts, we see the telltale signs of *overfitting*. While accuracy increased
# with each successive epoch on the training set, the validation accuracy peaked at about 4 
# or 5 epochs into the training. After just the second epoch, we were learning
# representations that only really apply to the training data.<br/>

#Next, we'll retain the model from scratch, but use only four epochs


#%%
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

model.fit(xTrain, yTrain, epochs = 4, batch_size = 512)
results = model.evaluate(xTest, yTest)

#%% [markdown]
# In the above model, we achieved slightly higher accuracy than our first model (88& vs 85%)
# using a simpler, naive method that was far less computationally expensive.


#%% [markdown]
#### 3.4.5 Using a trained network to generate predictions on new data
# Now that the model is trained, we can call *predict* to have it tell us the
# likelihood that a review is positive or negative

#%%
model.predict(xTest)

#%% [markdown]
### 3.4.6 Further Experiments
# The following experiments will help convince you that the architecture choices 
# you’ve made are all fairly reasonable, although there’s still room for improvement:

# * You used two hidden layers. Try using one or three hidden layers, and see how doing so affects validation and test accuracy.
# * Try using layers with more hidden units or fewer hidden units: 32 units, 64 units, and so on.
# * Try using the mse loss function instead of binary_crossentropy.
# * Try using the tanh activation (an activation that was popular in the early days of neural networks) instead of relu.

#%%
# Using 1 hidden layer:

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit(partialXtrain,
                    partialYtrain,
                    epochs=4,
                    batch_size=512,
                    validation_data=(xVal, yVal))

historyDict = history.history

# Plotting the training and validation accuracy
plt.clf()
accValues = historyDict['acc']
valAccValues = historyDict['val_acc']

epochs = range(1, len(accValues) + 1)

plt.plot(epochs, accValues, 'bo', label = 'Training Accuracy')
plt.plot(epochs, valAccValues, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate test data
model.evaluate(xTest, yTest)

#%%
# Using 3 hidden layers:

model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit(partialXtrain,
                    partialYtrain,
                    epochs=4,
                    batch_size=512,
                    validation_data=(xVal, yVal))

historyDict = history.history

# Plotting the training and validation accuracy
plt.clf()
accValues = historyDict['acc']
valAccValues = historyDict['val_acc']

epochs = range(1, len(accValues) + 1)

plt.plot(epochs, accValues, 'bo', label = 'Training Accuracy')
plt.plot(epochs, valAccValues, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate test data
model.evaluate(xTest, yTest)

#%% [markdown]
# Decreasing and increasing layers caused slight changes to loss and accuracy. 
# Interestingly, the single layer model performed (marginally) better than either
# the 2 or 3 layer models.


#%%
# Using MSE instead of binary crossentropy
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'mse',
              metrics = ['accuracy'])

history = model.fit(partialXtrain,
                    partialYtrain,
                    epochs=4,
                    batch_size=512,
                    validation_data=(xVal, yVal))

historyDict = history.history

# Plotting the training and validation accuracy
plt.clf()
accValues = historyDict['acc']
valAccValues = historyDict['val_acc']

epochs = range(1, len(accValues) + 1)

plt.plot(epochs, accValues, 'bo', label = 'Training Accuracy')
plt.plot(epochs, valAccValues, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy with MSE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate test data
model.evaluate(xTest, yTest)

#%% [markdown]
# With MSE as the loss function, loss dropped significantly, but I'm not sure if that's
# because MSE and crossentropy produce values on a different scale or not. Test set accuracy
# was slightly lower than crossentropy.

#%%
# Try using the tanh activation
model = models.Sequential()
model.add(layers.Dense(16, activation = 'tanh', input_shape = (10000, )))
model.add(layers.Dense(16, activation = 'tanh'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit(partialXtrain,
                    partialYtrain,
                    epochs=4,
                    batch_size=512,
                    validation_data=(xVal, yVal))

historyDict = history.history

# Plotting the training and validation accuracy
plt.clf()
accValues = historyDict['acc']
valAccValues = historyDict['val_acc']

epochs = range(1, len(accValues) + 1)

plt.plot(epochs, accValues, 'bo', label = 'Training Accuracy')
plt.plot(epochs, valAccValues, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy with MSE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate test data
model.evaluate(xTest, yTest)

#%% [markdown]
# tanh activation returned similar results to binary crossentropy with MSE loss function

#%%
#Try using layers with more hidden units or fewer hidden units

# 8 hiddent units
model = models.Sequential()
model.add(layers.Dense(8, activation = 'tanh', input_shape = (10000, )))
model.add(layers.Dense(8, activation = 'tanh'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit(partialXtrain,
                    partialYtrain,
                    epochs=4,
                    batch_size=512,
                    validation_data=(xVal, yVal))

historyDict = history.history

# Plotting the training and validation accuracy
plt.clf()
accValues = historyDict['acc']
valAccValues = historyDict['val_acc']

epochs = range(1, len(accValues) + 1)

plt.plot(epochs, accValues, 'bo', label = 'Training Accuracy')
plt.plot(epochs, valAccValues, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy with MSE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate test data
model.evaluate(xTest, yTest)

# 32 hidden units

model = models.Sequential()
model.add(layers.Dense(32, activation = 'tanh', input_shape = (10000, )))
model.add(layers.Dense(32, activation = 'tanh'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit(partialXtrain,
                    partialYtrain,
                    epochs=4,
                    batch_size=512,
                    validation_data=(xVal, yVal))

historyDict = history.history

# Plotting the training and validation accuracy
plt.clf()
accValues = historyDict['acc']
valAccValues = historyDict['val_acc']

epochs = range(1, len(accValues) + 1)

plt.plot(epochs, accValues, 'bo', label = 'Training Accuracy')
plt.plot(epochs, valAccValues, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy with MSE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate test data
model.evaluate(xTest, yTest)

#%% [markdown]
# 8 hidden units produced a slight increase in accuracy, 32 a slight drop.

#### Note that none of these changes in accuracy tell us any base truths about these changes. They all depend on the data

#%%
