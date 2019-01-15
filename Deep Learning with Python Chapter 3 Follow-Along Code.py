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
    results = np.zeroes((len(sequences), dimension))
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



#%%
