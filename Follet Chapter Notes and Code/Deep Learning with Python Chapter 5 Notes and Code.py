#%% [markdown]
### "Deep Learning with Python Chapter 5 Notes and Follow-Along Code"
##### *Notebook created by: Jacob Kreider*

# This notebook covers *Chapter 5: Deep Learning for Computer Vison* from
# *Deep Learning with Python* by Francois Chollet, published by Manning
# Publications, 2018.<br/><br/>

#### Chapter Topics:
# * Undertanding convolutional neural networks (hereafter, covnets)
# * Using data augmentation to mitigate overfitting
# * Using a pretrained covnet to do feature extraction
# * Fine-tuning a pretrained covnet
# * Visualizing what covnets learn and how they make classification decisions

#### 5.1 : Introduction to covnets

# The code starts with creating and training a covnet on the MNIST dataset,
# which we used in Chapter 2. Back then, we used a *densely connected network*
# and achieved a test accuracy of 97.8%. <br/><br/>

# Here, we'll use a stack of *Conv2D* and *MaxPooling2D* layers to improve
# on that test accuracy.

#%%
# Instantiating a small covnet

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu'
                                  , input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

#%% [markdown]
# Covnets take input tensors of shape (height, width, channels), so we configured
# the covnet to take inputs of size (28, 28, 1), which is the format of our images
# in the MNIST database. 

#%%
# Displaying our covnet architecture:

model.summary() 


#%% [markdown]
# In the above output, the output of each layer is a 3D tensor with the same 
# shape as our inputs; however, the width and height dimensions shrink
# at each subsequent layer. The number of channels is controlled by the
# first argument passed (in this model, wither 32 or 64 channels). <br/><br/>
#
# Next, we'll add some layers that feed the final output tensor into a
# densely connected classifier network (like the ones we created in 
# chapters 2 and 3.)


#%%
# Adding a classifier on top of the covnet

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.summary()

#%%
# Npw, we train the above covnet on the MNIST digits. Much of the 
# following code is the same as Chapter 2

from keras.datasets import mnist
from keras.utils import to_categorical

(trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

# reshape and retype the training and test data
trainImages = trainImages.reshape((60000, 28, 28, 1))
trainImages = trainImages.astype('float32') / 255

testImages = testImages.reshape((10000, 28, 28, 1))
testImages = testImages.astype('float32') / 255

trainLabels = to_categorical(trainLabels)
testLabels = to_categorical(testLabels)

# Select the optimizer, loss function, and success metrics for evaluation
model.compile(optimizer = 'rmsprop'
             , loss = 'categorical_crossentropy'
             , metrics = ['accuracy'])

# Fit the model
model.fit(trainImages, trainLabels, epochs=5, batch_size=64)

# Evaluate on the test data
testLoss, testAcc = model.evaluate(testImages, testLabels)

testAcc

#%% [markdown]

##### 5.1.1 The convolution operation
# How did adding the convnet to our network increase accuracy by 1.5 
# absolute percentage points? <br/><br/>

# ***The funadmental difference between a densely connected layer and a 
# convolutional layer is that Dense layers learn global patterns, while
# convolutional layers learn local patterns*** <br/><br/>
#
# Two interesting properties arise from this charcteristic:
# * The patterns they learn are "translation invariant"
# * They can learn spatial hierarchies of patterns.
#
#
# Translation invariant means that once the network learns a pattern
# anywhere, it can recognize it *everywhere*. This allows the network
# to learn on fewer training examples and generalize what they find.
#
#
# A first convolutional layer will learn small, local patters, while the
# next layer will learn larger patterns made up of those smaller patterns.
# Continuing this through the layers allows the network to learn 
# extremely complex and abstract patterns.
#

##### Two key paraneters defining convolutions:
# * Size of the patches etracted from the inputs
# * Depth of the output feature map
#
# 
# *Size of the patches* : the convolution operation extracts patches from
# its input feature map and applies the same transformation to all of them.
# This creates the *output feature map*. Common choices here are either
# 3x3 or 5x5 patches. Depth doesn't matter.
#
#
# *Depth of the output feaure map* : The number of *filters* computed by
# the convolution. Once the input channels have been converted into an
# output feature map, the depth axis no longer stands for, say, specific
# RGB colors, they now stand for filters. Filters encode specific aspects
# of the input data (e.g "presence of a face")

##### What does a convolution do?
#
# A convolution slides the window size (of the patch, 3x3, 5x5, etc) over
# the entire 3D input and extracting the features at each possible point.
# Each 3D patch is then transformed by the *convolutional kernel* (by
# tensor product with the same learned weight matrix) into a 1D vector of
# shape(outputDepth). Then, all of these vectors are reassembled into the
# 3D output map. <br/>
#
# Output width nd height often differ from input width and height due to
# *boarder effects* and the use of *strides*
#
##### Understanding border effects and padding
# When the input feature map and the patch size create areas of the input
# the can't be mapped, the output width/height has to shrink. (For instance,
# a 5x5 input map only has 9 tiles that a 3x3 patch can be centered on.)
# <br/>
#
# To counter this effect, you can use *padding*-- essentially, you add 
# the rows and columns necessary to ensure that all input tiles can be
# mapped. (In keras, this is done via the padding argument, which defaults
# to 'valid' (no padding) and can be set to 'same' (adds padding).) <br/>
#
##### Understanding convolutional strides
# The stride is the distance between two succesive windows when mapping
# with patches. The default is set at 1 (the center tiles of the 
# convolution are all contiguous), but can be set higher. This is rarely
# done in practice. 
#
#### 5.1.2 The max-pooling operation
#
# The role of max pooling is to "*aggressively downsample feature maps*"
# <br/>
#
# Max pooling works similarly to convolution, except that it: 1) local
# patches are transfored using a max tensor operation instead of the 
# convolution kernel; 2) They are usually done with 2x2 windows; and,
# 3) they usually have stride 2. <br/>
#
# Why downsample?
# * Reduces the number of coefficients to process
# * Induces spatial-filter hierarchies by making successive layers look 
# at larger windows
# * Max pooling specifically works well because the max presence of features 
# is more informative than the average presence
#%%
