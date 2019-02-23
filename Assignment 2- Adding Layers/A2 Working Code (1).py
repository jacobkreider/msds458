
# coding: utf-8

# Starting with Dr Maren code as the base. Then going to split it apart below and reshape to give outputs I'm looking for
# 

# In[4]:


from random import seed
import random
from random import randint
import pandas as pd
import os
os.chdir(r"/home/jacob/MSDS-git/msds458/Data")
import csv

# We want to use the exp function (e to the x); 
# it's part of our transfer function definition
from math import exp

# Biting the bullet and starting to use NumPy for arrays
import numpy as np

# So we can make a separate list from an initial one
import copy

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True) 


# In[5]:


# Some "worker function" that eist for specific tasks
# Compute neuron activation using sigmoid transfer function
def computeTransferFnctn(summedNeuronInput, alpha):
    activation = 1.0 / (1.0 + exp(-alpha*summedNeuronInput)) 
    return activation
    
# Compute derivative of transfer function
def computeTransferFnctnDeriv(NeuronOutput, alpha):
    return alpha*NeuronOutput*(1.0 -NeuronOutput)     


def matrixDotProduct (matrx1,matrx2):
    dotProduct = np.dot(matrx1,matrx2)
    
    return(dotProduct) 


# In[6]:


# Function to obtain the neural network size specifications

def obtainNeuralNetworkSizeSpecs (): 
    # Define params for GB1 subnet
    GB1numInputNodes = 81
    GB1numHiddenNodes = 6
    GB1numOutputNodes = 9  
    
    # Define params for full network
    numInputNodes = 90
    numHiddenNodes = 6
    numOutputNodes = 9
    
    print()
    print("  The number of nodes at each level are:")
    print("    Input: 9x9 (square array)")
    print("    Hidden: ", numHiddenNodes)
    print("    Output: ", numOutputNodes)
            
# We create a list containing the crucial SIZES for the connection weight arrays                
    arraySizeList = (GB1numInputNodes, GB1numHiddenNodes, GB1numOutputNodes
                     , numInputNodes, numHiddenNodes, numOutputNodes)
    
# We return this list to the calling procedure, 'main'.       
    return (arraySizeList)  


# In[7]:


# Function to initialize a specific connection weight with a randomly-generated 
# number between 0 & 1

def InitializeWeight ():

    randomNum = random.random()
    weight=1-2*randomNum
#    print weight
           
    return (weight) 


# In[8]:


# Function to initialize the node-to-node connection weight arrays

def initializeWeightArray (weightArraySizeList):
    numLowerNodes = weightArraySizeList[0] 
    numUpperNodes = weightArraySizeList[1] 
   
    weightArray = np.zeros((numUpperNodes,numLowerNodes))    # iniitalize the weight matrix with 0's
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(numLowerNodes):  # number of columns in matrix 2
            weightArray[row,col] = InitializeWeight ()                 
         
    return (weightArray)


# In[9]:


# Function to initialize the bias weight arrays

def initializeBiasWeightArray (numBiasNodes):
    biasWeightArray = np.zeros(numBiasNodes)    # iniitalize the weight matrix with 0's
    for node in range(numBiasNodes):  #  Number of nodes in bias weight set
        biasWeightArray[node] = InitializeWeight ()
      
    return (biasWeightArray) 


# In[10]:


# Function to return a trainingDataList

def obtainSelectedAlphabetTrainingValues (dataSet):
    
# Note: Nine possible output classes: 0 .. 8 trainingDataListXX [4]    
    trainingDataListA0 =  (1,[0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0
                              , 0,1,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1
                              , 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1]
                           ,1,'A',0,'A') 
    trainingDataListB0 =  (2,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1
                              , 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,1,0
                              , 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,0]
                           ,2,'B',1,'B') 
    trainingDataListC0 =  (3,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0
                              , 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0
                              , 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1]
                           ,3,'C',2,'C') 
    trainingDataListD0 =  (4,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1
                              , 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0
                              ,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1
                              ,1,0],4,'D',3,'O') 
    trainingDataListE0 =  (5,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0
                              , 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0
                              ,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1
                              ,1,1],5,'E',4,'E') 
    trainingDataListF0 =  (6,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0
                              , 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0
                              ,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0
                              ,0,0],6,'F',4,'E') 
    trainingDataListG0 =  (7,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0
                              , 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,1,1
                              ,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1
                              ,1,1],7,'G',1,'C')
    trainingDataListH0 =  (8,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1
                              , 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0
                              ,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0
                              ,0,1],8,'H',0,'A') 
    trainingDataListI0 =  (9,[0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0
                              , 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0
                              ,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,1,1,1,1,1
                              ,0,0],9,'I',5,'I') 
    trainingDataListJ0 = (10,[0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0
                              , 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1
                              ,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,1,1,0
                              ,0,0],10,'J',5,'I') 
    trainingDataListK0 = (11,[1,0,0,0,0,0,1,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0,0
                              , 1,0,0,1,0,0,0,0,0, 1,1,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0
                              ,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1
                              ,0,0],11,'K',6,'K')    
    trainingDataListL0 = (12,[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0
                              , 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0
                              ,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1
                              ,1,1],12,'L',7,'L') 
    trainingDataListM0 = (13,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1
                              , 1,0,1,0,0,0,1,0,1, 1,0,1,0,0,0,1,0,1, 1,0,0,1,0,1,0,0
                              ,1, 1,0,0,1,0,1,0,0,1, 1,1,0,0,1,0,0,0,1, 1,0,0,0,1,0,0
                              ,0,1],13,'M',8,'M')            
    trainingDataListN0 = (14,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,0,1
                              , 1,0,0,1,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,0,0,1,0,0
                              ,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0
                              ,0,1],14,'N',8,'M') 
    trainingDataListO0 = (15,[0,1,1,1,1,1,1,1,0, 1,1,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1
                              , 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0
                              ,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,1,1,1,1,1
                              ,1,0],15,'O',3,'O') 
    trainingDataListP0 = (16,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1
                              , 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0
                              ,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0
                              ,0,0],16,'P',1, 'B') 


    if dataSet == 1: trainingDataList = trainingDataListA0
    if dataSet == 2: trainingDataList = trainingDataListB0 
    if dataSet == 3: trainingDataList = trainingDataListC0
    if dataSet == 4: trainingDataList = trainingDataListD0     
    if dataSet == 5: trainingDataList = trainingDataListE0
    if dataSet == 6: trainingDataList = trainingDataListF0 
    if dataSet == 7: trainingDataList = trainingDataListG0 
    if dataSet == 8: trainingDataList = trainingDataListH0
    if dataSet == 9: trainingDataList = trainingDataListI0
    if dataSet == 10: trainingDataList = trainingDataListJ0    

    if dataSet == 11: trainingDataList = trainingDataListK0 
    if dataSet == 12: trainingDataList = trainingDataListL0
    if dataSet == 13: trainingDataList = trainingDataListM0
    if dataSet == 14: trainingDataList = trainingDataListN0 
    if dataSet == 15: trainingDataList = trainingDataListO0 
    if dataSet == 16: trainingDataList = trainingDataListP0  

                           
    return (trainingDataList)  


# ### Feedforward Pass

# In[11]:


def ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray
                                              , GB1wWeightArray, GB1wBiasWeightArray):         
# iniitalize the sum of inputs into the hidden array with 0's  
    GB1sumIntoHiddenArray = np.zeros(GB1hiddenArrayLength)    
    GB1hiddenArray = np.zeros(GB1hiddenArrayLength)   

    GB1sumIntoHiddenArray = matrixDotProduct (GB1wWeightArray,GB1inputDataArray)
    
    for node in range(GB1hiddenArrayLength):  #  Number of hidden nodes
        GB1hiddenNodeSumInput=GB1sumIntoHiddenArray[node]+GB1wBiasWeightArray[node]
        GB1hiddenArray[node] = computeTransferFnctn(GB1hiddenNodeSumInput, alpha)

#    print ' '
#    print 'Back in ComputeSingleFeedforwardPass'
#    print 'The activations for the hidden nodes are:'
#    print '  Hidden0 = %.4f' % hiddenActivation0, 'Hidden1 = %.4f' % hiddenActivation1

                                                                                                    
    return (GB1hiddenArray);


# In[12]:


def ComputeSingleFeedforwardPassFirstStep (alpha, inputDataList, wWeightArray, biasHiddenWeightArray):
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoHiddenArray = np.zeros(hiddenArrayLength)    
    hiddenArray = np.zeros(hiddenArrayLength)   

    sumIntoHiddenArray = matrixDotProduct (wWeightArray,inputDataList)
    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        hiddenNodeSumInput=sumIntoHiddenArray[node]+biasHiddenWeightArray[node]
        hiddenArray[node] = computeTransferFnctn(hiddenNodeSumInput, alpha)

#    print ' '
#    print 'Back in ComputeSingleFeedforwardPass'
#    print 'The activations for the hidden nodes are:'
#    print '  Hidden0 = %.4f' % hiddenActivation0, 'Hidden1 = %.4f' % hiddenActivation1

                                                                                                    
    return (hiddenArray);


# In[13]:


def ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray, vWeightArray, biasOutputWeightArray):
    
# initialize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray = matrixDotProduct (vWeightArray,hiddenArray)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput=sumIntoOutputArray[node]+biasOutputWeightArray[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput, alpha)
                                                                                                   
    return (outputArray)


# In[14]:


def ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray, GB1vWeightArray, GB1vBiasWeightArray):
    
# initialize the sum of inputs into the hidden array with 0's  
    GB1sumIntoOutputArray = np.zeros(GB1hiddenArrayLength)    
    GB1outputArray = np.zeros(GB1outputArrayLength)   

    GB1sumIntoOutputArray = matrixDotProduct (GB1vWeightArray,GB1hiddenArray)
    
    for node in range(GB1outputArrayLength):  #  Number of hidden nodes
        GB1outputNodeSumInput=GB1sumIntoOutputArray[node]+GB1vBiasWeightArray[node]
        GB1outputArray[node] = computeTransferFnctn(GB1outputNodeSumInput, alpha)
                                                                                                   
    return (GB1outputArray);


# For the ComputeOutputsAcrossAllTrainingData in the next section, I actually call this outside of
# a function in a later section-- I never actually use the function. 
# 
# I found it easier to get the returns I wanted outside of a function call-- the same reason I did away with
# the 'Main' procedure. Obviously not what you want to do for production work, but for this it made my life a bit easier.

# In[15]:


# Procedure to compute the output node activations and determine errors across the entire training
#  data set, and print results.

def ComputeOutputsAcrossAllTrainingData (alpha, numTrainingDataSets, wWeightArray, 
biasHiddenWeightArray, vWeightArray, biasOutputWeightArray, GB1wWeightArray, GB1wBiasWeightArray, 
GB1vWeightArray, GB1vBiasWeightArray):

    selectedTrainingDataSet = 1                              
                              

    while selectedTrainingDataSet < numTrainingDataSets + 1: 

        trainingDataList = obtainSelectedAlphabetTrainingValues (selectedTrainingDataSet)
# Note: the trainingDataList is a list comprising several values:
#    - the 0th is the list number 
#    - the 1st is the actual list of the input training values
#    - etc. 


        trainingDataInputList = trainingDataList[1]      

# Obtain the outputs from GB1
            
        GB1inputDataList = [] 
        GB1inputDataArray = np.zeros(GB1inputArrayLength) 
        for node in range(GB1inputArrayLength): 
            trainingData = trainingDataInputList[node]  
            GB1inputDataList.append(trainingData)
            GB1inputDataArray[node] = trainingData

#        print ' ' 
#        print ' before running Grey Box 1'
        GB1hiddenArray = ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray
                                                                   , GB1wWeightArray, GB1wBiasWeightArray)
        GB1outputArray = ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray
                                                                    , GB1vWeightArray, GB1vBiasWeightArray)                        
#        print ' after running Grey Box 1'
#        print ' ' 



# Obtain the outputs from the full multi-component network

# First, obtain a full input vector
        inputDataList = [] 
        inputDataArray = np.zeros(inputArrayLength) 

#        print ' ' 
#        print ' about to create training data for the multicomponent network'        
# Fill the first part of the training data list with the usual inputs
        for node in range(GB1inputArrayLength): 
            trainingData = trainingDataInputList[node]  
            inputDataList.append(trainingData)
#        print ' first part inputDataList:'
#        print inputDataList

# Fill the second part of the training data list with the outputs from GB1          
        for node in range(GB1outputArrayLength): 
            trainingData = GB1outputArray[node]  
            inputDataList.append(trainingData)
#        print ' ' 
#        print ' the whole inputDataList'
#        print inputDataList          

# Create an input array with both the original training data and the outputs from GB1
        for node in range(inputArrayLength): 
            inputDataArray[node] = inputDataList[node]            
#        print ' ' 
#        print ' the whole inputDataArray'
#        print inputDataArray
        
        letterNum = trainingDataList[2] +1
        letterChar = trainingDataList[3]  
        print (' ')
        print ('  Data Set Number', selectedTrainingDataSet, ' for letter ', letterChar
               , ' with letter number ', letterNum) 

        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, inputDataArray
                                                             , wWeightArray, biasHiddenWeightArray)

        print (' ')
        print (' The hidden node activations are:')
        print (hiddenArray)

        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray
                                                              , vWeightArray, biasOutputWeightArray)
    
        print (' ')
        print (' The output node activations are:')
        print (outputArray)   

        desiredOutputArray = np.zeros(outputArrayLength) # iniitalize the output array with 0's
        desiredClass = trainingDataList[4]                 # identify the desired class number
        print()
        print('Desired Class Number is:  ', desiredClass)
        print()
        desiredOutputArray[desiredClass] = 1                # set the desired output for that class to 1
     
        print (' ')
        print (' The desired output array values are: ')
        print (desiredOutputArray)  
       
                        
# Determine the error between actual and desired outputs

# Initialize the error array
        errorArray = np.zeros(outputArrayLength) 
    
        newSSE = 0.0
        for node in range(outputArrayLength):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

        print (' ')
        print (' The error values are:')
        print (errorArray)   
        
# Print the Summed Squared Error  
        print ('New SSE = %.6f' % newSSE) 
         
        selectedTrainingDataSet = selectedTrainingDataSet +1 
        


# ### Backpropagation

# In[16]:


# Backpropagate weight changes onto the hidden-to-output connection weights

def backpropagateOutputToHidden (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray):

# Unpack array lengths
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]

    transferFuncDerivArray = np.zeros(outputArrayLength)    
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
                        
    deltaVWtArray = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray = np.zeros((outputArrayLength, hiddenArrayLength)) # initialize an array for the new hidden weights
        
    for row in range(outputArrayLength):  #  Number of rows in weightMatrix

        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt = -errorArray[row]*transferFuncDerivArray[row]*hiddenArray[col]
            deltaVWtArray[row,col] = -eta*partialSSE_w_V_Wt
            newVWeightArray[row,col] = vWeightArray[row,col] + deltaVWtArray[row,col]                                                                                        
                                                                  
                                                                                                                                                                                                            
    return (newVWeightArray)


# In[17]:


# Backpropagate weight changes onto the bias-to-output connection weights

def backpropagateBiasOutputWeights (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray):

#  Unpack the output array length
    outputArrayLength = arraySizeList [5]

    deltaBiasOutputArray = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray = np.zeros(outputArrayLength) # initialize an array for the new output bias weights
    transferFuncDerivArray = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 

    for node in range(outputArrayLength):  #  Number of nodes in output array (same as number of output bias nodes)    
        partialSSE_w_BiasOutput = -errorArray[node]*transferFuncDerivArray[node]
        deltaBiasOutputArray[node] = -eta*partialSSE_w_BiasOutput  
        newBiasOutputWeightArray[node] =  biasOutputWeightArray[node] + deltaBiasOutputArray[node]           
                                                                                                          
    return (newBiasOutputWeightArray)


# In[18]:


# Backpropagate weight changes onto the input-to-hidden connection weights

def backpropagateHiddenToInput (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          

    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray[node]=computeTransferFnctnDeriv(hiddenArray[node], alpha)
        
    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength) # initialize array
    transferFuncDerivOutputArray    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray              = np.zeros(hiddenArrayLength) # initialize array
      
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray[outputNode]
        
    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode]             + vWeightArray[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray[outputNode]
             
    deltaWWtArray = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray = np.zeros((hiddenArrayLength, inputArrayLength)) # initialize an array for the new input-to-hidden weights
        
    for row in range(hiddenArrayLength):  

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts = -transferFuncDerivHiddenArray[row]*inputArray[col]*weightedErrorArray[row]
            deltaWWtArray[row,col] = -eta*partialSSE_w_W_Wts
            newWWeightArray[row,col] = wWeightArray[row,col] + deltaWWtArray[row,col]                                                                                     
                                        
    return (newWWeightArray)


# In[19]:


# Backpropagate weight changes onto the bias-to-hidden connection weights

def backpropagateBiasHiddenWeights (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):
# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]  
               

    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength)    
    transferFuncDerivOutputArray    = np.zeros(outputArrayLength) 
    weightedErrorArray              = np.zeros(hiddenArrayLength)    

    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)  
    partialSSE_w_BiasHidden      = np.zeros(hiddenArrayLength)  
    deltaBiasHiddenArray         = np.zeros(hiddenArrayLength)  
    newBiasHiddenWeightArray     = np.zeros(hiddenArrayLength)  
          
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray[node]=computeTransferFnctnDeriv(hiddenArray[node], alpha)      
                  
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha) 
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray[hiddenNode] = (weightedErrorArray[hiddenNode]
            + vWeightArray[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray[outputNode])

    for hiddenNode in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix           
        partialSSE_w_BiasHidden[hiddenNode] = -transferFuncDerivHiddenArray[hiddenNode]*weightedErrorArray[hiddenNode]
        deltaBiasHiddenArray[hiddenNode] = -eta*partialSSE_w_BiasHidden[hiddenNode]
        newBiasHiddenWeightArray[hiddenNode] = biasHiddenWeightArray[hiddenNode] + deltaBiasHiddenArray[hiddenNode]                                                                                                                                                                                                                                                         
  
                                                                                                                                            
    return (newBiasHiddenWeightArray)


# ### Expanding Grid Boundaries
# 
# "The following modules expand the boundaries around a chosen letter, and apply a masking filter to that expanded letter. The result is an array (9x9 in this case) of units, with activation values where 0 <= v <= 1.  

# In[20]:


# Function to expand the grid containing a letter by one pixel in each direction

def expandLetterBoundaries (trainingDataList):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    pixelArray = trainingDataList[1]


    expandedLetterArray = np.zeros(shape=(eGH,eGW)) 

    iterAcrossRow = 0
    iterOverAllRows = 0



# For logical completeness: The first element of each row in the expanded letter is set to zero
    iterAcrossRow = 0
#    print ' Zeroth row:'
    while iterAcrossRow < eGW:
        expandedLetterArray[iterOverAllRows,iterAcrossRow] = 0  
#        print iterAcrossRow, expandedLetterArray[iterOverAllRows,iterAcrossRow] 
        iterAcrossRow = iterAcrossRow + 1


# Fill in the elements of the expandedLetterArray; rows 1 .. eGH-1
    
    rowVal = 1
    while rowVal <eGH-1:
#        print 'iterOverAllRows = ', iterOverAllRows

# For the next gridWidth elements in the row, in the expanded letter is set to zero       
        iterAcrossRow = 0
        expandedLetterArray[iterOverAllRows,iterAcrossRow] = 0
        
        iterAcrossRow = 1       
        while iterAcrossRow < eGW-1:
            expandedLetterArray[rowVal,iterAcrossRow] = 0
            #Note: We start counting in the pixelArray at iterAcrossRow-1, because that array 
            #      starts at with the first element at position '0'
            #      and iterAcrossRow is one count beyond that 
            if pixelArray[iterAcrossRow-1+(rowVal-1)*gridWidth] > 0.9: 
                expandedLetterArray[rowVal,iterAcrossRow] = 1
#            print iterAcrossRow, expandedLetterArray[iterOverAllRows,iterAcrossRow]
            iterAcrossRow = iterAcrossRow +1
#        print ' '
        iterAcrossRow = 0  #re-initialize iteration count  
        rowVal = rowVal +1

        # For logical completeness: The last element of each row in the expanded letter is set to zero
        # Note: The last element in the row is at position eGW-1, as the row count starts with 0
    rowVal = eGH-1
    iterAcrossRow = 0
    while iterAcrossRow < eGW-1:
        expandedLetterArray[rowVal,iterAcrossRow] = 0  
#        print iterAcrossRow, expandedLetterArray[iterOverAllRows,iterAcrossRow] 
        iterAcrossRow = iterAcrossRow + 1      
      
#    print ' '    
    return expandedLetterArray


# In[21]:


# Procedure to print out a letter, given the number of the letter code

def printLetter (trainingDataList):    

    print (' ')
    print (' in procedure printLetter')
    print (' ')                         
    print ('The training data set is ', trainingDataList[0])
    print ('The data set is for the letter', trainingDataList[3], ', which is alphabet number ', trainingDataList[2])

    if trainingDataList[0] > 25: (print('This is a variant pattern for letter ', trainingDataList[3])) 

    pixelArray = trainingDataList[1]
                
    iterAcrossRow = 0
    iterOverAllRows = 0
    while iterOverAllRows <gridHeight:
        while iterAcrossRow < gridWidth:
#            arrayElement = pixelArray [iterAcrossRow+iterOverAllRows*gridWidth]
#            if arrayElement <0.9: printElement = ' '
#            else: printElement = 'X'
#            print printElement, 
            iterAcrossRow = iterAcrossRow+1
#        print ' '
        iterOverAllRows = iterOverAllRows + 1
        iterAcrossRow = 0 #re-initialize so the row-print can begin again
    
    return


# In[22]:


# Procedure to print the expanded letter (with a one-pixel border of zeros around the original)  

def printExpandedLetter (expandedLetterArray):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    print (' The expanded letter is:')
    print (expandedLetterArray)   
        
           
    return


# In[23]:


# Function to return the letterArray after mask1 has been applied to it

def mask1LetterFunc(expandedLetterArray):

    
    mask1LetterArray = np.zeros(shape=(gridHeight,gridWidth))
    
   
    rowVal = 1
    colVal = 1
        

    while rowVal <gridHeight+1: 

        arrayRow = rowVal - 1 

        while colVal <gridWidth+1:           
            e0 =  expandedLetterArray[rowVal-1, colVal-1]
            e1 =  expandedLetterArray[rowVal-1, colVal]
            e2 =  expandedLetterArray[rowVal-1, colVal+1]   
            e3 =  expandedLetterArray[rowVal, colVal-1]
            e4 =  expandedLetterArray[rowVal, colVal]
            e5 =  expandedLetterArray[rowVal, colVal+1]   
            e6 =  expandedLetterArray[rowVal+1, colVal-1]
            e7 =  expandedLetterArray[rowVal+1, colVal]
            e8 =  expandedLetterArray[rowVal+1, colVal+1]               
              
            mask1ArrayVal    =  (e0*mask1[0] + e1*mask1[1] + e2*mask1[2] + 
                                e3*mask1[3] + e4*mask1[4] + e5*mask1[5] + 
                                e6*mask1[6] + e7*mask1[7] + e8*mask1[8] ) / 3.0                        
                         
            arrayCol = colVal - 1

            mask1LetterArray[arrayRow,arrayCol] = mask1ArrayVal 
            colVal = colVal + 1

        rowVal = rowVal + 1
        colVal = 1

                                        
    return mask1LetterArray 


# In[24]:


# Procedure to convert the 2x2 array produced by maskLetter into a list and return the list 

def convertArrayToList(mask1LetterArray):

    mask1LetterList = list()

    for row in range(gridHeight):  #  Number of rows in a masked input grid
        for col in range(gridWidth):  # number of columns in a masked input grid
            localGridElement = mask1LetterArray[row,col] 
            mask1LetterList.append(localGridElement)   

    return (mask1LetterList)


# ### Access Needed Data Files from Grey Box Run
# 
# The following are a series of functions to access the data files and convert the retrieved data from lists into arrays
# 

# In[25]:


def readGB1wWeightFile (): 

    GB1wWeightList = list()
    with open("GB1wWeightFile", "r") as infile:

        reader = csv.reader(infile)
        for row in reader:
            colnum = 0
            theRow = row
            for col in row:
                data = float(theRow[colnum])
            GB1wWeightList.append(data)
    
    return GB1wWeightList 


# In[26]:


def readGB1vWeightFile (): 

    GB1vWeightList = list()
    with open('GB1vWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:
            colnum = 0
            theRow = row
            for col in row:
                data = float(theRow[colnum])
            GB1vWeightList.append(data)
       
    return GB1vWeightList     


# In[27]:


def reconstructGB1wWeightArray (GB1wWeightList):

    numUpperNodes = GB1hiddenArrayLength
    numLowerNodes = GB1inputArrayLength 
    
    GB1wWeightArray = np.zeros((numUpperNodes,numLowerNodes))    # initialize the weight matrix with 0's     

  
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(numLowerNodes):  # number of columns in matrix 2
            localPosition = row*numLowerNodes + col            
            localWeight = GB1wWeightList[localPosition]
            GB1wWeightArray[row,col] = localWeight
    print (' ')
    print (' In reconstructWeightArray')
    print()
    print('Length of GB1wWeightArray is: ', len(GB1wWeightArray))

                                                     
    return GB1wWeightArray  


# In[28]:


def reconstructGB1vWeightArray (GB1vWeightList):

    numUpperNodes = GB1outputArrayLength
    numLowerNodes = GB1hiddenArrayLength 
    
    GB1vWeightArray = np.zeros((numUpperNodes,numLowerNodes))    # iniitalize the weight matrix with 0's     
  
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For a hidden-to-output weight matrix, the rows correspond to the number of output nodes
        #    and the columns correspond to the number of hidden nodes.
        #    This creates an OxH matrix, which can be multiplied by the hidden nodes matrix (expressed as a column)

        for col in range(numLowerNodes):  # number of columns in matrix 2
            localPosition = row*numLowerNodes + col
            localWeight = GB1vWeightList[localPosition]
            GB1vWeightArray[row,col] = localWeight

                                                     
    return GB1vWeightArray 


# In[29]:


def readGB1wBiasWeightFile (): 

    GB1wBiasWeightList = list()
    with open('GB1wBiasWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:
            colnum = 0
            theRow = row
            for col in row:
                data = float(theRow[colnum])
            GB1wBiasWeightList.append(data)
    print (' ')
    print (' Reading the GB1wBiasWeight bias weights back from the file:')
    print()
    print('Length of GB1wBiasWeightList is: ', len(GB1wBiasWeightList))
    return GB1wBiasWeightList                                                  


# In[30]:


def readGB1vBiasWeightFile (): 

    GB1vBiasWeightList = list()
    with open('GB1vBiasWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:
            colnum = 0
            theRow = row
            for col in row:
                data = float(theRow[colnum])
            GB1vBiasWeightList.append(data)
       
    return GB1vBiasWeightList  


# In[31]:


def reconstructGB1wBiasWeightArray (GB1wBiasWeightList):

    GB1wBiasWeightArray = np.zeros(GB1hiddenArrayLength)    # initialize the weight matrix with 0's     

    print(' ')
    print (' in reconstructGB1wWeightArray')  
    for node in range(GB1hiddenArrayLength):  #  Number of hidden bias nodes          
            localWeight = GB1wBiasWeightList[node]
            GB1wBiasWeightArray[node] = localWeight
    print (' ')
    print (' In reconstructGB1wBiasWeightArray') 
    print()
    print('The length of GB1wBiasWeightArray is:  ',len(GB1wBiasWeightArray))
    print()
    print (' The recovered hidden bias weight matrix is: ')
    print()
    print (GB1wBiasWeightArray)
                                                     
    return GB1wBiasWeightArray 


# In[32]:


def reconstructGB1vBiasWeightArray (GB1vBiasWeightList):
    
    GB1vBiasWeightArray = np.zeros(GB1outputArrayLength)    # iniitalize the weight matrix with 0's     
  
    for node in range(GB1outputArrayLength):  #  Number of output bias nodes
            localWeight = GB1vBiasWeightList[node]
            GB1vBiasWeightArray[node] = localWeight

                                                     
    return GB1vBiasWeightArray  


# ### The 'Main' Procedure-- split out into separate code snippets

# In[33]:


# Define the global variables        
global inputArrayLength
global hiddenArrayLength
global outputArrayLength
global GB1inputArrayLength
global GB1hiddenArrayLength
global GB1outputArrayLength    
global gridWidth
global gridHeight
global eGH # expandedGridHeight, defined in function expandLetterBoundaries 
global eGW # expandedGridWidth defined in function expandLetterBoundaries 
global mask1 


# In[50]:


arraySizeList = list() # empty list

# Obtain the actual sizes for each layer of the network       
arraySizeList = obtainNeuralNetworkSizeSpecs ()
    
# Unpack the list; ascribe the various elements of the list to the sizes of different network layers
# Note: A word on Python encoding ... the actually length of the array, in each of these three cases, 
#       will be xArrayLength. For example, the inputArrayLength for the 9x9 pixel array is 81. 
#       These values are passed to various procedures. They start filling in actual array values,
#       where the array values start their count at element 0. However, when filling them in using a
#       "for node in range[limit]" statement, the "for" loop fills from 0 up to limit-1. Thus, the
#       original xArrayLength size is preserved.   
GB1inputArrayLength = arraySizeList [0] 
GB1hiddenArrayLength = arraySizeList [1] 
GB1outputArrayLength = arraySizeList [2] 
inputArrayLength = arraySizeList [3] 
hiddenArrayLength = arraySizeList [4] 
outputArrayLength = arraySizeList [5] 
    
print()
print(" inputArrayLength = ", inputArrayLength)
print(" hiddenArrayLength = ", hiddenArrayLength)
print(" outputArrayLength = ", outputArrayLength)  

# Trust that the 2-D array size is the square root oft he inputArrayLength
gridSizeFloat = (inputArrayLength+1)**(1/2.0) # convert back to the total number of nodes
gridSize = int(gridSizeFloat+0.1) # add a smidge before converting to integer

print (' gridSize = ', gridSize)

gridWidth = gridSize
gridHeight = gridSize
expandedGridHeight = gridHeight+2
expandedGridWidth = gridWidth+2 
eGH = expandedGridHeight
eGW = expandedGridWidth       

mask1 = (0,1,0,0,1,0,0,1,0) 

# Parameter definitions for backpropagation, to be replaced with user inputs
alpha = 1.0
eta = 0.5  
maxNumIterations = 5000    # temporarily set to 10 for testing
epsilon = 0.01
iteration = 0
SSE = 0.0
numTrainingDataSets = 16
allHiddenActivations = []


# In[51]:


# Grey Box 1: 

#   Read in the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output

# Read the GB1wWeights from stored data back into this program, into a list; return the list
GB1wWeightList = readGB1wWeightFile()
    
# Convert the GB1wWeight list back into a 2-D weight array
GB1wWeightArray = reconstructGB1wWeightArray(GB1wWeightList) 
    
# Read the GB1vWeights from stored data back into this program, into a list; return the list
GB1vWeightList = readGB1vWeightFile()
    
# Convert the GB1vWeight list back into a 2-D weight array
GB1vWeightArray = reconstructGB1vWeightArray (GB1vWeightList) 

# Obtain the bias weights from stored data

# The GB1wBiasWeightArray is for hidden node biases in Grey Box 1
# The GB1vBiasWeightArray is for output node biases in Grey Box 1

# Read the GB1wBiasWeights from stored data back into this program, into a list; return the list
GB1wBiasWeightList = readGB1wBiasWeightFile()
    
# Convert the GB1wBiasWeight list back into a 2-D weight array
GB1wBiasWeightArray = reconstructGB1wBiasWeightArray (GB1wBiasWeightList) 
    
# Read the GB1vBiasWeights from stored data back into this program, into a list; return the list
GB1vBiasWeightList = readGB1vBiasWeightFile()
    
# Convert the GB1vBiasWeight list back into a 2-D weight array
GB1vBiasWeightArray = reconstructGB1vBiasWeightArray (GB1vBiasWeightList) 


# In[52]:


# Initialize the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
####################################################################################################                
seed(79)

#
# The wWeightArray is for Input-to-Hidden
# The vWeightArray is for Hidden-to-Output

wWeightArraySizeList = (inputArrayLength, hiddenArrayLength)
vWeightArraySizeList = (hiddenArrayLength, outputArrayLength)
biasHiddenWeightArraySize = hiddenArrayLength
biasOutputWeightArraySize = outputArrayLength        

# The node-to-node connection weights are stored in a 2-D array

wWeightArray = initializeWeightArray (wWeightArraySizeList)
  
vWeightArray = initializeWeightArray (vWeightArraySizeList)

# The bias weights are stored in a 1-D array         
biasHiddenWeightArray = initializeBiasWeightArray (biasHiddenWeightArraySize)
biasOutputWeightArray = initializeBiasWeightArray (biasOutputWeightArraySize) 


# In[53]:


# Before we start training, get a baseline set of outputs, errors, and SSE 
####################################################################################################                
                            
print (' ')
print ('  Before training:')
   
ComputeOutputsAcrossAllTrainingData (alpha, numTrainingDataSets, wWeightArray, biasHiddenWeightArray, 
    vWeightArray, biasOutputWeightArray, GB1wWeightArray, GB1wBiasWeightArray, GB1vWeightArray, GB1vBiasWeightArray)                           
    


# In[54]:


# Perform backpropagation during each iteration
# This code pulled from 'Main' and modified to return the new weights for use
# outside of the loop

vWeightArrayPost = np.array([])
wWeightArrayPost = np.array([])
biasHiddenWeightsPost = np.array([])
biasOutputWeightsPost = np.array([])

while iteration < maxNumIterations: 
    
    # Increment the iteration count
    iteration = iteration +1
    
    vWeightArrayPost = np.array([]) # Re-initializing the new weight arrays 
    wWeightArrayPost = np.array([]) # so they only contain the final weights
    biasHiddenWeightsPost = np.array([])
    biasOutputWeightsPost = np.array([])
                          
    # Re-initialize the training list at the start of each iteration
    trainingDataList = (0,[0,0,0,0,0,0,0,0
                        , 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0
                        , 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0
                        , 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0],0,' ', 0, ' ')
    # Populate the training list with a random data set
    dataSet = random.randint(1, numTrainingDataSets)
    trainingDataList = obtainSelectedAlphabetTrainingValues(dataSet)
    
    # Create an input array based on the input training data list
    GB1inputDataList = []
    GB1inputDataArray = np.zeros(GB1inputArrayLength)
    
    # Use the items in index 1 as the training inputs
    thisTrainingDataList = list()
    thisTrainingDataList = trainingDataList[1]
    
    for node in range(GB1inputArrayLength):
        trainingData = thisTrainingDataList[node]
        GB1inputDataList.append(trainingData)
      #inputDataArray[node] = trainingData
      
    # Create desired output array, from 4th element
    GB1desiredOutputArray = np.zeros(GB1outputArrayLength)
    GB1desiredClass = trainingDataList[4]
    GB1desiredOutputArray[GB1desiredClass] = 1
    
    GB1hiddenArray = ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray, GB1wWeightArray, GB1wBiasWeightArray)

    GB1outputArray = ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray,GB1vWeightArray, GB1vBiasWeightArray)

    # STEP 2: Create a masked version of the original input

    expandedLetterArray = list()
    expandedLetterArray = expandLetterBoundaries (trainingDataList)


    
    mask1LetterArray = mask1LetterFunc(expandedLetterArray)
    mask1LetterList = convertArrayToList(mask1LetterArray)
    
    # Step 3: Create the new input array, combining results from GB1 together with the masking filter result(s)

    inputDataList = [] 
    inputDataArray = np.zeros(inputArrayLength) 


# Note: This duplicates some steps done earlier, creating the inputs for GB1
# Fill the first part of the training data list with the usual inputs

    inputDataList = []      
    inputDataArray =  np.zeros(inputArrayLength)
        
    thisTrainingDataList = list()                                                                            
    thisTrainingDataList = trainingDataList[1]    # the 81 input array    
    for node in range(GB1inputArrayLength):         # this should be length 81
        trainingData = thisTrainingDataList[node]  
        inputDataList.append(trainingData)

# Fill the second part of the training data list with the outputs from GB1          
    for node in range(GB1outputArrayLength):  # this should be equal to the number of classes used
        trainingData = GB1outputArray[node]  # this should be the weights saved at the output level from GB1
        inputDataList.append(trainingData)    # appending GB1 output weights to the list of original inputs directly above        

# Create an input array with both the original training data and the outputs from GB1
    for node in range(inputArrayLength):  # defined as number of original inputs plus number of outputs from GB1
        inputDataArray[node] = inputDataList[node]  
        
# Step 4: Create the new desired output array, using the full number of classes in the input data

    desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
    desiredClass = trainingDataList[4]                 # identify the desired class number
    desiredOutputArray[desiredClass] = 1  
    
# Step 5: Do backpropagation training using the combined (GB1 + MF) inputs and full set of desired outputs
    
    
    # Compute feedforward pass
    hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, inputDataArray
                                                         , wWeightArray
                                                         , biasHiddenWeightArray)
    outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray
                                                          ,vWeightArray
                                                          , biasOutputWeightArray)
    # Initialize the rror array
    errorArray = np.zeros(outputArrayLength)
    
    # Determine the error and fill the array plus calculate new SSE
    newSSE = 0.0
    for node in range(outputArrayLength):
      errorArray[node] = desiredOutputArray[node] - outputArray[node]
      newSSE += errorArray[node]*errorArray[node]
      
    # Backpropagation
    
    # Ouput to Hidden backprop
    newVWeightArray = backpropagateOutputToHidden (alpha, eta, arraySizeList
                                                   , errorArray, outputArray
                                                   , hiddenArray, vWeightArray)
    
    
    newBiasOutputWeightArray = backpropagateBiasOutputWeights (alpha, eta
                                                               , arraySizeList
                                                               , errorArray
                                                               , outputArray
                                                               , biasOutputWeightArray) 
    # Hidden to Input backprop
    newWWeightArray = backpropagateHiddenToInput (alpha, eta, arraySizeList
                                                  , errorArray, outputArray
                                                  , hiddenArray, inputDataList
                                                  , vWeightArray, wWeightArray
                                                  , biasHiddenWeightArray
                                                  , biasOutputWeightArray)
    newBiasHiddenWeightArray = backpropagateBiasHiddenWeights (alpha, eta
                                                               , arraySizeList
                                                               , errorArray
                                                               , outputArray
                                                               , hiddenArray
                                                               , inputDataList
                                                               , vWeightArray
                                                               , wWeightArray
                                                               , biasHiddenWeightArray
                                                               , biasOutputWeightArray)
    
    # Update the weight and bias matrices
    # Hidden-to-output update
    vWeightArray = newVWeightArray[:]
    
    
    biasOutputWeightArray = newBiasOutputWeightArray[:]
    
    # Input-to-hidden update
    wWeightArray = newWWeightArray[:]  
    
    biasHiddenWeightArray = newBiasHiddenWeightArray[:] 
    
    # Perform a forward pass with the new weights
    hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, inputDataList
                                                         , wWeightArray
                                                         , biasHiddenWeightArray)
    outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray
                                                          , vWeightArray
                                                          , biasOutputWeightArray)
    
    
    # Check the new SSE
    newSSE = 0.0
    for node in range(outputArrayLength):
      errorArray[node] - desiredOutputArray[node] - outputArray[node]
      newSSE += errorArray[node]*errorArray[node]
      
    if newSSE < epsilon:
      break
# Append to our w weight array
vWeightArrayPost = vWeightArray
wWeightArrayPost = wWeightArray
biasOutputWeightsPost = biasOutputWeightArray
biasHiddenWeightsPost = biasHiddenWeightArray
print("Out of while loop at iteration ", iteration)
    

    
    
  
  
 


# In[55]:


print()
print("  After training:")  

# This is the code from ComputeOutputsAcrossAllTrainingData, modified for
# the outputs I am looking for

selectedTrainingDataSet = 1 
seed(79)
                                
                              
    
allHiddenActivationsPostLearn = np.array([])
allOutputActivationsPostLearn = np.array([])
desiredOutputPost = np.array([])

while selectedTrainingDataSet < numTrainingDataSets + 1: 
    #print()
    #print(" the selected Training Data Set is ", selectedTrainingDataSet)
    trainingDataList = obtainSelectedAlphabetTrainingValues (selectedTrainingDataSet)
# Note: the trainingDataList is a list comprising several values:
#    - the 0th is the list number 
#    - the 1st is the actual list of the input training values
#    - etc. 

    trainingDataInputList = trainingDataList[1]   
    
    # Obtain the outputs from GB1
            
    GB1inputDataList = [] 
    GB1inputDataArray = np.zeros(GB1inputArrayLength) 
    for node in range(GB1inputArrayLength): 
        trainingData = trainingDataInputList[node]  
        GB1inputDataList.append(trainingData)
        GB1inputDataArray[node] = trainingData


    GB1hiddenArray = ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray, GB1wWeightArray, GB1wBiasWeightArray)
    GB1outputArray = ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray, GB1vWeightArray, GB1vBiasWeightArray)   
        
    inputDataList = [] 
    inputDataArray = np.zeros(inputArrayLength) 

    for node in range(GB1inputArrayLength): 
        trainingData = trainingDataInputList[node]  
        inputDataList.append(trainingData)
    
    for node in range(GB1outputArrayLength): 
            trainingData = GB1outputArray[node]  
            inputDataList.append(trainingData)
    
    for node in range(inputArrayLength): 
            inputDataArray[node] = inputDataList[node]    
    

    letterNum = trainingDataList[2] +1
    letterChar = trainingDataList[3]  
    #print()
    #print("  Data Set Number", selectedTrainingDataSet, " for letter ", letterChar, " with letter number ", letterNum) 

    hiddenArray = np.array(ComputeSingleFeedforwardPassFirstStep (alpha, inputDataArray, wWeightArrayPost, biasHiddenWeightsPost))
    allHiddenActivationsPostLearn = np.append(allHiddenActivationsPostLearn, hiddenArray)
    
    outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray, vWeightArrayPost, biasOutputWeightsPost)
    allOutputActivationsPostLearn = np.append(allOutputActivationsPostLearn, outputArray)
    
    desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
    desiredClass = trainingDataList[4]                 # identify the desired class
    desiredOutputArray[desiredClass] = 1
    desiredOutputPost = np.append(desiredOutputPost, desiredOutputArray)
    
    
    
    selectedTrainingDataSet = selectedTrainingDataSet +1
    


# ### Output

# In[56]:


# Create dataframes for pre-training values of all activations

import pandas as pd

preHidden = pd.DataFrame([allHiddenActivationsPreLearn[0:6]
                         ,allHiddenActivationsPreLearn[6:12]
                         ,allHiddenActivationsPreLearn[12:18]
                         ,allHiddenActivationsPreLearn[18:24]
                         ,allHiddenActivationsPreLearn[24:30]
                         ,allHiddenActivationsPreLearn[30:36]
                         ,allHiddenActivationsPreLearn[36:42]
                         ,allHiddenActivationsPreLearn[42:48]
                         ,allHiddenActivationsPreLearn[48:54]
                         ,allHiddenActivationsPreLearn[54:60]
                         ,allHiddenActivationsPreLearn[60:66]
                         ,allHiddenActivationsPreLearn[66:72]
                         ,allHiddenActivationsPreLearn[72:78]
                         ,allHiddenActivationsPreLearn[78:84]
                         ,allHiddenActivationsPreLearn[84:90]
                         ,allHiddenActivationsPreLearn[90:96]]
                         , index = ["A", "B", "C", "D"
                                   ,"E", "F", "G", "H"
                                   ,"I", "J", "K", "L"
                                   ,"M", "N", "O", "P"]
                        , columns = ["H0", "H1", "H2"
                                    , "H3", "H4", "H5"])

preOutput = pd.DataFrame([allOutputActivationsPreLearn[0:9]
                         ,allOutputActivationsPreLearn[9:18]
                         ,allOutputActivationsPreLearn[18:27]
                         ,allOutputActivationsPreLearn[27:36]
                         ,allOutputActivationsPreLearn[36:45]
                         ,allOutputActivationsPreLearn[45:54]
                         ,allOutputActivationsPreLearn[54:63]
                         ,allOutputActivationsPreLearn[63:72]
                         ,allOutputActivationsPreLearn[72:81]
                         ,allOutputActivationsPreLearn[81:90]
                         ,allOutputActivationsPreLearn[90:99]
                         ,allOutputActivationsPreLearn[99:108]
                         ,allOutputActivationsPreLearn[108:117]
                         ,allOutputActivationsPreLearn[117:126]
                         ,allOutputActivationsPreLearn[126:135]
                         ,allOutputActivationsPreLearn[135:144]]
                         , index = ["A", "B", "C", "D"
                                   ,"E", "F", "G", "H"
                                   ,"I", "J", "K", "L"
                                   ,"M", "N", "O", "P"]
                        , columns = ["o0", "o1", "o2"
                                    , "o3", "o4", "o5"
                                    , "o6", "o7", "o8"])

desired = pd.DataFrame([desiredOutput[0:9]
                         ,desiredOutput[9:18]
                         ,desiredOutput[18:27]
                         ,desiredOutput[27:36]
                         ,desiredOutput[36:45]
                         ,desiredOutput[45:54]
                         ,desiredOutput[54:63]
                         ,desiredOutput[63:72]
                         ,desiredOutput[72:81]
                         ,desiredOutput[81:90]
                         ,desiredOutput[90:99]
                         ,desiredOutput[99:108]
                         ,desiredOutput[108:117]
                         ,desiredOutput[117:126]
                         ,desiredOutput[126:135]
                         ,desiredOutput[135:144]]
                         , index = ["A", "B", "C", "D"
                                   ,"E", "F", "G", "H"
                                   ,"I", "J", "K", "L"
                                   ,"M", "N", "O", "P"]
                        , columns = ["o0", "o1", "o2"
                                    , "o3", "o4", "o5"
                                    , "o6", "o7", "o8"])


# In[57]:


# Create data frames for post-output values of all activations

postHidden = pd.DataFrame([allHiddenActivationsPostLearn[0:6]
                         ,allHiddenActivationsPostLearn[6:12]
                         ,allHiddenActivationsPostLearn[12:18]
                         ,allHiddenActivationsPostLearn[18:24]
                         ,allHiddenActivationsPostLearn[24:30]
                         ,allHiddenActivationsPostLearn[30:36]
                         ,allHiddenActivationsPostLearn[36:42]
                         ,allHiddenActivationsPostLearn[42:48]
                         ,allHiddenActivationsPostLearn[48:54]
                         ,allHiddenActivationsPostLearn[54:60]
                         ,allHiddenActivationsPostLearn[60:66]
                         ,allHiddenActivationsPostLearn[66:72]
                         ,allHiddenActivationsPostLearn[72:78]
                         ,allHiddenActivationsPostLearn[78:84]
                         ,allHiddenActivationsPostLearn[84:90]
                         ,allHiddenActivationsPostLearn[90:96]]
                         , index = ["A", "B", "C", "D"
                                   ,"E", "F", "G", "H"
                                   ,"I", "J", "K", "L"
                                   ,"M", "N", "O", "P"]
                        , columns = ["H0", "H1", "H2"
                                    , "H3", "H4", "H5"])

postOutput = pd.DataFrame([allOutputActivationsPostLearn[0:9]
                         ,allOutputActivationsPostLearn[9:18]
                         ,allOutputActivationsPostLearn[18:27]
                         ,allOutputActivationsPostLearn[27:36]
                         ,allOutputActivationsPostLearn[36:45]
                         ,allOutputActivationsPostLearn[45:54]
                         ,allOutputActivationsPostLearn[54:63]
                         ,allOutputActivationsPostLearn[63:72]
                         ,allOutputActivationsPostLearn[72:81]
                         ,allOutputActivationsPostLearn[81:90]
                         ,allOutputActivationsPostLearn[90:99]
                         ,allOutputActivationsPostLearn[99:108]
                         ,allOutputActivationsPostLearn[108:117]
                         ,allOutputActivationsPostLearn[117:126]
                         ,allOutputActivationsPostLearn[126:135]
                         ,allOutputActivationsPostLearn[135:144]]
                         , index = ["A", "B", "C", "D"
                                   ,"E", "F", "G", "H"
                                   ,"I", "J", "K", "L"
                                   ,"M", "N", "O", "P"]
                        , columns = ["o0", "o1", "o2"
                                    , "o3", "o4", "o5"
                                    , "o6", "o7", "o8"])

postDesired = pd.DataFrame([desiredOutput[0:9]
                         ,desiredOutput[9:18]
                         ,desiredOutput[18:27]
                         ,desiredOutput[27:36]
                         ,desiredOutput[36:45]
                         ,desiredOutput[45:54]
                         ,desiredOutput[54:63]
                         ,desiredOutput[63:72]
                         ,desiredOutput[72:81]
                         ,desiredOutput[81:90]
                         ,desiredOutput[90:99]
                         ,desiredOutput[99:108]
                         ,desiredOutput[108:117]
                         ,desiredOutput[117:126]
                         ,desiredOutput[126:135]
                         ,desiredOutput[135:144]]
                         , index = ["A", "B", "C", "D"
                                   ,"E", "F", "G", "H"
                                   ,"I", "J", "K", "L"
                                   ,"M", "N", "O", "P"]
                        , columns = ["o0", "o1", "o2"
                                    , "o3", "o4", "o5"
                                    , "o6", "o7", "o8"])


# In[58]:


# Create DataFrames for the final weights going into and out of each hidden node

HiddenToOutputFinalWeights = pd.DataFrame([vWeightArrayPost[0]
                                          ,vWeightArrayPost[1]
                                          ,vWeightArrayPost[2]
                                          ,vWeightArrayPost[3]
                                          ,vWeightArrayPost[4]
                                          ,vWeightArrayPost[5]
                                          ,vWeightArrayPost[6]
                                          ,vWeightArrayPost[7]
                                          ,vWeightArrayPost[8]
                                          ], columns = ["H0", "H1", "H2", "H3"
                                                       ,"H4","H5"]
                                         , index = ["o0", "o1", "o2", "o3", "o4"
                                                   , "o5", "o6", "o7", "o8"])
HiddenToOutputFinalWeights = HiddenToOutputFinalWeights.transpose()
HiddenToOutputFinalWeights

InputToHiddenFinalWeights = pd.DataFrame([wWeightArrayPost[0]
                                          ,wWeightArrayPost[1]
                                          ,wWeightArrayPost[2]
                                          ,wWeightArrayPost[3]
                                          ,wWeightArrayPost[4]
                                          ,wWeightArrayPost[5]]
                                          , index = ["H0", "H1", "H2", "H3"
                                                       ,"H4","H5"])
InputToHiddenFinalWeights = InputToHiddenFinalWeights.transpose()

InputToHiddenFinalWeights


# ### Results

# In[59]:


HiddenToOutputWeightIndex = round(HiddenToOutputFinalWeights.div(HiddenToOutputFinalWeights.mean(axis=1), axis = 0)*-100,0)
HiddenToOutputWeightIndex


# In[84]:


HiddenToOutputWeightIndex.to_csv("HiddenToOutPutIndexFIXED.csv")


# In[61]:


postHidden


# In[87]:


postOutput.to_csv("A2-postOutput.csv")


# In[90]:


desiredOutputPost


# In[70]:


HiddenToOutputFinalWeights.mean(axis=1)


# In[68]:


HiddenToOutputFinalWeights


# In[82]:


HiddenToOutputWeightIndex.iloc[3,:] = HiddenToOutputWeightIndex.iloc[3,:]*-1


# In[83]:


HiddenToOutputWeightIndex


# In[85]:


hiddenArray


# In[86]:


outputArray

