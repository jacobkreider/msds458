
# coding: utf-8

# Starting with Dr Maren code as the base. Then going to split it apart below and reshape to give outputs I'm looking for
# 

# In[1]:


from random import seed
import random
from random import randint
import pandas as pd

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


# In[2]:


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


# In[3]:


# Function to obtain the neural network size specifications

def obtainNeuralNetworkSizeSpecs (): 

    numInputNodes = 81
    numHiddenNodes = 6
    numOutputNodes = 9   
    print()
    print("  The number of nodes at each level are:")
    print("    Input: 9x9 (square array)")
    print("    Hidden: ", numHiddenNodes)
    print("    Output: ", numOutputNodes)
            
# We create a list containing the crucial SIZES for the connection weight arrays                
    arraySizeList = (numInputNodes, numHiddenNodes, numOutputNodes)
    
# We return this list to the calling procedure, 'main'.       
    return (arraySizeList)  


# In[4]:


# Function to initialize a specific connection weight with a randomly-generated 
# number between 0 & 1

def InitializeWeight ():

    randomNum = random.random()
    weight=1-2*randomNum
#    print weight
           
    return (weight) 


# In[5]:


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


# In[6]:


# Function to initialize the bias weight arrays

def initializeBiasWeightArray (numBiasNodes):
    biasWeightArray = np.zeros(numBiasNodes)    # iniitalize the weight matrix with 0's
    for node in range(numBiasNodes):  #  Number of nodes in bias weight set
        biasWeightArray[node] = InitializeWeight ()
      
    return (biasWeightArray) 


# In[7]:


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

# In[8]:


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


# In[9]:


def ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray, vWeightArray, biasOutputWeightArray):
    
# initialize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray = matrixDotProduct (vWeightArray,hiddenArray)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput=sumIntoOutputArray[node]+biasOutputWeightArray[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput, alpha)
                                                                                                   
    return (outputArray)


# For the ComputeOutputsAcrossAllTrainingData in the next section, I actually call this outside of
# a function in a later section-- I never actually use the function. 
# 
# I found it easier to get the returns I wanted outside of a function call-- the same reason I did away with
# the 'Main' procedure. Obviously not what you want to do for production work, but for this it made my life a bit easier.

# In[10]:


# Procedure to compute the output node activations and determine errors across the entire training
#  data set, and print results.

def ComputeOutputsAcrossAllTrainingData (alpha, numTrainingDataSets, wWeightArray, 
biasHiddenWeightArray, vWeightArray, biasOutputWeightArray):

    selectedTrainingDataSet = 1 
                                
                              
    
    allHiddenActivations = [] 
    while selectedTrainingDataSet < numTrainingDataSets + 1: 
        print()
        print(" the selected Training Data Set is ", selectedTrainingDataSet)
        trainingDataList = obtainSelectedAlphabetTrainingValues (selectedTrainingDataSet)
 

        trainingDataInputList = trainingDataList[1]      
        
        inputDataList = [] 
        for node in range(inputArrayLength): 
            trainingData = trainingDataInputList[node]  
            inputDataList.append(trainingData)

        letterNum = trainingDataList[2]
        letterChar = trainingDataList[3]  
        print()
        print("  Data Set Number", selectedTrainingDataSet, " for letter ", letterChar, " with letter number ", letterNum) 

        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, inputDataList, wWeightArray, biasHiddenWeightArray)

        print()
        print(" The hidden node activations are: ")
        print(hiddenArray)
        

        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray, vWeightArray, biasOutputWeightArray)
    
        print()
        print(" The output node activations are: ")
        print(outputArray)   

        desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
        desiredClass = trainingDataList[4]                 # identify the desired class
        desiredOutputArray[desiredClass] = 1                # set the desired output for that class to 1
     
        print()
        print(" The desired output array values are: ")
        print(desiredOutputArray)  
     
                        
# Determine the error between actual and desired outputs

# Initialize the error array
        errorArray = np.zeros(outputArrayLength) 
    
        newSSE = 0.0
        for node in range(outputArrayLength):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

        print()
        print(" ' The error values are:")
        print(errorArray)   
        
# Print the Summed Squared Error  
        print("New SSE = %.6f" % newSSE) 
         
        selectedTrainingDataSet = selectedTrainingDataSet +1 


# ### Backpropagation

# In[11]:


# Backpropagate weight changes onto the hidden-to-output connection weights

def backpropagateOutputToHidden (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray):

# Unpack array lengths
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]

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


# In[12]:


# Backpropagate weight changes onto the bias-to-output connection weights

def backpropagateBiasOutputWeights (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray):

#  Unpack the output array length
    outputArrayLength = arraySizeList [2]

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


# In[13]:


# Backpropagate weight changes onto the input-to-hidden connection weights

def backpropagateHiddenToInput (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):

# Unpack array lengths
    inputArrayLength = arraySizeList [0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]              
                                          

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


# In[14]:


# Backpropagate weight changes onto the bias-to-hidden connection weights

def backpropagateBiasHiddenWeights (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):
# Unpack array lengths
    inputArrayLength = arraySizeList [0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]  
               

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


# ### The 'Main' Procedure-- split out into separate code snippets

# In[17]:


# Define the global variables        
global inputArrayLength
global hiddenArrayLength
global outputArrayLength
global gridWidth
global gridHeight
global eGH # expandedGridHeight, defined in function expandLetterBoundaries 
global eGW # expandedGridWidth defined in function expandLetterBoundaries 
global mask1    


# In[18]:


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
inputArrayLength = arraySizeList [0] 
hiddenArrayLength = arraySizeList [1] 
outputArrayLength = arraySizeList [2] 
    
print()
print(" inputArrayLength = ", inputArrayLength)
print(" hiddenArrayLength = ", hiddenArrayLength)
print(" outputArrayLength = ", outputArrayLength)        


# Parameter definitions for backpropagation, to be replaced with user inputs
alpha = 1.0
eta = 0.5    
maxNumIterations = 5000    # temporarily set to 10 for testing
epsilon = 0.01
iteration = 0
SSE = 0.0
numTrainingDataSets = 16
allHiddenActivations = []


# In[19]:


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


# In[20]:


# This is the code from ComputeOutputsAcrossAllTrainingData, modified for
# the outputs I am looking for

selectedTrainingDataSet = 1 
seed(79)
                                
                              
    
allHiddenActivationsPreLearn = np.array([])
allOutputActivationsPreLearn = np.array([])
desiredOutput = np.array([])

while selectedTrainingDataSet < numTrainingDataSets + 1: 
    #print()
    #print(" the selected Training Data Set is ", selectedTrainingDataSet)
    trainingDataList = obtainSelectedAlphabetTrainingValues (selectedTrainingDataSet)
# Note: the trainingDataList is a list comprising several values:
#    - the 0th is the list number 
#    - the 1st is the actual list of the input training values
#    - etc. 

    trainingDataInputList = trainingDataList[1]      
        
    inputDataList = [] 
    for node in range(inputArrayLength): 
        trainingData = trainingDataInputList[node]  
        inputDataList.append(trainingData)

    letterNum = trainingDataList[2]
    letterChar = trainingDataList[3]  
    #print()
    #print("  Data Set Number", selectedTrainingDataSet, " for letter ", letterChar, " with letter number ", letterNum) 

    hiddenArray = np.array(ComputeSingleFeedforwardPassFirstStep (alpha, inputDataList, wWeightArray, biasHiddenWeightArray))
    allHiddenActivationsPreLearn = np.append(allHiddenActivationsPreLearn, hiddenArray)
    
    outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray, vWeightArray, biasOutputWeightArray)
    allOutputActivationsPreLearn = np.append(allOutputActivationsPreLearn, outputArray)
    
    desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
    desiredClass = trainingDataList[4]                 # identify the desired class
    desiredOutputArray[desiredClass] = 1
    desiredOutput = np.append(desiredOutput, desiredOutputArray)
    
    
    
    selectedTrainingDataSet = selectedTrainingDataSet +1
    


# In[21]:


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
    trainingDataList = (0,0,0,0,0,0,0,0,0
                        , 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0
                        , 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0
                        , 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0, ' ')
    # Populate the training list with a random data set
    dataSet = random.randint(1, numTrainingDataSets)
    trainingDataList = obtainSelectedAlphabetTrainingValues(dataSet)
    
    # Create an input array based on the input training data list
    inputDataList = []
    inputDataArray = np.zeros(inputArrayLength)
    
    # Use the items in index 1 as the training inputs
    thisTrainingDataList = list()
    thisTrainingDataList = trainingDataList[1]
    
    for node in range(inputArrayLength):
      trainingData = thisTrainingDataList[node]
      inputDataList.append(trainingData)
      inputDataArray[node] = trainingData
      
    # Create desired output array, from 4th element
    desiredOutputArray = np.zeros(outputArrayLength)
    desiredClass = trainingDataList[4]
    desiredOutputArray[desiredClass] = 1
    
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
    

    
    
  
  
 


# In[22]:


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
        
    inputDataList = [] 
    for node in range(inputArrayLength): 
        trainingData = trainingDataInputList[node]  
        inputDataList.append(trainingData)

    letterNum = trainingDataList[2]
    letterChar = trainingDataList[3]  
    #print()
    #print("  Data Set Number", selectedTrainingDataSet, " for letter ", letterChar, " with letter number ", letterNum) 

    hiddenArray = np.array(ComputeSingleFeedforwardPassFirstStep (alpha, inputDataList, wWeightArrayPost, biasHiddenWeightsPost))
    allHiddenActivationsPostLearn = np.append(allHiddenActivationsPostLearn, hiddenArray)
    
    outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray, vWeightArrayPost, biasOutputWeightsPost)
    allOutputActivationsPostLearn = np.append(allOutputActivationsPostLearn, outputArray)
    
    desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
    desiredClass = trainingDataList[4]                 # identify the desired class
    desiredOutputArray[desiredClass] = 1
    desiredOutputPost = np.append(desiredOutputPost, desiredOutputArray)
    
    
    
    selectedTrainingDataSet = selectedTrainingDataSet +1
    


# ### Output

# In[23]:


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


# In[24]:


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


# In[25]:


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

# In[38]:


HiddenToOutputWeightIndex = round(HiddenToOutputFinalWeights.div(HiddenToOutputFinalWeights.mean(axis=1), axis = 0)*-100,0)


# In[40]:


HiddenToOutputWeightIndex.to_csv("HiddenToOutPutIndex.csv")


# In[41]:


postHidden

