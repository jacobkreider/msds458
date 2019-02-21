# -*- coding: utf-8 -*-
# We will randomly define initial values for connection weights, and also randomly select
#   which training data that we will use for a given run.
import os
os.chdir(r"D:\MSDS-git\msds458\Data")

import random
from random import randint

# We want to use the exp function (e to the x); it's part of our transfer function definition
from math import exp

# Biting the bullet and starting to use NumPy for arrays
import numpy as np

# So we can make a separate list from an initial one
import copy

# So we can read in the Grey Box 1 datafiles, which are stored in CSV (comma separated value) format
import csv

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True) 

####################################################################################################
####################################################################################################
#
# This is a tutorial program, designed for those who are learning Python, and specifically using 
#   Python for neural networks applications
#
# It is a multi-component neural network, comprising: 
#  - A "Grey Box," which is subnet with pre-determined weights (read in from data files), that 
#    determines the "big shape class" into which a given input pattern belongs, and
#  - A Convolution Neural Network (CNN) component, currently with a SINGLE (vertical) masking field
#    applied to the original input data (MF1), and retaining the original size of the input data set. 
#    (This is done via expanding the original input data with a one-pixel-width border prior to
#    applying the masking field.)
# The input to the NN is an 81-unit ("pixel") list which represents a 9x9 grid layout for an
#    alphabetic character. 
# This input feeds into both GB1 and the CNN MF1. 
# The network then functions normally with a single hidden layer, where the inputs to the hidden layer 
#   come from Grey Box 1 (GB1) and the result of Masking Field 1 (MF1). 
# The outputs are the set of all 26 (capital) alphabet letters. 
# Ideally, the neural network is trained on both variants and noisy data. This version only provides 
#   a limited (16-character) subset of the inputs, and no variants or noise. 
#
####################################################################################################
####################################################################################################
#
# Code Map: List of Procedures / Functions
# - welcome
#
# == set of basic functions ==
# - computeTransferFnctn
# - computeTransferFnctnDeriv
# - matrixDotProduct
#
# == identify crucial parameters (these can be changed by the user) ==
# - obtainNeuralNetworkSizeSpecs
#    -- initializeWeight
# - initializeWeightArray
# - initializeBiasWeightArray
#
# == obtain data from external data files for GB1
# - readGB1wWeightFile
# - readGB1vWeightFile
# - reconstructGB1wWeightArray
# - reconstructGB1vWeightArray
# - readGB1wBiasWeightFile
# - readGB1vBiasWeightFile
# - reconstructGB1wBiasWeightArray
# - reconstructGB1vBiasWeightArray
#
# == obtain the training data (two possible routes; user selection & random) ==
# - obtainSelectedAlphabetTrainingValues
# - obtainRandomAlphabetTrainingValues
#
# == the feedforward modules ==
#   -- ComputeSingleFeedforwardPassFirstStep
#   -- ComputeSingleFeedforwardPassSecondStep
# - ComputeOutputsAcrossAllTrainingData
#
# == the backpropagation training modules ==
# - backpropagateOutputToHidden
# - backpropagateBiasOutputWeights
# - backpropagateHiddenToInput
# - backpropagateBiasHiddenWeights
#
#
# - main




####################################################################################################
####################################################################################################
#
# Procedure to welcome the user and identify the code
#
####################################################################################################
####################################################################################################


def welcome ():


    print(' ')
    print ('******************************************************************************')
    print(' ')
    print ('Welcome to the Multilayer Perceptron Neural Network')
    print ('  trained using the backpropagation method.')
    print ('Version 0.4, 03/05/2017, A.J. Maren')
    print ('For comments, questions, or bug-fixes, contact: alianna.maren@northwestern.edu')
    print (' ') 
    print ('This program learns to distinguish between broad classes of capital letters')
    print ('It allows users to examine the hidden weights to identify learned features')
    print(' ')
    print ('******************************************************************************')
    print(' ')
    return()

        

####################################################################################################
####################################################################################################
#
# A collection of worker-functions, designed to do specific small tasks
#
####################################################################################################
####################################################################################################

   
#------------------------------------------------------#    

# Compute neuron activation using sigmoid transfer function
def computeTransferFnctn(summedNeuronInput, alpha):
    activation = 1.0 / (1.0 + exp(-alpha*summedNeuronInput)) 
    return activation
  

#------------------------------------------------------# 
    
# Compute derivative of transfer function
def computeTransferFnctnDeriv(NeuronOutput, alpha):
    return alpha*NeuronOutput*(1.0 -NeuronOutput)     


#------------------------------------------------------# 
def matrixDotProduct (matrx1,matrx2):
    dotProduct = np.dot(matrx1,matrx2)
    
    return(dotProduct)    


####################################################################################################
####################################################################################################
#
# Function to obtain the neural network size specifications
#
####################################################################################################
####################################################################################################

def obtainNeuralNetworkSizeSpecs ():

# This procedure operates as a function, as it returns a single value (which really is a list of 
#    three values). It is called directly from 'main.'
#        
# This procedure allows the user to specify the size of the input (I), hidden (H), 
#    and output (O) layers.  
# These values will be stored in a list, the arraySizeList. 
# This list will be used to specify the sizes of two different weight arrays:
#   - wWeights; the Input-to-Hidden array, and
#   - vWeights; the Hidden-to-Output array. 
# However, even though we're calling this procedure, we will still hard-code the array sizes for now.   

# Define parameters for the Grey Box 1 (GB1) subnet
    GB1numInputNodes = 81
    GB1numHiddenNodes = 6
    GB1numOutputNodes = 9  

# Define parameters for the full network        
    numInputNodes = 90
    numHiddenNodes = 6
    numOutputNodes = 16 
          
    print (' ')
    print ('  For the Grey Box 1 subnet, the number of nodes at each level are:')
    print ('    Input: 9x9 (square array) = ', GB1numInputNodes)
    print ('    Hidden: ', GB1numHiddenNodes)
    print ('    Output: ', GB1numOutputNodes)

    print ('  For the full multi-component network, the number of nodes at each level are:')
    print ('    Input: 9x9 (square array) plus the GB 1 outputs =', numInputNodes)
    print ('    Hidden: ', numHiddenNodes)
    print ('    Output: ', numOutputNodes)           
                                    
# We create a list containing the crucial SIZES for the connection weight arrays                
    arraySizeList = (GB1numInputNodes, GB1numHiddenNodes, GB1numOutputNodes, numInputNodes, numHiddenNodes, numOutputNodes)
    
# We return this list to the calling procedure, 'main'.       
    return (arraySizeList)  


####################################################################################################
#
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
#
####################################################################################################

def InitializeWeight ():

    randomNum = random.random()
    weight=1-2*randomNum
#    print weight
           
    return (weight)  



####################################################################################################
####################################################################################################
#
# Function to initialize the node-to-node connection weight arrays
#
####################################################################################################
####################################################################################################

def initializeWeightArray (weightArraySizeList):

# This procedure is also called directly from 'main.'
#        
# This procedure takes in the two parameters, the number of nodes on the bottom (of any two layers), 
#   and the number of nodes in the layer just above it. 
#   It will use these two sizes to create a weight array.
# The weights will initially be assigned random values here, and 
#   this array is passed back to the 'main' procedure. 

    
    numLowerNodes = weightArraySizeList[0] 
    numUpperNodes = weightArraySizeList[1] 
    
#    print ' '
#    print ' inside procedure initializeWeightArray'
#    print ' the number of lower nodes is', numLowerNodes
#    print ' the number of upper nodes is', numUpperNodes    
#
# Initialize the weight variables with random weights    
    weightArray = np.zeros((numUpperNodes,numLowerNodes))    # iniitalize the weight matrix with 0's
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(numLowerNodes):  # number of columns in matrix 2
            weightArray[row,col] = InitializeWeight ()
            
#    print weightArray                  
    
# We return the array to the calling procedure, 'main'.       
    return (weightArray)  


####################################################################################################
####################################################################################################
#
# Function to initialize the bias weight arrays
#
####################################################################################################
####################################################################################################

def initializeBiasWeightArray (numBiasNodes):

# This procedure is also called directly from 'main.'

# Initialize the bias weight variables with random weights    
    biasWeightArray = np.zeros(numBiasNodes)    # iniitalize the weight matrix with 0's
    for node in range(numBiasNodes):  #  Number of nodes in bias weight set
        biasWeightArray[node] = InitializeWeight ()
                  
# Print the entire weights array. 
#    print biasWeightArray
                  
    
# We return the array to the calling procedure, 'main'.       
    return (biasWeightArray)  




####################################################################################################
####################################################################################################
#
# Function to return a trainingDataList
#
####################################################################################################
####################################################################################################

def obtainSelectedAlphabetTrainingValues (dataSet):
    
    # Insert training datasets for 9x9 letter classification
    trainingDataListA0 =  (1,[0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],0,'A',0,'A') # training data list 0 selected for the letter 'A'
    trainingDataListB0 =  (2,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,0],1,'B',1,'B') # training data list 1, letter 'E', courtesy AJM
    trainingDataListC0 =  (3,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],2,'C',2,'C') # training data list 2, letter 'C', courtesy PKVR
    trainingDataListD0 =  (4,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,0],3,'D',3,'O') # training data list 3, letter 'D', courtesy TD
    trainingDataListE0 =  (5,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],4,'E',4,'E') # training data list 4, letter 'E', courtesy BMcD 
    trainingDataListF0 =  (6,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],5,'F',4,'E') # training data list 5, letter 'F', courtesy SK
    trainingDataListG0 =  (7,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],6,'G',2,'C')

    trainingDataListH0 =  (8,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],7,'H',0,'A') # training data list 7, letter 'H', courtesy JC
    trainingDataListI0 =  (9,[0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,1,1,1,1,1,0,0],8,'I',5,'I') # training data list 8, letter 'I', courtesy GR
    trainingDataListJ0 = (10,[0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,1,1,0,0,0],9,'J',5,'I') # training data list 9 selected for the letter 'L', courtesy JT
    trainingDataListK0 = (11,[1,0,0,0,0,0,1,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,1,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0],10,'K',6,'K') # training data list 10 selected for the letter 'K', courtesy EO      
    trainingDataListL0 = (12,[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],11,'L',7,'L') # training data list 11 selected for the letter 'L', courtesy PV
    trainingDataListM0 = (13,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,0,1,0,0,0,1,0,1, 1,0,1,0,0,0,1,0,1, 1,0,0,1,0,1,0,0,1, 1,0,0,1,0,1,0,0,1, 1,1,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1],12,'M',8,'M') # training data list 12 selected for the letter 'M', courtesy GR            
    trainingDataListN0 = (14,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,0,1, 1,0,0,1,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1],13,'N',8,'M') # training data list 13 selected for the letter 'N'
    trainingDataListO0 = (15,[0,1,1,1,1,1,1,1,0, 1,1,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,1,1,1,1,1,1,0],14,'O',3,'O') # training data list 14, letter 'O', courtesy TD
    trainingDataListP0 = (16,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],15,'P',1,'B') # training data list 15, letter 'P', courtesy MT 
    trainingDataListQ0 = (17,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,1],16,'Q',3,'O') # training data list 16, letter 'Q', courtesy AJM (square corners)
    trainingDataListR0 = (18,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],17,'R',1,'B') # training data list 17, letter 'R', courtesy AJM (variant on 'P') 
    trainingDataListS0 = (19,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],18,'S',4,'E') # training data list 18, letter 'S', courtesy RG (square corners)
    trainingDataListT0 = (20,[0,1,1,1,1,1,1,1,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],19,'T',5,'I') # training data list 19, letter 'T', courtesy JR
    trainingDataListU0 = (21,[1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 0,1,1,0,0,0,1,1,0, 0,0,1,1,1,1,1,0,0],20,'U',7,'L') # training data list 20, letter 'U', courtesy JD
 
    trainingDataListW0 = (23, [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0],22,'W',8,'M') # training data list 22, letter 'W', courtesy KW
    trainingDataListX0 = (24,[1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1],23,'X',8,'M') # training data list 23, letter 'X', courtesy JD

    trainingDataListZ0 = (26,[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,],25,'Z',4,'E') # training data list 25, letter 'Z', courtesy ZW
         
    if dataSet == 0: trainingDataList = trainingDataListA0
    elif dataSet == 1: trainingDataList = trainingDataListB0
    elif dataSet == 2: trainingDataList = trainingDataListC0
    elif dataSet == 3: trainingDataList = trainingDataListD0
    elif dataSet == 4: trainingDataList = trainingDataListE0
    elif dataSet == 5: trainingDataList = trainingDataListF0
    elif dataSet == 6: trainingDataList = trainingDataListG0
    elif dataSet == 7: trainingDataList = trainingDataListH0
    elif dataSet == 8: trainingDataList = trainingDataListI0
    elif dataSet == 9: trainingDataList = trainingDataListJ0
    elif dataSet == 10: trainingDataList = trainingDataListK0
    elif dataSet == 11: trainingDataList = trainingDataListL0
    elif dataSet == 12: trainingDataList = trainingDataListM0
    elif dataSet == 13: trainingDataList = trainingDataListN0
    elif dataSet == 14: trainingDataList = trainingDataListO0
    elif dataSet == 15: trainingDataList = trainingDataListP0
    elif dataSet == 16: trainingDataList = trainingDataListQ0
    elif dataSet == 17: trainingDataList = trainingDataListR0
    elif dataSet == 18: trainingDataList = trainingDataListS0
    elif dataSet == 19: trainingDataList = trainingDataListT0
    elif dataSet == 20: trainingDataList = trainingDataListU0
    elif dataSet == 21: trainingDataList = trainingDataListW0
    elif dataSet == 22: trainingDataList = trainingDataListX0
    elif dataSet == 23: trainingDataList = trainingDataListZ0
    else: print('Error occurred')
    
    return (trainingDataList)
       
####################################################################################################
####################################################################################################
#
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
#
####################################################################################################
####################################################################################################


def obtainRandomAlphabetTrainingValues (numTrainingDataSets):

   
    # The training data list will have the  values for the X-OR problem:
    #   - First 81 values will be the 9x9 pixel-grid representation of the letter
    #       represented as a 1-D array (0 or 1 for each)
    #   - 82nd value will be the output class (0 .. totalClasses - 1)
    #   - 83rd value will be the string associated with that class, e.g., 'X'
    # We are starting with five letters in the training set: X, M, N, H, and A
    # Thus there are five choices for training data, which we'll select on random basis
      
    dataSet = random.randint(0, numTrainingDataSets)
    
    # Insert training datasets for 9x9 letter classification
    trainingDataListA0 =  (1,[0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],0,'A',0,'A') # training data list 0 selected for the letter 'A'
    trainingDataListB0 =  (2,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,0],1,'B',1,'B') # training data list 1, letter 'E', courtesy AJM
    trainingDataListC0 =  (3,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],2,'C',2,'C') # training data list 2, letter 'C', courtesy PKVR
    trainingDataListD0 =  (4,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,0],3,'D',3,'O') # training data list 3, letter 'D', courtesy TD
    trainingDataListE0 =  (5,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],4,'E',4,'E') # training data list 4, letter 'E', courtesy BMcD 
    trainingDataListF0 =  (6,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],5,'F',4,'E') # training data list 5, letter 'F', courtesy SK
    trainingDataListG0 =  (7,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],6,'G',2,'C')

    trainingDataListH0 =  (8,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],7,'H',0,'A') # training data list 7, letter 'H', courtesy JC
    trainingDataListI0 =  (9,[0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,1,1,1,1,1,0,0],8,'I',5,'I') # training data list 8, letter 'I', courtesy GR
    trainingDataListJ0 = (10,[0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,1,1,0,0,0],9,'J',5,'I') # training data list 9 selected for the letter 'L', courtesy JT
    trainingDataListK0 = (11,[1,0,0,0,0,0,1,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,1,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0],10,'K',6,'K') # training data list 10 selected for the letter 'K', courtesy EO      
    trainingDataListL0 = (12,[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],11,'L',7,'L') # training data list 11 selected for the letter 'L', courtesy PV
    trainingDataListM0 = (13,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,0,1,0,0,0,1,0,1, 1,0,1,0,0,0,1,0,1, 1,0,0,1,0,1,0,0,1, 1,0,0,1,0,1,0,0,1, 1,1,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1],12,'M',8,'M') # training data list 12 selected for the letter 'M', courtesy GR            
    trainingDataListN0 = (14,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,0,1, 1,0,0,1,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1],13,'N',8,'M') # training data list 13 selected for the letter 'N'
    trainingDataListO0 = (15,[0,1,1,1,1,1,1,1,0, 1,1,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,1,1,1,1,1,1,0],14,'O',3,'O') # training data list 14, letter 'O', courtesy TD
    trainingDataListP0 = (16,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],15,'P',1,'B') # training data list 15, letter 'P', courtesy MT 
    trainingDataListQ0 = (17,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,1],16,'Q',3,'O') # training data list 16, letter 'Q', courtesy AJM (square corners)
    trainingDataListR0 = (18,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],17,'R',1,'B') # training data list 17, letter 'R', courtesy AJM (variant on 'P') 
    trainingDataListS0 = (19,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],18,'S',4,'E') # training data list 18, letter 'S', courtesy RG (square corners)
    trainingDataListT0 = (20,[0,1,1,1,1,1,1,1,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],19,'T',5,'I') # training data list 19, letter 'T', courtesy JR
    trainingDataListU0 = (21,[1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 0,1,1,0,0,0,1,1,0, 0,0,1,1,1,1,1,0,0],20,'U',7,'L') # training data list 20, letter 'U', courtesy JD
 
    trainingDataListW0 = (23, [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0],22,'W',8,'M') # training data list 22, letter 'W', courtesy KW
    trainingDataListX0 = (24,[1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1],23,'X',8,'M') # training data list 23, letter 'X', courtesy JD

    trainingDataListZ0 = (26,[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,],25,'Z',4,'E') # training data list 25, letter 'Z', courtesy ZW
         
    if dataSet == 0: trainingDataList = trainingDataListA0
    elif dataSet == 1: trainingDataList = trainingDataListB0
    elif dataSet == 2: trainingDataList = trainingDataListC0
    elif dataSet == 3: trainingDataList = trainingDataListD0
    elif dataSet == 4: trainingDataList = trainingDataListE0
    elif dataSet == 5: trainingDataList = trainingDataListF0
    elif dataSet == 6: trainingDataList = trainingDataListG0
    elif dataSet == 7: trainingDataList = trainingDataListH0
    elif dataSet == 8: trainingDataList = trainingDataListI0
    elif dataSet == 9: trainingDataList = trainingDataListJ0
    elif dataSet == 10: trainingDataList = trainingDataListK0
    elif dataSet == 11: trainingDataList = trainingDataListL0
    elif dataSet == 12: trainingDataList = trainingDataListM0
    elif dataSet == 13: trainingDataList = trainingDataListN0
    elif dataSet == 14: trainingDataList = trainingDataListO0
    elif dataSet == 15: trainingDataList = trainingDataListP0
    elif dataSet == 16: trainingDataList = trainingDataListQ0
    elif dataSet == 17: trainingDataList = trainingDataListR0
    elif dataSet == 18: trainingDataList = trainingDataListS0
    elif dataSet == 19: trainingDataList = trainingDataListT0
    elif dataSet == 20: trainingDataList = trainingDataListU0
    elif dataSet == 21: trainingDataList = trainingDataListW0
    elif dataSet == 22: trainingDataList = trainingDataListX0
    elif dataSet == 23: trainingDataList = trainingDataListZ0
    else: print('Error occurred')
    
    return (trainingDataList)

####################################################################################################
####################################################################################################
#
# Perform a single feedforward pass
#
####################################################################################################
####################################################################################################



####################################################################################################
#
# Grey Box 1 Function to compute the GB1 hidden node activations as first part of a feedforward pass and return
#   the results in GB1hiddenArray
#
####################################################################################################


def ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray, GB1wWeightArray, GB1wBiasWeightArray):     
            
# iniitalize the sum of inputs into the hidden array with 0's  
    GB1sumIntoHiddenArray = np.zeros(GB1hiddenArrayLength)    
    GB1hiddenArray = np.zeros(GB1hiddenArrayLength)  
    
    print()
    print('length of GB1hiddenArrayLength is:  ', GB1hiddenArrayLength)
    print()

#    print (' ' )
#    print (' the wWeightArray (from GB1)')
#    print (wWeightArray)
#    print (' ') 
#    print (' the inputDataList (going into GB1)')
#    print (inputDataList)

    GB1sumIntoHiddenArray = matrixDotProduct (GB1wWeightArray,GB1inputDataArray)
    print()
    print('length of GB1sumIntoHiddenArray is:  ', len(GB1sumIntoHiddenArray))
    
    
    for node in range(GB1hiddenArrayLength):  #  Number of hidden nodes
        GB1hiddenNodeSumInput=GB1sumIntoHiddenArray[node]+GB1wBiasWeightArray[node]
        GB1hiddenArray[node] = computeTransferFnctn(GB1hiddenNodeSumInput, alpha)

#    print (' ')
#    print ('Back in ComputeSingleFeedforwardPass')
#    print ('The activations for the hidden nodes are:')
#    print ('  Hidden0 = %.4f' % hiddenActivation0, 'Hidden1 = %.4f' % hiddenActivation1)

                                                                                                    
    return (GB1hiddenArray);
  


####################################################################################################
#
# Grey Box 1 Function to compute the output node activations, given the GB1 hidden node activations, 
#   the GB1 hidden-to output connection weights, and the GB1 output bias weights.
# Function returns the array of GB1 output node activations for Grey Box 1.
#
####################################################################################################

def ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray, GB1vWeightArray, GB1vBiasWeightArray):
    
# initialize the sum of inputs into the hidden array with 0's  
    GB1sumIntoOutputArray = np.zeros(GB1hiddenArrayLength)    
    GB1outputArray = np.zeros(GB1outputArrayLength)   

    GB1sumIntoOutputArray = matrixDotProduct (GB1vWeightArray,GB1hiddenArray)
    
    for node in range(GB1outputArrayLength):  #  Number of hidden nodes
        GB1outputNodeSumInput=GB1sumIntoOutputArray[node]+GB1vBiasWeightArray[node]
        GB1outputArray[node] = computeTransferFnctn(GB1outputNodeSumInput, alpha)
                                                                                                   
    return (GB1outputArray);
  



####################################################################################################
#
# Function to compute the hidden node activations as first part of a feedforward pass and return
#   the results in hiddenArray
#
####################################################################################################


def ComputeSingleFeedforwardPassFirstStep (alpha, inputDataList, wWeightArray, biasHiddenWeightArray):
    
# iniitalize the sum of inputs into the hidden array with 0's  
    sumIntoHiddenArray = np.zeros(hiddenArrayLength)    
    hiddenArray = np.zeros(hiddenArrayLength)   

#    print ' ' 
#    print ' the wWeightArray (from GB1)'
#    print wWeightArray
#    print ' ' 
#    print ' the inputDataList (going into GB1)'
#    print inputDataList

    sumIntoHiddenArray = matrixDotProduct (wWeightArray,inputDataList)
    
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        hiddenNodeSumInput=sumIntoHiddenArray[node]+biasHiddenWeightArray[node]
        hiddenArray[node] = computeTransferFnctn(hiddenNodeSumInput, alpha)

#    print ' '
#    print 'Back in ComputeSingleFeedforwardPass'
#    print 'The activations for the hidden nodes are:'
#    print '  Hidden0 = %.4f' % hiddenActivation0, 'Hidden1 = %.4f' % hiddenActivation1

                                                                                                    
    return (hiddenArray);
  


####################################################################################################
#
# Function to compute the output node activations, given the hidden node activations, the hidden-to
#   output connection weights, and the output bias weights.
# Function returns the array of output node activations.
#
####################################################################################################

def ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray, vWeightArray, biasOutputWeightArray):
    
# initialize the sum of inputs into the hidden array with 0's  
    sumIntoOutputArray = np.zeros(hiddenArrayLength)    
    outputArray = np.zeros(outputArrayLength)   

    sumIntoOutputArray = matrixDotProduct (vWeightArray,hiddenArray)
    
    for node in range(outputArrayLength):  #  Number of hidden nodes
        outputNodeSumInput=sumIntoOutputArray[node]+biasOutputWeightArray[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput, alpha)
                                                                                                   
    return (outputArray);
  


####################################################################################################
#
# Procedure to compute the output node activations and determine errors across the entire training
#  data set, and print results.
#
####################################################################################################

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
        GB1hiddenArray = ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray, GB1wWeightArray, GB1wBiasWeightArray)
        GB1outputArray = ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray, GB1vWeightArray, GB1vBiasWeightArray)                        
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
        print ('  Data Set Number', selectedTrainingDataSet, ' for letter ', letterChar, ' with letter number ', letterNum) 

        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, inputDataArray, wWeightArray, biasHiddenWeightArray)

        print (' ')
        print (' The hidden node activations are:')
        print (hiddenArray)

        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray, vWeightArray, biasOutputWeightArray)
    
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
        

                        


####################################################################################################
#**************************************************************************************************#
####################################################################################################
#
#   Backpropgation Section
#
####################################################################################################
#**************************************************************************************************#
####################################################################################################

   
            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the hidden-to-output connection weights
#
####################################################################################################
####################################################################################################


def backpropagateOutputToHidden (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray):

# The first step here applies a backpropagation-based weight change to the hidden-to-output wts v. 
# Core equation for the first part of backpropagation: 
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# -- Hidden(h) = the output of hidden node h. 

# We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight v. 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com


# Unpack array lengths
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]

    transferFuncDerivArray = np.zeros(outputArrayLength)    # iniitalize an array for the transfer function
      
    for node in range(outputArrayLength):  #  Number of hidden nodes
        transferFuncDerivArray[node]=computeTransferFnctnDeriv(outputArray[node], alpha)
 


# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in the equations for the deltas in the connection weights    
#    print ' '
#    print ' The transfer function derivative is: '
#    print transferFuncDerivArray
                        
    deltaVWtArray = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray = np.zeros((outputArrayLength, hiddenArrayLength)) # initialize an array for the new hidden weights
        
    for row in range(outputArrayLength):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes,
        #    and the columns correspond to the number of hidden nodes,
        #    which can be multiplied by the hidden node array (expressed as a column).
        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt = -errorArray[row]*transferFuncDerivArray[row]*hiddenArray[col]
            deltaVWtArray[row,col] = -eta*partialSSE_w_V_Wt
            newVWeightArray[row,col] = vWeightArray[row,col] + deltaVWtArray[row,col]                                                                                     

#    print ' '
#    print ' The previous hidden-to-output connection weights are: '
#    print vWeightArray
#    print ' '
#    print ' The new hidden-to-output connection weights are: '
#    print newVWeightArray

#    PrintAndTraceBackpropagateOutputToHidden (alpha, nu, errorList, actualAllNodesOutputList, 
#    transFuncDerivList, deltaVWtArray, vWeightArray, newHiddenWeightArray)    
                                                                  
                                                                                                                                                                                                            
    return (newVWeightArray);     

            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-output connection weights
#
####################################################################################################
####################################################################################################


def backpropagateBiasOutputWeights (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray):

# The first step here applies a backpropagation-based weight change to the hidden-to-output wts v. 
# Core equation for the first part of backpropagation: 
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# -- Hidden(h) = the output of hidden node h. 

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight biasOutput(o). 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com


# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations 

# The equation for the actual dependence of the Summed Squared Error on a given bias-to-output 
#   weight biasOutput(o) is:
#   partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
# The transfer function derivative (transFuncDeriv) returned from computeTransferFnctnDeriv is given as:
#   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
# Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
#   partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
#   The parameter alpha is included in transFuncDeriv


# Unpack the output array length
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
   
#    print ' '
#    print ' The previous biases for the output nodes are: '
#    print biasOutputWeightArray
#    print ' '
#    print ' The new biases for the output nodes are: '
#    print newBiasOutputWeightArray
                                                                                                                                                
    return (newBiasOutputWeightArray);     


####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the input-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def backpropagateHiddenToInput (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):

# The first step here applies a backpropagation-based weight change to the input-to-hidden wts w. 
# Core equation for the second part of backpropagation: 
# d(SSE)/dw(i,h) = -eta*alpha*F(h)(1-F(h))*Input(i)*sum(v(h,o)*Error(o))
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- w(i,h) is the connection weight w between the input node i and the hidden node h
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1 
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# ---- NOTE: in this second step, the transfer function is applied to the output of the hidden node,
# ------ so that F = F(h)
# -- Hidden(h) = the output of hidden node h (used in computing the derivative of the transfer function). 
# -- Input(i) = the input at node i.

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# Unpack the errorList and the vWeightArray

# We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight w. 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight
 
# For the second step in backpropagation (computing deltas on the input-to-hidden weights)
#   we need the transfer function derivative is applied to the output at the hidden node        

# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations       
    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)    # initialize an array for the transfer function deriv 
      
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
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode] \
            + vWeightArray[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray[outputNode]
             
    deltaWWtArray = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray = np.zeros((hiddenArrayLength, inputArrayLength)) # initialize an array for the new input-to-hidden weights
        
    for row in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts = -transferFuncDerivHiddenArray[row]*inputArray[col]*weightedErrorArray[row]
            deltaWWtArray[row,col] = -eta*partialSSE_w_W_Wts
            newWWeightArray[row,col] = wWeightArray[row,col] + deltaWWtArray[row,col]                                                                                     

#    print ' '
#    print ' The previous hidden-to-output connection weights are: '
#    print wWeightArray
#    print ' '
#    print ' The new hidden-to-output connection weights are: '
#    print newWWeightArray    
       
                                                                    
    return (newWWeightArray);     
    
            
####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def backpropagateBiasHiddenWeights (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
    inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):

# The first step here applies a backpropagation-based weight change to the hidden-to-output wts v. 
# Core equation for the first part of backpropagation: 
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# where:
# -- SSE = sum of squared errors, and only the error associated with a given output node counts
# -- v(h,o) is the connection weight v between the hidden node h and the output node o
# -- alpha is the scaling term within the transfer function, often set to 1
# ---- (this is included in transfFuncDeriv) 
# -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# -- F = transfer function, here using the sigmoid transfer function
# -- Hidden(h) = the output of hidden node h. 

# Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n, 
#   scales amount of change to connection weight

# We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
#   of the SSE w/r/t the weight biasOutput(o). 
# This means, since there is a minus sign in that derivative, that we will add a small amount. 
# (Decrementing is -, applied to a (-), which yields a positive.)

# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com


# Unpack array lengths
    inputArrayLength = arraySizeList [3]
    hiddenArrayLength = arraySizeList [4]
    outputArrayLength = arraySizeList [5]              
                                          
   
# Compute the transfer function derivatives as a function of the output nodes.
# Note: As this is being done after the call to the backpropagation on the hidden-to-output weights,
#   the transfer function derivative computed there could have been used here; the calculations are
#   being redone here only to maintain module independence              

    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength) # initialize array    
    transferFuncDerivOutputArray    = np.zeros(outputArrayLength) # initialize array
    weightedErrorArray              = np.zeros(hiddenArrayLength) # initialize array    

    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv 
    partialSSE_w_BiasHidden      = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray         = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray     = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights
          
    for node in range(hiddenArrayLength):  #  Number of hidden nodes
        transferFuncDerivHiddenArray[node]=computeTransferFnctnDeriv(hiddenArray[node], alpha)      
                  
    for outputNode in range(outputArrayLength):  #  Number of output nodes
        transferFuncDerivOutputArray[outputNode]=computeTransferFnctnDeriv(outputArray[outputNode], alpha) 
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode]*transferFuncDerivOutputArray[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  #  Number of output nodes    
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode]
            + vWeightArray[outputNode, hiddenNode]*errorTimesTFuncDerivOutputArray[outputNode]
            
# Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
#   and so is not included explicitly in these equations 


# ===>>> AJM needs to double-check these equations in the comments area
# ===>>> The code should be fine. 
# The equation for the actual dependence of the Summed Squared Error on a given bias-to-output 
#   weight biasOutput(o) is:
#   partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
# The transfer function derivative (transFuncDeriv) returned from computeTransferFnctnDeriv is given as:
#   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
# Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
#   partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
#   The parameter alpha is included in transFuncDeriv

    for hiddenNode in range(hiddenArrayLength):  #  Number of rows in input-to-hidden weightMatrix           
        partialSSE_w_BiasHidden[hiddenNode] = -transferFuncDerivHiddenArray[hiddenNode]*weightedErrorArray[hiddenNode]
        deltaBiasHiddenArray[hiddenNode] = -eta*partialSSE_w_BiasHidden[hiddenNode]
        newBiasHiddenWeightArray[hiddenNode] = biasHiddenWeightArray[hiddenNode] + deltaBiasHiddenArray[hiddenNode]                                                                                                                                                                                                                                                         
  
                                                                                                                                            
    return (newBiasHiddenWeightArray); 


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

####################################################################################################
####################################################################################################
#
# The following modules expand the boundaries around a chosen letter, and apply a masking filter to 
#   that expanded letter. The result is an array (9x9 in this case) of units, with activation values
#   where 0 <= v <= 1.  
#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
# 
####################################################################################################
####################################################################################################
#
# Function to expand the grid containing a letter by one pixel in each direction
#
####################################################################################################
####################################################################################################

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


####################################################################################################
####################################################################################################
#
# Procedure to print out a letter, given the number of the letter code
#
####################################################################################################
####################################################################################################

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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
####################################################################################################
####################################################################################################
#
# Procedure to print the expanded letter (with a one-pixel border of zeros around the original)  
#
####################################################################################################
####################################################################################################

def printExpandedLetter (expandedLetterArray):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    print (' The expanded letter is:')
    print (expandedLetterArray)   
        
           
    return


####################################################################################################
####################################################################################################
#
# Function to return the letterArray after mask1 has been applied to it
#
####################################################################################################
####################################################################################################


def mask1LetterFunc(expandedLetterArray):

    
    mask1LetterArray = np.zeros(shape=(gridHeight,gridWidth))
    
   
    rowVal = 1
    colVal = 1
        
#    print ' expanded letter array element is:'
#    print '   for array elements 0,0: ', expandedLetterArray[colVal-1,rowVal-1]
#    print '   for array elements 0,1: ', expandedLetterArray[colVal-1,rowVal]
#    print '   for array elements 0,2: ', expandedLetterArray[colVal-1,rowVal+1]    
#    print '   for array elements 1,0: ', expandedLetterArray[colVal,rowVal-1]
#    print '   for array elements 1,1: ', expandedLetterArray[colVal,rowVal]
#    print '   for array elements 1,2: ', expandedLetterArray[colVal,rowVal+1]   
#    print '   for array elements 2,0: ', expandedLetterArray[colVal+1,rowVal-1]
#    print '   for array elements 2,1: ', expandedLetterArray[colVal+1,rowVal]
#    print '   for array elements 2,2: ', expandedLetterArray[colVal+1,rowVal+1]   
    

    
    while rowVal <gridHeight+1: 
#        print ' '
#        print ' for expanded letter row = ', rowVal
        arrayRow = rowVal - 1 
#        print '   for the masked result letter row = ', arrayRow
#        print '   for the maasked result letter column:'
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
#            print ' col = ', arrayCol, 'val = %.3f' % mask1ArrayVal                                                                                                                                          
            colVal = colVal + 1

        rowVal = rowVal + 1
        colVal = 1


#    print ' '
#   print '  The letter with masking applied:'
#    print mask1LetterArray                  
                        
                                        
    return mask1LetterArray 
    

####################################################################################################
####################################################################################################
#
# Procedure to convert the 2x2 array produced by maskLetter into a list and return the list 
#
####################################################################################################
####################################################################################################

def convertArrayToList(mask1LetterArray):

    mask1LetterList = list()

    for row in range(gridHeight):  #  Number of rows in a masked input grid
        for col in range(gridWidth):  # number of columns in a masked input grid
            localGridElement = mask1LetterArray[row,col] 
            mask1LetterList.append(localGridElement)   

    return (mask1LetterList)
    

####################################################################################################
####################################################################################################
#
# The following are a series of functions to access the data files and convert the retrieved data
#   from lists into arrays
#
####################################################################################################
####################################################################################################

####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def readGB1wWeightFile (): 

    GB1wWeightList = list()
    with open("GB1wWeightFile", "r") as infile:

        reader = csv.reader(infile)
        for row in reader:
#        print row
            colnum = 0
            theRow = row
            for col in row:
#                print theRow[colnum], col
                data = float(theRow[colnum])
#                print data
            GB1wWeightList.append(data)
    print (' ')
    print (' Reading the GB1wWeights weights back from the file:')
    print()
    print('readGB1wWeightFile function:')
    print()
    print('Length of GB1wWeightList is: ', len(GB1wWeightList))
#    print (GB1wWeightList)        
    return GB1wWeightList                                                  


####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def readGB1vWeightFile (): 

    GB1vWeightList = list()
    with open('GB1vWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:
#        print row
            colnum = 0
            theRow = row
            for col in row:
#                print theRow[colnum], col
                data = float(theRow[colnum])
#                print data
            GB1vWeightList.append(data)
#    print ' '
#     print ' Reading the weights back from the file:'
#     print GB1vWeightList        
    return GB1vWeightList                                                  

####################################################################################################
#**************************************************************************************************#
####################################################################################################

def reconstructGB1wWeightArray (GB1wWeightList):

    numUpperNodes = GB1hiddenArrayLength
    numLowerNodes = GB1inputArrayLength 
    
    GB1wWeightArray = np.zeros((numUpperNodes,numLowerNodes))    # initialize the weight matrix with 0's     

#    print ' '
#    print ' in reconstructGB1wWeightArray'  
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
#        print ' row: ', row
        for col in range(numLowerNodes):  # number of columns in matrix 2
            localPosition = row*numLowerNodes + col            
            localWeight = GB1wWeightList[localPosition]
#            print '   row: ', row, ' col: ', col, ' localPosition: ', localPosition, ' localWeight: ', localWeight
            GB1wWeightArray[row,col] = localWeight
    print (' ')
    print (' In reconstructWeightArray')
    print()
    print('Length of GB1wWeightArray is: ', len(GB1wWeightArray))
#    print ' The recovered weight matrix is: '
#    print GB1wWeightArray
                                                     
    return GB1wWeightArray  



####################################################################################################
#**************************************************************************************************#
####################################################################################################

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
#    print ' '
#    print ' In reconstructWeightArray'  
#    print ' The recovered weight matrix is: '
#    print GB1vWeightArray
                                                     
    return GB1vWeightArray  
    


####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def readGB1wBiasWeightFile (): 

    GB1wBiasWeightList = list()
    with open('GB1wBiasWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:
#        print row
            colnum = 0
            theRow = row
            for col in row:
#                print theRow[colnum], col
                data = float(theRow[colnum])
#                print data
            GB1wBiasWeightList.append(data)
    print (' ')
    print (' Reading the GB1wBiasWeight bias weights back from the file:')
    print()
    print('Length of GB1wBiasWeightList is: ', len(GB1wBiasWeightList))
#    print (GB1wBiasWeightList)       
    return GB1wBiasWeightList                                                  


####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def readGB1vBiasWeightFile (): 

    GB1vBiasWeightList = list()
    with open('GB1vBiasWeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:
#        print row
            colnum = 0
            theRow = row
            for col in row:
#                print theRow[colnum], col
                data = float(theRow[colnum])
#                print data
            GB1vBiasWeightList.append(data)
#    print ' '
#     print ' Reading the bias weights back from the file:'
#     print GB1vBiasWeightList        
    return GB1vBiasWeightList                                                  

####################################################################################################
#**************************************************************************************************#
####################################################################################################

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



####################################################################################################
#**************************************************************************************************#
####################################################################################################

def reconstructGB1vBiasWeightArray (GB1vBiasWeightList):
    
    GB1vBiasWeightArray = np.zeros(GB1outputArrayLength)    # iniitalize the weight matrix with 0's     
  
    for node in range(GB1outputArrayLength):  #  Number of output bias nodes
            localWeight = GB1vBiasWeightList[node]
            GB1vBiasWeightArray[node] = localWeight
#    print ' '
#    print ' In reconstructGB1vBiasWeightArray'  
#    print ' The recovered output bias weight matrix is: '
#    print GB1vBiasWeightArray
                                                     
    return GB1vBiasWeightArray  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
####################################################################################################
#**************************************************************************************************#
####################################################################################################    
        
            
                    
            
####################################################################################################
#**************************************************************************************************#
####################################################################################################
#
# The MAIN module comprising of calls to:
#   (1) Welcome
#   (2) Obtain neural network size specifications for a three-layer network consisting of:
#       - Input layer
#       - Hidden layer
#       - Output layer (all the sizes are currently hard-coded to two nodes per layer right now)
#   (3) Initialize connection weight values
#       - w: Input-to-Hidden nodes
#       - v: Hidden-to-Output nodes
#   (4) Compute a feedforward pass in two steps
#       - Randomly select a single training data set
#       - Input-to-Hidden
#       - Hidden-to-Output
#       - Compute the error array
#       - Compute the new Summed Squared Error (SSE)
#   (5) Perform a single backpropagation training pass

# (not yet complete; needs updating)
#
####################################################################################################
#**************************************************************************************************#
####################################################################################################


def main():

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

####################################################################################################
# Obtain unit array size in terms of array_length (M) and layers (N)
####################################################################################################                

# This calls the procedure 'welcome,' which just prints out a welcoming message. 
# All procedures need an argument list. 
# This procedure has a list, but it is an empty list; welcome().

    welcome()

    
# Right now, for simplicity, we're going to hard-code the numbers of layers that we have in our 
#   multilayer Perceptron (MLP) neural network. 
# We will have an input layer (I), an output layer (O), and a single hidden layer (H). 

# Define the variable arraySizeList, which is a list. It is initially an empty list. 
# Its purpose is to store the size of the array.

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
    
    print (' ')
    print (' inputArrayLength = ', inputArrayLength)
    print (' hiddenArrayLength = ', hiddenArrayLength)
    print (' outputArrayLength = ', outputArrayLength)        

# Trust that the 2-D array size is the square root oft he inputArrayLength
    gridSizeFloat = (inputArrayLength+1)**(1/2.0) # convert back to the total number of nodes
    gridSize = int(gridSizeFloat+0.1) # add a smidge before converting to integer

    print (' gridSize = ', gridSize)

# Parameters and values for applying a masking field to the input data. 
#   Define the sizes of the letter grid; we are using a 9x9 grid for this example

    
    gridWidth = gridSize
    gridHeight = gridSize
    expandedGridHeight = gridHeight+2
    expandedGridWidth = gridWidth+2 
    eGH = expandedGridHeight
    eGW = expandedGridWidth       

    mask1 = (0,1,0,0,1,0,0,1,0)        

# Parameter definitions for backpropagation, to be replaced with user inputs
    alpha = 1.0
    eta = 1    
    maxNumIterations = 10    # temporarily set to 10 for testing
    epsilon = 0.01
    iteration = 0
    SSE = 0.0
    numTrainingDataSets = 16

                           

####################################################################################################
# 
# Grey Box 1: 
#   Read in the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
#
####################################################################################################                

# Obtain the connection weights from stored data
#
# The GB1wWeightArray is for Input-to-Hidden in Grey Box 1
# The GB1vWeightArray is for Hidden-to-Output in Grey Box 1

# Read the GB1wWeights from stored data back into this program, into a list; return the list
    GB1wWeightList = readGB1wWeightFile()
    
# Convert the GB1wWeight list back into a 2-D weight array
    GB1wWeightArray = reconstructGB1wWeightArray (GB1wWeightList) 
    
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

        
####################################################################################################
# Initialize the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
####################################################################################################                

#
# The wWeightArray is for Input-to-Hidden
# The vWeightArray is for Hidden-to-Output

    wWeightArraySizeList = (inputArrayLength, hiddenArrayLength)
    vWeightArraySizeList = (hiddenArrayLength, outputArrayLength)
    biasHiddenWeightArraySize = hiddenArrayLength
    biasOutputWeightArraySize = outputArrayLength        

# The node-to-node connection weights are stored in a 2-D array
    print (' ')
    print (' about to call initializeWeightArray for the w weights')
    print (' the number of lower and upper nodes is ', wWeightArraySizeList) 
    wWeightArray = initializeWeightArray (wWeightArraySizeList)
    print (' about to call initializeWeightArray for the v weights')
    print (' the number of lower and upper nodes is ', vWeightArraySizeList)     
    vWeightArray = initializeWeightArray (vWeightArraySizeList)

# The bias weights are stored in a 1-D array         
    biasHiddenWeightArray = initializeBiasWeightArray (biasHiddenWeightArraySize)
    biasOutputWeightArray = initializeBiasWeightArray (biasOutputWeightArraySize) 

    
          
####################################################################################################
# Starting the backpropagation work
####################################################################################################     



# Notice in the very beginning of the program, we have 
#   np.set_printoptions(precision=4) (sets number of dec. places in print)
#     and 'np.set_printoptions(suppress=True)', which keeps it from printing in scientific format
#   Debug print: 
#    print
#    print 'The initial weights for this neural network are:'
#    print '       Input-to-Hidden '
#    print wWeightArray
#    print '       Hidden-to-Output'
#    print vWeightArray
#    print ' '
#    print 'The initial bias weights for this neural network are:'
#    print '        Hidden Bias = ', biasHiddenWeightArray                         
#    print '        Output Bias = ', biasOutputWeightArray
  

          
####################################################################################################
# Before we start training, get a baseline set of outputs, errors, and SSE 
####################################################################################################                
                            
    print (' ')
    print ('  Before training:')
    
    ComputeOutputsAcrossAllTrainingData (alpha, numTrainingDataSets, wWeightArray, biasHiddenWeightArray, 
    vWeightArray, biasOutputWeightArray, GB1wWeightArray, GB1wBiasWeightArray, GB1vWeightArray, GB1vBiasWeightArray)                           
                                             
          
####################################################################################################
# Next step - Obtain a single set of randomly-selected training values for alpha-classification 
####################################################################################################                
  
  
    while iteration < maxNumIterations:           

# Increment the iteration count
        iteration = iteration +1

# For any given pass, we re-initialize the training list
#        trainingDataList = (0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0, ' ')                          
        trainingDataList = (0,[0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0],0,' ',0,' ')        
                                                                                  
# Randomly select one of four training sets; the inputs will be randomly assigned to 0 or 1
        dataSet = random.randint(1, numTrainingDataSets)
        
# Optional print / debug:
#        print ' '
#        print ' in main; the dataSet selected is ', dataSet        

# We return the list from the function, with values placed inside the list.           
#        print ' in main, about to call obtainSelected'
        trainingDataList = obtainSelectedAlphabetTrainingValues (dataSet)  
#        print ' in main; the trainingDataList number is ', trainingDataList[0]
                
# Optional print/debug
#        printLetter(trainingDataList)        

#################################################################################################### 
# STEP 1: Push the input data through Grey Box 1 (GB1)          
####################################################################################################                   
                
        GB1inputDataList = []      
        GB1inputDataArray =  np.zeros(GB1inputArrayLength)
        
        thisTrainingDataList = list()                                                                            
        thisTrainingDataList = trainingDataList[1]
        for node in range(GB1inputArrayLength): 
            trainingData = thisTrainingDataList[node]
            GB1inputDataList.append(trainingData)
    

        GB1desiredOutputArray = np.zeros(GB1outputArrayLength)    # iniitalize the output array with 0's
        GB1desiredClass = trainingDataList[4]                 # identify the desired class
        GB1desiredOutputArray[GB1desiredClass] = 1                # set the desired output for that class to 1 

        GB1hiddenArray = ComputeGB1SingleFeedforwardPassFirstStep (alpha, GB1inputDataArray, GB1wWeightArray, GB1wBiasWeightArray)
    
#        print ' '
#        print ' The hidden node activations are:'
#        print hiddenArray

        GB1outputArray = ComputeGB1SingleFeedforwardPassSecondStep (alpha, GB1hiddenArray,GB1vWeightArray, GB1vBiasWeightArray)
    
#        print ' '
#        print ' The output node activations are:'
#        print outputArray                                      
                                                                          
                                                                                                                                                    
####################################################################################################
# STEP 2: Create a masked version of the original input
####################################################################################################     
    
# The next step will be to create a padded version of this letter
#    (Expand boundaries by one pixel all around)
        expandedLetterArray = list()
        expandedLetterArray = expandLetterBoundaries (trainingDataList)

# Optional print/debug
#        printExpandedLetter (expandedLetterArray)
    
        mask1LetterArray = mask1LetterFunc(expandedLetterArray)
        mask1LetterList = convertArrayToList(mask1LetterArray)
                 
          
####################################################################################################
# Step 3: Create the new input array, combining results from GB1 together with the masking filter result(s)
####################################################################################################                

# In this version, we're using the original input training values, NOT the masked filter values

# First, obtain a full input vector
        inputDataList = [] 
        inputDataArray = np.zeros(inputArrayLength) 


# Note: This duplicates some steps done earlier, creating the inputs for GB1

#        print ' ' 
#        print ' about to create training data for the multicomponent network'        
# Fill the first part of the training data list with the usual inputs

        inputDataList = []      
        inputDataArray =  np.zeros(inputArrayLength)
        
        thisTrainingDataList = list()                                                                            
        thisTrainingDataList = trainingDataList[1]    # the 81 input array    
        for node in range(GB1inputArrayLength):         # this should be length 81
            trainingData = thisTrainingDataList[node]  
            inputDataList.append(trainingData)
#        print ' first part inputDataList:'
#        print inputDataList

# Fill the second part of the training data list with the outputs from GB1          
        for node in range(GB1outputArrayLength):  # this should be equal to the number of classes used
            trainingData = GB1outputArray[node]  # this should be the weights saved at the output level from GB1
            inputDataList.append(trainingData)    # appending GB1 output weights to the list of original inputs directly above
#        print ' ' 
#        print ' the whole inputDataList'
#        print inputDataList          

# Create an input array with both the original training data and the outputs from GB1
        for node in range(inputArrayLength):  # defined as number of original inputs plus number of outputs from GB1
            inputDataArray[node] = inputDataList[node]            
 
          
####################################################################################################
# Step 4: Create the new desired output array, using the full number of classes in the input data
####################################################################################################                

# Note: Earlier, the "desired class" was for the big shape (element 4 in the trainingDataList); 
#       Now, the "desired class" is the final classification into an alphabetic character

        desiredOutputArray = np.zeros(outputArrayLength)    # iniitalize the output array with 0's
        desiredClass = trainingDataList[4]                 # identify the desired class number
        desiredOutputArray[desiredClass] = 1                                       
          
####################################################################################################
# Step 5: Do backpropagation training using the combined (GB1 + MF) inputs and full set of desired outputs
####################################################################################################                

                                                                                                       
        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, inputDataArray, wWeightArray, biasHiddenWeightArray)
    
#        print ' '
#        print ' The hidden node activations are:'
#        print hiddenArray

        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray,vWeightArray, biasOutputWeightArray)
    
#        print ' '
#        print ' The output node activations are:'
 #       print outputArray    

#  Optional alternative code for later use:
#  Assign the hidden and output values to specific different variables
#    for node in range(hiddenArrayLength):    
#        actualHiddenOutput[node] = actualAllNodesOutputList [node]
    
#    for node in range(outputArrayLength):    
#        actualOutput[node] = actualAllNodesOutputList [hiddenArrayLength + node]
 
# Initialize the error array
        errorArray = np.zeros(outputArrayLength) 
    
# Determine the error between actual and desired outputs        
        newSSE = 0.0
        for node in range(outputArrayLength):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

#        print ' '
#        print ' The error values are:'
#        print errorArray   
        
# Print the Summed Squared Error  
#        print 'Initial SSE = %.6f' % newSSE
#        SSE = newSSE

         
          
####################################################################################################
# Perform backpropagation
####################################################################################################                
                

# Perform first part of the backpropagation of weight changes    
        newVWeightArray = backpropagateOutputToHidden (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray)
        newBiasOutputWeightArray = backpropagateBiasOutputWeights (alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray) 

# Perform first part of the backpropagation of weight changes       
        newWWeightArray = backpropagateHiddenToInput (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
        inputDataList, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray)

        newBiasHiddenWeightArray = backpropagateBiasHiddenWeights (alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
        inputDataList, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray)  
    
                    
# Assign new values to the weight matrices
# Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
        vWeightArray = newVWeightArray[:]
    
        biasOutputWeightArray = newBiasOutputWeightArray[:]
    
# Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
        wWeightArray = newWWeightArray[:]  
    
        biasHiddenWeightArray = newBiasHiddenWeightArray[:] 
    
# Compute a forward pass, test the new SSE                                                                                
                                                                                                                                    
        hiddenArray = ComputeSingleFeedforwardPassFirstStep (alpha, inputDataArray, wWeightArray, biasHiddenWeightArray)
    
#    print ' '
#    print ' The hidden node activations are:'
#    print hiddenArray

        outputArray = ComputeSingleFeedforwardPassSecondStep (alpha, hiddenArray, vWeightArray, biasOutputWeightArray)
    
#    print ' '
#    print ' The output node activations are:'
#    print outputArray    

    
# Determine the error between actual and desired outputs

        newSSE = 0.0
        for node in range(outputArrayLength):  #  Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node]*errorArray[node]        

#        print ' '
#        print ' The error values are:'
#        print errorArray   
        
# Print the Summed Squared Error  
#        print 'Previous SSE = %.6f' % SSE
#        print 'New SSE = %.6f' % newSSE 
    
#        print ' '
#        print 'Iteration number ', iteration
#        iteration = iteration + 1

        if newSSE < epsilon:

            
            break
    print ('Out of while loop at iteration ', iteration) 
    
####################################################################################################
# After training, get a new comparative set of outputs, errors, and SSE 
####################################################################################################                           

    print (' ')
    print ('  After training:')                  
                                                      
    ComputeOutputsAcrossAllTrainingData (alpha, numTrainingDataSets, wWeightArray, 
biasHiddenWeightArray, vWeightArray, biasOutputWeightArray, GB1wWeightArray, GB1wBiasWeightArray, 
GB1vWeightArray, GB1vBiasWeightArray) 

                 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                                              
####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################                
    
if __name__ == "__main__": main()

####################################################################################################
# End program
#################################################################################################### 

