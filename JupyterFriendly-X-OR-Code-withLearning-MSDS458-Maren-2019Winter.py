#%% [markdown]
# *Notebook Created by: Jacob Kreider*
# 2019-01-18

#### Notes before proceeding:
# The following notebook contains code written by Dr. A.J. Maren for the 
# 2019 Winter term Deep Learning class in the MSDS program at Northwestern
# University. All comments wrapped in single-line or inline quotation marks, 
# except where specified, are hers. Any editing of those comments is done 
# for the sole purpose of displaying the contents in a Jupyter-friendly format.
#
# Non-quoted commentary and code comments are mine alone. No code has been
# changed in any way.
# 
# I will, however, remove placeholder comments and other non-essential
# text when I feel its absence makes the tutorial clearer for me to
# follow. For the complete, unedited version, see the original file.

##### Start of copied script:

#%% [markdown]
##### Initial imports

#%% 
import random # We'll be defining our initial connection weight randomly,
              # and randomly selecting training data
from math import exp # We'll use the exp function (e^x) as part of our 
                     # transfer function
import numpy as np   # For our arrays

#%% [markdown]
###### "
# This is a tutorial program, designed for those who are learning Python, 
# and specifically using Python for neural networks applications.
#
##### SPECIAL NOTE re/ Python code structure: 
# Python uses a 'main' module, which is typically located at the end of the code. 
# There is a short line after that which actual RUNS the 'main' module. 
# All supporting tasks are defined as various procedures and functions. 
# They are stored higher in the code.
#  
# Typically, the most general and global procedures and functions are at 
# the bottom of the code, and the more detail-specific ones are at the top. 
# Thus, if you want to understand a piece of code, you might want to read
# from bottom-to-top (or from more general-to-specific), instead of top-
# to-bottom (detailed-specific-to-general).
#
# Notice that control (e.g., nesting) is defined by spacing. 
# You can define the number of spaces for a tab, but you have to be consistent. 
# Notice that once a procedure is defined, every command within it must 
# be tabbed in.  
###### "

#%% [markdown]
##### Procedure to welcome the user and identify the code
# This procedure is called from the 'main' program.
# Notice that it has an empty parameter list. 
# Procedures require a parameter list, but it can be empty. 

#%%
def welcome(): # The parameter list here is empty, which is ok. 
               # Every function needs a list, even if it is an empty one.
    print()
    print('*******************************************************************')
    print()
    print('Welcome to the Multilayer Perceptron Neural Network')
    print('  trained using the backpropagation method.')
    print('Version 0.2, 01/10/2017, A.J. Maren')
    print('For comments, questions, or bug-fixes,') 
    print('contact: alianna.maren@northwestern.edu')
    print()
    print('*******************************************************************')
    print()
    return() # This statement returns us to 'main'. The parameter list is
             # still empty, as the function did not specify anything to 
             # put in it.

#%% [markdown]
##### A collection of worker-functions, designed to do specific small tasks
# Here, we will define to functions:
# * "Compute neuron activation using sigmoid transfer function"
# * "Compute derivative of transfer function"

#%%
# "Compute neuron activation using sigmoid transfer function"
def computeTransferFnctn(summedNeuronInput, alpha):
    activation = 1.0 / (1.0 + exp(-alpha*summedNeuronInput)) 
    return activation   

# "Compute derivative of transfer function"
def computeTransferFnctnDeriv(NeuronOutput, alpha):
    return alpha*NeuronOutput*(1.0 -NeuronOutput)  

#%% [markdown]
##### Procedure to obtain the neural network size specifications
# The below procedure is called from 'main' and operates as a function.
# It returns a list of three values (the list is technically a single
# value). 
#
# The purpose here is to allow the user to "specify the size of the input
# (I), hidden (H), and output (O) layers." The object 'arraySizeList' will
# store these three values for us.
#
# This list is the basis for the sizes of two different weight arrays:
# * wWeights (the Input-to-Hidden array); and,
# * vWeights (the Hidden-to-Output array)
#
# *Note: For this tutorial, we're hard-coding those weight array sizes.*




#%%
# "Procedure to obtain the neural network size specifications"
def obtainNeuralNetworkSizeSpecs ():
    numInputNodes = 2
    numHiddenNodes = 2
    numOutputNodes = 2   
    print(' ')
    print('This network is set up to run the X-OR problem.')
    print('The numbers of nodes in the input, hidden, and output layers have')
    print('been set to 2 each.')
    # Create an list containing these needed array sizes
    arraySizeList = (numInputNodes, numHiddenNodes, numOutputNodes) 
    return (arraySizeList) #Return this list to 'main'

#%% [markdown]
##### "Function to initialize a specific connection weight with a randomly-generated number between 0 & 1"

#%%
def InitializeWeight ():

    randomNum = random.random()
    weight=1-2*randomNum
#    print(weight)
           
    return (weight)

#%% [markdown]
##### Procedure to initialize the connection weight arrays
###### "
# This procedure is also called directly from 'main.'
#        
# This procedure takes in the two parameters, the number of nodes on the 
# bottom (of any two layers), and the number of nodes in the layer just
# above it. It will use these two sizes to create a weight array.
# 
# The weights will initially be given assigned values here, so that we
# can trace the creation and transfer of this array back to the 'main' 
# procedure. Later, we will randomly define initial weight values.   
#
# These values will be stored in an array, the weightArray. 
#
# This procedure is being called as a function; the returned value is the
# new connection weight matrix. 
#
# Right now, for simplicity and test purposes, the weightArray is set to 
# a single value.
#
# Note my variable-naming convention: 
# * If it is an array, I call it variableNameArray
# * If it is a list, I call it variableNameList
# * It's a lot like calling your dog Roger "rogerDog"
#        and your cat Fluffy "fluffyCat"
#        but until we're better at telling cats from dogs, this helps.
###### "
#%%
# "Procedure to initialize the connection weight arrays"
def initializeWeightArray (weightArraySizeList, debugInitializeOff):
    numBottomNodes = weightArraySizeList[0] # This is unused? Why is it here?
    numUpperNodes = weightArraySizeList[1]  # This is unused? Why is it here? 
    # Initialize the weight variables with random weights
    wt00=InitializeWeight ()
    wt01=InitializeWeight ()
    wt10=InitializeWeight ()
    wt11=InitializeWeight ()    
    weightArray=np.array([[wt00,wt10],[wt01,wt11]])

    # "Debug mode: if debug is set to False, then we DO NOT do the prints"
    if not debugInitializeOff:
       # "Print the weights"
        print(' ')
        print('  Inside initializeWeightArray')
        print('    The weights just initialized are: ')
        print('      weight00 = %.4f,' % wt00)
        print('      weight01 = %.4f,' % wt01)
        print('      weight10 = %.4f,' % wt10)
        print('      weight11 = %.4f,' % wt11)  

    # "Debug mode: if debug is set to False, then we DO NOT do the prints"
    if not debugInitializeOff:
    # "Print the entire weights array: "
        print(' ')
        print('    The weight Array just established is: ', weightArray)
        print(' ') 
        print('    Within this array: ')
        print('      weight00 = %.4f    weight10 = %.4f' % (weightArray[0,0], weightArray[0,1]))
        print('      weight01 = %.4f    weight11 = %.4f' % (weightArray[1,0], weightArray[1,1]))    
        print('  Returning to calling procedure')

    return (weightArray) # Return the created weight array to 'main'

#%% [markdown]
###### Some notes on the above procedure:
###### "
# The weight array is set up so that it lists the rows, and within the rows, the columns
#    wt00   wt10
#    wt10   wt11
#
#
# The sum of weighted terms should be carried out as follows: 
#
#`[wt 00  wt10]` * `[node0]` = wt00*node0 + wt10*node1 = sum-weighted-nodes-to-higher-node0       
#`[wt 10  wt11]` * `[node1]` = wt10*node0 + wt11*node1 = sum-weighted-nodes-to-higher-node1
#
# Notice that the weight positions are being labeled according to how 
# Python numbers elements in an array... so, the first one is in position `[0,0]`.
#
# Notice that the position of the weights in the weightArray is not as would be expected:
#
# wt00 = weight connecting 0th lower-level node to 0th upper-level node = weightArray `[0,0]`<br/>
# wt10 = weight connecting 1st lower-level node to 0th upper-level node = weightArray `[0,1]`<br/>
# wt01 = weight connecting 0th lower-level node to 1st upper-level node = weightArray `[1,0]`<br/>
# wt11 = weight connecting 1st lower-level node to 1st upper-level node = weightArray `[1,1]`<br/>
#
# Notice that wt01 & wt10 are reversed from what we'd expect 
###### "
#<br/>
#
##### Function to initialize the bias weight arrays
#
# This procedure follows the same basic idea as the one above. It initializes
# weight variables with random wights and outputs an array to hold them. 
# The difference here is that this array holds the *bias* weights, not the
# weights between each node connection.



#%%
# Function to initialize the bias weight arrays
def initializeBiasWeightArray (weightArray1DSize):
    numBiasNodes = weightArray1DSize # This is a placeholder step for now.
                                     # Once we start using array operations,
                                     # It will be more important
    # Initialize the weight variables with random weights
    biasWeight0=InitializeWeight ()
    biasWeight1=InitializeWeight ()
      
    biasWeightArray=np.array([biasWeight0,biasWeight1]) 

    return (biasWeightArray)

#%% [markdown]
##### Function to build random training data
# * Training data lists will contain:
# * - Two input values (0 or 1)
# * - Two output values (o or 1)
# * - A value equal to the number of the random-chosen dataset
# 
# The dataset number is included so that we can calculate the SSE against
# the appropriate dataset. We can't stop the training until all the SSEs
# are below a predefined minimum.

#%%
# Function to build random training data
def obtainRandomXORTrainingValues ():
    trainingDataSetNum = random.randint(1, 4)
    if trainingDataSetNum >1.1: # The selection is for training lists 2-4
        if trainingDataSetNum > 2.1: # The selection is for training lists 3 & 4
            if trainingDataSetNum > 3.1: # The selection is for training list 4
                trainingDataList = (1,1,0,1,3) # training data list 4 selected
            else: trainingDataList = (1,0,1,0,2) # training data list 3 selected   
        else: trainingDataList = (0,1,1,0,1) # training data list 2 selected     
    else: trainingDataList = (0,0,0,1,0) # training data list 1 selected 
           
    return (trainingDataList) 

#%% [markdown]
##### "Compute neuron activation"
# "this is the summed weighted inputs after passing through transfer fnctn"


#%%
def computeSingleNeuronActivation(alpha, wt0, wt1, input0, input1, bias, 
                                  debugComputeSingleNeuronActivationOff):
    # Obtain the inputs into the neuron (the sum of weights times inputs)
    summedNeuronInput = wt0*input0+wt1*input1+bias
    # Pass the above result and the transfer function parameter (alpha)
    # into the transfer function
    activation = computeTransferFnctn(summedNeuronInput, alpha)

    if not debugComputeSingleNeuronActivationOff:        
        print(' ')
        print('  In computeSingleNeuronActivation with input0,') 
        print('  input 1 given as: ', input0, ', ', input1)
        print('    The summed neuron input is %.4f' % summedNeuronInput)
        print('    The activation (applied transfer function) for') 
        print('    that neuron is %.4f' % activation)
    return activation

#%% [markdown]
##### Perform a single feedforward pass
# A reminder from Dr. Maren on assigning the weights from the weight arrays:
###### "
# Recall from InitializeWeightArray: 
# Notice that the position of the weights in the weightArray is not as would be expected:
#
#  wt00 = weight connecting 0th lower-level node to 0th upper-level node = weightArray `[0,0]`<br/>
#  wt10 = weight connecting 1st lower-level node to 0th upper-level node = weightArray `[0,1]`<br/>
#  wt01 = weight connecting 0th lower-level node to 1st upper-level node = weightArray `[1,0]`<br/>
#  wt11 = weight connecting 1st lower-level node to 1st upper-level node = weightArray `[1,1]`<br/>
#
# Notice that wt01 & wt10 are reversed from what we'd expect 
###### "

#%%
def ComputeSingleFeedforwardPass (alpha, inputDataList, wWeightArray, 
                                  vWeightArray, biasHiddenWeightArray, 
                                  biasOutputWeightArray, 
                                  debugComputeSingleFeedforwardPassOff):
    input0 = inputDataList[0]
    input1 = inputDataList[1]

    # Assign the input-to-hidden weights to specific variables
    wWt00 = wWeightArray[0,0]
    wWt10 = wWeightArray[0,1]
    wWt01 = wWeightArray[1,0]       
    wWt11 = wWeightArray[1,1] # See note in above comments on these assignments

    # Assign the hidden-to-output weights to specific variables
    vWt00 = vWeightArray[0,0]
    vWt10 = vWeightArray[0,1]
    vWt01 = vWeightArray[1,0]       
    vWt11 = vWeightArray[1,1] # See note in above comments on these assignments

    biasHidden0 = biasHiddenWeightArray[0]
    biasHidden1 = biasHiddenWeightArray[1]
    biasOutput0 = biasOutputWeightArray[0]
    biasOutput1 = biasOutputWeightArray[1]

    # Obtain the activations of the hidden nodes  
    if not debugComputeSingleFeedforwardPassOff:
        debugComputeSingleNeuronActivationOff = False
    else: 
        debugComputeSingleNeuronActivationOff = True
        
    if not debugComputeSingleNeuronActivationOff:
        print(' ')
        print('  For hiddenActivation0 from input0, input1 = ',input0,', ',input1)
    
    hiddenActivation0 = computeSingleNeuronActivation(alpha, wWt00, wWt10, input0,
                 input1, biasHidden0, debugComputeSingleNeuronActivationOff) 
   
    if not debugComputeSingleNeuronActivationOff:
        print(' ')
        print('  For hiddenActivation1 from input0, input1 = ',input0,', ',input1)    
   
    hiddenActivation1 = computeSingleNeuronActivation(alpha, wWt01, wWt11, input0,
                 input1, biasHidden1, debugComputeSingleNeuronActivationOff)

    if not debugComputeSingleFeedforwardPassOff: 
        print(' ')
        print('In computeSingleFeedforwardPass: ')
        print('Input node values: ', input0, ', ', input1)
        print('The activations for the hidden nodes are:')
        print('Hidden0 = %.4f' % hiddenActivation0) 
        print('Hidden1 = %.4f' % hiddenActivation1)

    # Obtain the activations of the output nodes    
    outputActivation0 = computeSingleNeuronActivation(alpha, vWt00, vWt10, 
            hiddenActivation0, hiddenActivation1, biasOutput0, 
            debugComputeSingleNeuronActivationOff)
    outputActivation1 = computeSingleNeuronActivation(alpha, vWt01, vWt11, 
            hiddenActivation0, hiddenActivation1, biasOutput1, 
            debugComputeSingleNeuronActivationOff)

    if not debugComputeSingleFeedforwardPassOff: 
        print(' ')
        print('  Computing the output neuron activations')
        print(' ')
        print('Back in ComputeSingleFeedforwardPass(for hidden-to-output computations)')
        print('  The activations for the output nodes are:')
        print('Output0 = %.4f' % outputActivation0)
        print('Output1 = %.4f' % outputActivation1)

    actualAllNodesOutputList = (hiddenActivation0, hiddenActivation1, 
                                outputActivation0, outputActivation1)
                                                                                                
    return (actualAllNodesOutputList)    

#%% [markdown]
###### "Determine initial and Total Sum Squared Errors"
# This function will return an array containing:
# * The SSE of each of our training data sets plus the total
#
# Note that this function calls ComputeSingleFeedforwardPass, which we
# just created above


#%%
def computeSSE_Values (alpha, SSE_InitialArray, wWeightArray, vWeightArray, 
                biasHiddenWeightArray, biasOutputWeightArray, 
                debugSSE_InitialComputationOff): 
    if not debugSSE_InitialComputationOff:
        debugComputeSingleFeedforwardPassOff = False
    else: 
        debugComputeSingleFeedforwardPassOff = True
    
    # "Compute a single feed-forward pass & obtain the Actual Outputs 
    # for ZEROTH data set"
    inputDataList = (0, 0)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass (alpha, inputDataList, 
                               wWeightArray, vWeightArray, biasHiddenWeightArray, 
                               biasOutputWeightArray, 
                               debugComputeSingleFeedforwardPassOff)        
    actualOutput0 = actualAllNodesOutputList [2]
    actualOutput1 = actualAllNodesOutputList [3] 
    error0 = 0.0 - actualOutput0
    error1 = 1.0 - actualOutput1
    SSE_InitialArray[0] = error0**2 + error1**2

    # "debug print for function":
    if not debugSSE_InitialComputationOff:
        print(' ')
        print('  In computeSSE_Values')

    # "debug print for (0,0)"":
    if not debugSSE_InitialComputationOff: 
        input0 = inputDataList [0]
        input1 = inputDataList [1]
        print(' ')
        print('Actual Node Outputs for (0,0) training set:')
        print('input0 = ', input0, '   input1 = ', input1)
        print('actualOutput0 = %.4f   actualOutput1 = %.4f' %(actualOutput0, actualOutput1))
        print('error0 =        %.4f   error1 =        %.4f' %(error0, error1))
        print('Initial SSE for (0,0) = %.4f' % SSE_InitialArray[0])

    # "Compute a single feed-forward pass & obtain the Actual Outputs 
    # for FIRST data set"
    inputDataList = (0, 1)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass (alpha, inputDataList, 
                               wWeightArray, vWeightArray, biasHiddenWeightArray, 
                               biasOutputWeightArray, 
                               debugComputeSingleFeedforwardPassOff)        
    actualOutput0 = actualAllNodesOutputList [2]
    actualOutput1 = actualAllNodesOutputList [3] 
    error0 = 0.0 - actualOutput0
    error1 = 1.0 - actualOutput1
    SSE_InitialArray[1] = error0**2 + error1**2

    # "debug print for (0,1)":
    if not debugSSE_InitialComputationOff: 
        input0 = inputDataList [0]
        input1 = inputDataList [1]
        print(' ')
        print('Actual Node Outputs for (0,1) training set:')
        print('input0 = ', input0, '   input1 = ', input1)
        print('actualOutput0 = %.4f   actualOutput1 = %.4f' %(actualOutput0, actualOutput1))
        print('error0 =        %.4f   error1 =        %.4f' %(error0, error1))
        print('Initial SSE for (0,1) = %.4f' % SSE_InitialArray[1])

    # "Compute a single feed-forward pass & obtain the Actual Outputs 
    # for SECOND data set"
    inputDataList = (1, 0)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass (alpha, inputDataList, 
                               wWeightArray, vWeightArray, biasHiddenWeightArray, 
                               biasOutputWeightArray, 
                               debugComputeSingleFeedforwardPassOff)        
    actualOutput0 = actualAllNodesOutputList [2]
    actualOutput1 = actualAllNodesOutputList [3] 
    error0 = 0.0 - actualOutput0
    error1 = 1.0 - actualOutput1
    SSE_InitialArray[2] = error0**2 + error1**2

    # "debug print for (1,0):"
    if not debugSSE_InitialComputationOff: 
        input0 = inputDataList [0]
        input1 = inputDataList [1]
        print(' ')
        print('Actual Node Outputs for (1,0) training set:')
        print('input0 = ', input0, '   input1 = ', input1)
        print('actualOutput0 = %.4f   actualOutput1 = %.4f' %(actualOutput0, actualOutput1))
        print('error0 =        %.4f   error1 =        %.4f' %(error0, error1))
        print('Initial SSE for (1,0) = %.4f' % SSE_InitialArray[2])

    # "Compute a single feed-forward pass & obtain the Actual Outputs 
    # for THIRD data set"
    inputDataList = (1, 1)           
    actualAllNodesOutputList = ComputeSingleFeedforwardPass (alpha, inputDataList, 
                               wWeightArray, vWeightArray, biasHiddenWeightArray, 
                               biasOutputWeightArray, 
                               debugComputeSingleFeedforwardPassOff)        
    actualOutput0 = actualAllNodesOutputList [2]
    actualOutput1 = actualAllNodesOutputList [3] 
    error0 = 0.0 - actualOutput0
    error1 = 1.0 - actualOutput1
    SSE_InitialArray[3] = error0**2 + error1**2

    # "debug print for (1,1):"
    if not debugSSE_InitialComputationOff: 
        input0 = inputDataList [0]
        input1 = inputDataList [1]
        print(' ')
        print('Actual Node Outputs for (1,1) training set:')
        print('input0 = ', input0, '   input1 = ', input1)
        print('actualOutput0 = %.4f   actualOutput1 = %.4f' %(actualOutput0, actualOutput1))
        print('error0 =        %.4f   error1 =        %.4f' %(error0, error1))
        print('Initial SSE for (1,1) = %.4f' % SSE_InitialArray[3])

    # "Initialize an array of SSE values"
    SSE_InitialTotal = (SSE_InitialArray[0] + SSE_InitialArray[1] 
                        + SSE_InitialArray[2] + SSE_InitialArray[3]) 

    # "debug print for SSE_InitialTotal:"
    if not debugSSE_InitialComputationOff: 
        print(' ')
        print('  The initial total of the SSEs is %.4f' %SSE_InitialTotal)

    SSE_InitialArray[4] = SSE_InitialTotal
    
    return SSE_InitialArray



#%% [markdown]
### "Backpropagation Section"
#
#### "Optional Debug and Code-Trace Print"
# <br/><br/>
#
# The two functions below will print the hidden node and output node 
# activations, as well as the transfer function derivatives.
# <br/><br/>
# The first function will then compute the deltas at each weight from eta, error, the
# transfer function derivative, and the hidden node activation. Finally, 
# it will print the hidden-to-output connection weights.
# <br/><br/>
# The second function will do the same, but the computation for the deltas
# will use the imput and a "SumTerm for given H" instead of the hidden
# node activation.
# <br/><br/>
##### Backpropagate the hidden-to-output connection weights


#%%
# Code-Trace Print: Hidden-to-Output
def PrintAndTraceBackpropagateOutputToHidden (alpha, eta, errorList, 
                                              actualAllNodesOutputList, 
                                              transFuncDerivList, deltaVWtArray, 
                                              vWeightArray, newVWeightArray):   
    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]
    transFuncDeriv0 = transFuncDerivList[0]
    transFuncDeriv1 = transFuncDerivList[1]

    deltaVWt00 = deltaVWtArray[0,0]
    deltaVWt01 = deltaVWtArray[1,0]
    deltaVWt10 = deltaVWtArray[0,1]
    deltaVWt11 = deltaVWtArray[1,1]    
    
    error0 = errorList[0]
    error1 = errorList[1]

    print(' ')
    print('In Print and Trace for Backpropagation: Hidden to Output Weights')
    print('  Assuming alpha = 1')
    print(' ')
    print('  The hidden node activations are:')
    print('    Hidden node 0: ', '  %.4f' % hiddenNode0, '  Hidden node 1: ', '  %.4f' % hiddenNode1)
    print(' ')
    print('  The output node activations are:')
    print('    Output node 0: ', '  %.3f' % outputNode0, '   Output node 1: ', '  %.3f' % outputNode1)
    print(' ')
    print('  The transfer function derivatives are: ')
    print('    Deriv-F(0): ', '     %.3f' % transFuncDeriv0, '   Deriv-F(1): ', '     %.3f' % transFuncDeriv1)

    print(' ') 
    print('The computed values for the deltas are: ')
    print('                eta  *  error  *   trFncDeriv *   hidden')
    print('  deltaVWt00 = ',' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % transFuncDeriv0, '  * %.4f' % hiddenNode0)
    print('  deltaVWt01 = ',' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % transFuncDeriv1, '  * %.4f' % hiddenNode0)                      
    print('  deltaVWt10 = ',' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % transFuncDeriv0, '  * %.4f' % hiddenNode1)
    print('  deltaVWt11 = ',' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % transFuncDeriv1, '  * %.4f' % hiddenNode1)
    print(' ')
    print('Values for the hidden-to-output connection weights:')
    print('           Old:     New:      eta*Delta:')
    print('[0,0]:   %.4f' % vWeightArray[0,0], '  %.4f' % newVWeightArray[0,0], '  %.4f' % deltaVWtArray[0,0])
    print('[0,1]:   %.4f' % vWeightArray[1,0], '  %.4f' % newVWeightArray[1,0], '  %.4f' % deltaVWtArray[1,0])
    print('[1,0]:   %.4f' % vWeightArray[0,1], '  %.4f' % newVWeightArray[0,1], '  %.4f' % deltaVWtArray[0,1])
    print('[1,1]:   %.4f' % vWeightArray[1,1], '  %.4f' % newVWeightArray[1,1], '  %.4f' % deltaVWtArray[1,1])

#%% [markdown]

##### "Backpropagate the input-to-hidden connection weights"


#%%
# "Code-Trace Print: Backpropagate the input-to-hidden connection weights"
def PrintAndTraceBackpropagateHiddenToInput (alpha, eta, inputDataList, errorList, 
                                             actualAllNodesOutputList, 
                                             transFuncDerivHiddenList, 
                                             transFuncDerivOutputList, 
                                             deltaWWtArray, vWeightArray, 
                                             wWeightArray, newWWeightArray, 
                                             biasWeightArray):

    inputNode0 = inputDataList[0]
    inputNode1 = inputDataList[1]    
    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]
    transFuncDerivHidden0 = transFuncDerivHiddenList[0]
    transFuncDerivHidden1 = transFuncDerivHiddenList[1]
    transFuncDerivOutput0 = transFuncDerivOutputList[0]
    transFuncDerivOutput1 = transFuncDerivOutputList[1] 

    wWt00 = wWeightArray[0,0]
    wWt01 = wWeightArray[1,0]
    wWt10 = wWeightArray[0,1]       
    wWt11 = wWeightArray[1,1]
    
    deltaWWt00 = deltaWWtArray[0,0]
    deltaWWt01 = deltaWWtArray[1,0]
    deltaWWt10 = deltaWWtArray[0,1]
    deltaWWt11 = deltaWWtArray[1,1] 
    
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[1,0]
    vWt10 = vWeightArray[0,1]
    vWt11 = vWeightArray[1,1]                
    
    error0 = errorList[0]
    error1 = errorList[1]

    errorTimesTransD0 = error0*transFuncDerivOutput0
    errorTimesTransD1 = error1*transFuncDerivOutput1    
            
    biasHidden0 = biasWeightArray[0,0]
    biasHidden1 = biasWeightArray[0,1]                                      

    partialSSE_w_Wwt00 = -transFuncDerivHidden0*inputNode0*(vWt00*error0 + vWt01*error1)                                                             
    partialSSE_w_Wwt01 = -transFuncDerivHidden1*inputNode0*(vWt10*error0 + vWt11*error1)
    partialSSE_w_Wwt10 = -transFuncDerivHidden0*inputNode1*(vWt00*error0 + vWt01*error1)
    partialSSE_w_Wwt11 = -transFuncDerivHidden1*inputNode1*(vWt10*error0 + vWt11*error1)
    
    sumTermH0 = vWt00*error0+vWt01*error1
    sumTermH1 = vWt10*error0+vWt11*error1

    print(' ')
    print('In Print and Trace for Backpropagation: Input to Hidden Weights')
    print('  Assuming alpha = 1')
    print(' ')
    print('  The hidden node activations are:')
    print('    Hidden node 0: ', '  %.4f' % hiddenNode0, '  Hidden node 1: ', '  %.4f' % hiddenNode1)
    print(' ')
    print('  The output node activations are:')
    print('    Output node 0: ', '  %.3f' % outputNode0, '   Output node 1: ', '  %.3f' % outputNode1)
    print(' ' )
    print('  The transfer function derivatives at the hidden nodes are: ')
    print('    Deriv-F(0): ', '     %.3f' % transFuncDeriv0, '   Deriv-F(1): ', '     %.3f' % transFuncDeriv1)

    print(' ')
    print('The computed values for the deltas are: ')
    print('                eta  *  error  *   trFncDeriv *   input    * SumTerm for given H')
    print('  deltaWWt00 = ',' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % transFuncDerivHidden0, '  * %.4f' % inputNode0, '  * %.4f' % sumTermH0)
    print('  deltaWWt01 = ',' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % transFuncDerivHidden1, '  * %.4f' % inputNode0, '  * %.4f' % sumTermH1)                      
    print('  deltaWWt10 = ',' %.2f' % eta, '* %.4f' % error0, ' * %.4f' % transFuncDerivHidden0, '  * %.4f' % inputNode1, '  * %.4f' % sumTermH0)
    print('  deltaWWt11 = ',' %.2f' % eta, '* %.4f' % error1, ' * %.4f' % transFuncDerivHidden1, '  * %.4f' % inputNode1, '  * %.4f' % sumTermH1)
    print(' ')
    print('Values for the input-to-hidden connection weights:')
    print('           Old:     New:      eta*Delta:')
    print('[0,0]:   %.4f' % wWeightArray[0,0], '  %.4f' % newWWeightArray[0,0], '  %.4f' % deltaWWtArray[0,0])
    print('[0,1]:   %.4f' % wWeightArray[1,0], '  %.4f' % newWWeightArray[1,0], '  %.4f' % deltaWWtArray[1,0])
    print('[1,0]:   %.4f' % wWeightArray[0,1], '  %.4f' % newWWeightArray[0,1], '  %.4f' % deltaWWtArray[0,1])
    print('[1,1]:   %.4f' % wWeightArray[1,1], '  %.4f' % newWWeightArray[1,1], '  %.4f' % deltaWWtArray[1,1])

#%% [markdown]
#### Backpropagate weight changes
# The comments from Dr. Maren below apply to the next four functions
# <br/> 
###### "
# The first step here applies a backpropagation-based weight change to 
# the hidden-to-output wts v.
# <br/><br/> 
# Core equation for the first part of backpropagation: <br/><br/>
# d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
# <br/><br/>
# where:
# * SSE = sum of squared errors, and only the error associated with a given output node counts
# * v(h,o) is the connection weight v between the hidden node h and the output node o
# * alpha is the scaling term within the transfer function, often set to 1
# * - (this is included in transfFuncDeriv) 
# * Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
# * F = transfer function, here using the sigmoid transfer function
# * Hidden(h) = the output of hidden node h. 
# <br/><br/>
# Note that the training rate parameter is assigned in main; Greek letter "eta" 
# (looks like n) scales amount of change to connection weight
#
# Unpack the errorList and the vWeightArray
#
# We will DECREMENT the connection weight v by a small amount proportional 
# to the derivative eqn of the SSE w/r/t the weight v. 
#
#
# This means, since there is a minus sign in that derivative, that we will 
# add a small amount. <br/> 
# (Decrementing is -, applied to a (-), which yields a positive.)
#
# For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand), 
#   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X. 
#   (Meaning: exact chapter is still TBD.) 
# For the latest updates, etc., please visit: www.aliannajmaren.com
###### "

# A few other inline notes from Dr. Maren:
#
###### "
# The equation for the actual dependence of the Summed Squared Error on a 
# given hidden-to-output weight v(h,o) is:
# <br/><br/>
#   partial(SSE)/partial(v(h,o)) = -alpha*E(o)*F(o)*`[`1-F(o)]*H(h)
# <br/><br/>
# The transfer function derivative (transFuncDeriv) returned from 
# computeTransferFnctnDeriv is given as:
# <br/><br/>
#   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput)
# <br/><br/>
# Therefore, we can write the equation for the partial(SSE)/partial(v(h,o)) as
#   partial(SSE)/partial(v(h,o)) = E(o)*transFuncDeriv*H(h)
# <br/><br/>
#   The parameter alpha is included in transFuncDeriv
###### "
#
##### Backpropagate weight changes onto the hidden-to-output connection weights

#%%
# "Backpropagate weight changes onto the hidden-to-output connection weights"
def BackpropagateOutputToHidden (alpha, eta, errorList, actualAllNodesOutputList
                                , vWeightArray):
    # "Unpack the errorList and the vWeightArray"
    error0 = errorList[0]
    error1 = errorList[1]
    
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[1,0]
    vWt10 = vWeightArray[0,1]       
    vWt11 = vWeightArray[1,1]  
    
    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]  
        
    transFuncDeriv0 = computeTransferFnctnDeriv(outputNode0, alpha) 
    transFuncDeriv1 = computeTransferFnctnDeriv(outputNode1, alpha)
    transFuncDerivList = (transFuncDeriv0, transFuncDeriv1) 

    # "Note: the parameter 'alpha' in the transfer function shows up in the 
    # transfer function derivative and so is not included explicitly in 
    # these equations"

    partialSSE_w_Vwt00 = -error0*transFuncDeriv0*hiddenNode0                                                             
    partialSSE_w_Vwt01 = -error1*transFuncDeriv1*hiddenNode0
    partialSSE_w_Vwt10 = -error0*transFuncDeriv0*hiddenNode1
    partialSSE_w_Vwt11 = -error1*transFuncDeriv1*hiddenNode1                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                
    deltaVWt00 = -eta*partialSSE_w_Vwt00
    deltaVWt01 = -eta*partialSSE_w_Vwt01        
    deltaVWt10 = -eta*partialSSE_w_Vwt10
    deltaVWt11 = -eta*partialSSE_w_Vwt11 
    deltaVWtArray = np.array([[deltaVWt00, deltaVWt10],[deltaVWt01, deltaVWt11]])

    vWt00 = vWt00+deltaVWt00
    vWt01 = vWt01+deltaVWt01
    vWt10 = vWt10+deltaVWt10
    vWt11 = vWt11+deltaVWt11 
    
    newVWeightArray = np.array([[vWt00, vWt10], [vWt01, vWt11]])

    PrintAndTraceBackpropagateOutputToHidden (alpha, eta, errorList 
                                              , actualAllNodesOutputList
                                              , transFuncDerivList, deltaVWtArray
                                              , vWeightArray, newVWeightArray)    
                                                                                                                                            
    return (newVWeightArray)  



#%% [markdown]
##### "Backpropagate weight changes onto the bias-to-output connection weights"
#
# This is handled similarly to the function above. 

#%%
def BackpropagateBiasOutputWeights (alpha, eta, errorList
                                    , actualAllNodesOutputList
                                    , biasOutputWeightArray):
    # "Unpack the errorList" 
    error0 = errorList[0]
    error1 = errorList[1]

    # "Unpack the biasOutputWeightArray, we will only be 
    # modifying the biasOutput terms   "
    biasOutputWt0 = biasOutputWeightArray[0]
    biasOutputWt1 = biasOutputWeightArray[1]

    # "Unpack the outputNodes  "
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3] 

    # "Compute the transfer function derivatives as a function of the 
    # output nodes. *Note: As this is being done after the call to the 
    # backpropagation on the hidden-to-output weights, the transfer 
    # function derivative computed there could have been used here; 
    # the calculations are being redone here only to maintain module independence"              
    transFuncDeriv0 = computeTransferFnctnDeriv(outputNode0, alpha) 
    transFuncDeriv1 = computeTransferFnctnDeriv(outputNode1, alpha) 

    # "This is actually an unnecessary step; 
    # we're not passing the list back. "
    transFuncDerivList = (transFuncDeriv0, transFuncDeriv1) 

    partialSSE_w_BiasOutput0 = -error0*transFuncDeriv0
    partialSSE_w_BiasOutput1 = -error1*transFuncDeriv1    
                                                                                                                                                                                                                                                                
    deltaBiasOutput0 = -eta*partialSSE_w_BiasOutput0
    deltaBiasOutput1 = -eta*partialSSE_w_BiasOutput1

    biasOutputWt0 = biasOutputWt0+deltaBiasOutput0
    biasOutputWt1 = biasOutputWt1+deltaBiasOutput1 

    # "Note that only the bias weights for the output nodes have been changed."
    newBiasOutputWeightArray = np.array([biasOutputWt0, biasOutputWt1])

    # The print trace function is missing because it hasn't been written
    return (newBiasOutputWeightArray)

#%% [markdown]
##### "Backpropagate weight changes onto the input-to-hidden connection weights"
#
# This is handled similarly to the function above. 

#%%
# "Backpropagate weight changes onto the input-to-hidden connection weights"
def BackpropagateHiddenToInput (alpha, eta, errorList, actualAllNodesOutputList
                                , inputDataList
                                , vWeightArray, wWeightArray
                                , biasHiddenWeightArray, biasOutputWeightArray):
    # Unpack the errorList and the vWeightArray
    error0 = errorList[0]
    error1 = errorList[1]
    
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[1,0]
    vWt10 = vWeightArray[0,1]       
    vWt11 = vWeightArray[1,1]  
    
    wWt00 = wWeightArray[0,0]
    wWt01 = wWeightArray[1,0]
    wWt10 = wWeightArray[0,1]       
    wWt11 = wWeightArray[1,1] 
    
    inputNode0 = inputDataList[0] 
    inputNode1 = inputDataList[1]         
    hiddenNode0 = actualAllNodesOutputList[0]
    hiddenNode1 = actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]

    # "For the second step in backpropagation (computing deltas on the 
    # input-to-hidden weights), we need the transfer function derivative 
    # applied to the output at the hidden node"
    transFuncDerivHidden0 = computeTransferFnctnDeriv(hiddenNode0, alpha) 
    transFuncDerivHidden1 = computeTransferFnctnDeriv(hiddenNode1, alpha)
    transFuncDerivHiddenList = (transFuncDerivHidden0, transFuncDerivHidden1)

    # "We also need the transfer function derivative applied to 
    # the output at the output node"
    transFuncDerivOutput0 = computeTransferFnctnDeriv(outputNode0, alpha) 
    transFuncDerivOutput1 = computeTransferFnctnDeriv(outputNode1, alpha)
    transFuncDerivOutputList = (transFuncDerivOutput0, transFuncDerivOutput1) 
    
    errorTimesTransDOutput0 = error0*transFuncDerivOutput0
    errorTimesTransDOutput1 = error1*transFuncDerivOutput1

    # "Note: the parameter 'alpha' in the transfer function shows up in 
    # the transfer function derivative and so is not included explicitly
    # in these equations"
    partialSSE_w_Wwt00 = (-transFuncDerivHidden0*inputNode0
                            *(vWt00*errorTimesTransDOutput0 
                            + vWt01*errorTimesTransDOutput1))                                                             
    partialSSE_w_Wwt01 = (-transFuncDerivHidden1*inputNode0
                            *(vWt10*errorTimesTransDOutput0 
                            + vWt11*errorTimesTransDOutput1))
    partialSSE_w_Wwt10 = (-transFuncDerivHidden0*inputNode1
                            *(vWt00*errorTimesTransDOutput0 
                            + vWt01*errorTimesTransDOutput1))
    partialSSE_w_Wwt11 = (-transFuncDerivHidden1*inputNode1
                            *(vWt10*errorTimesTransDOutput0 
                            + vWt11*errorTimesTransDOutput1))

    deltaWWt00 = -eta*partialSSE_w_Wwt00
    deltaWWt01 = -eta*partialSSE_w_Wwt01        
    deltaWWt10 = -eta*partialSSE_w_Wwt10
    deltaWWt11 = -eta*partialSSE_w_Wwt11 
    deltaWWtArray = np.array([[deltaWWt00, deltaWWt10]
                            ,[deltaWWt01, deltaWWt11]]) 

    wWt00 = wWt00+deltaWWt00
    wWt01 = wWt01+deltaWWt01
    wWt10 = wWt10+deltaWWt10
    wWt11 = wWt11+deltaWWt11 
    
    newWWeightArray = np.array([[wWt00, wWt10], [wWt01, wWt11]])

    return (newWWeightArray)

#%%
##### "Backpropagate weight changes onto the bias-to-hidden connection weights"

#%%
# "Backpropagate weight changes onto the bias-to-hidden connection weights"
def BackpropagateBiasHiddenWeights (alpha, eta, errorList
                                    , actualAllNodesOutputList, vWeightArray
                                    , biasHiddenWeightArray
                                    , biasOutputWeightArray):
   # "Unpack the errorList and vWeightArray"
    error0 = errorList[0]
    error1 = errorList[1]
    
    vWt00 = vWeightArray[0,0]
    vWt01 = vWeightArray[1,0]
    vWt10 = vWeightArray[0,1]       
    vWt11 = vWeightArray[1,1] 

    # "Unpack the biasWeightArray, we will only be modifying the biasOutput 
    # terms, but need to have all the bias weights for when we redefine 
    # the biasWeightArray"
    biasHiddenWt0 = biasHiddenWeightArray[0]
    biasHiddenWt1 = biasHiddenWeightArray[1]    
    biasOutputWt0 = biasOutputWeightArray[0]
    biasOutputWt1 = biasOutputWeightArray[1]

    # "Unpack the outputNodes"  
    hiddenNode0= actualAllNodesOutputList[0]
    hiddenNode1= actualAllNodesOutputList[1]
    outputNode0 = actualAllNodesOutputList[2]    
    outputNode1 = actualAllNodesOutputList[3]

    # "Compute the transfer function derivatives as a function of the 
    # output nodes. *Note: As this is being done after the call to the 
    # backpropagation on the hidden-to-output weights, the transfer 
    # function derivative computed there could have been used here; 
    # the calculations are being redone here only to maintain module independence"
    transFuncDerivOutput0 = computeTransferFnctnDeriv(outputNode0, alpha) 
    transFuncDerivOutput1 = computeTransferFnctnDeriv(outputNode1, alpha)
    transFuncDerivHidden0 = computeTransferFnctnDeriv(hiddenNode0, alpha) 
    transFuncDerivHidden1 = computeTransferFnctnDeriv(hiddenNode1, alpha) 

    # "This list will be used only if we call a print-and-trace debug function."" 
    transFuncDerivOutputList = (transFuncDerivOutput0, transFuncDerivOutput1)  

    errorTimesTransDOutput0 = error0*transFuncDerivOutput0
    errorTimesTransDOutput1 = error1*transFuncDerivOutput1
    
    partialSSE_w_BiasHidden0 = -transFuncDerivHidden0*(errorTimesTransDOutput0*vWt00 + 
    errorTimesTransDOutput1*vWt01)
    partialSSE_w_BiasHidden1 = -transFuncDerivHidden1*(errorTimesTransDOutput0*vWt10 + 
    errorTimesTransDOutput1*vWt11)  
                                                                                                                                                                                                                                                                
    deltaBiasHidden0 = -eta*partialSSE_w_BiasHidden0
    deltaBiasHidden1 = -eta*partialSSE_w_BiasHidden1

    biasHiddenWt0 = biasHiddenWt0+deltaBiasHidden0
    biasHiddenWt1 = biasHiddenWt1+deltaBiasHidden1 

    newBiasHiddenWeightArray = np.array([biasHiddenWt0, biasHiddenWt1])

    return (newBiasHiddenWeightArray) 

#%% [markdown]
### The MAIN Module
#
###### "
# The MAIN module comprising of calls to:
#   *  Welcome
#   * Obtain neural network size specifications for a three-layer network consisting of:
#   *    - Input layer
#   *    - Hidden layer
#   *    - Output layer (all the sizes are currently hard-coded to two nodes per layer right now)
#   * Initialize connection weight values
#   *    - w: Input-to-Hidden nodes
#   *    - v: Hidden-to-Output nodes
#   * Determine the initial Sum Squared Error (SSE) for each training pair, and also the total SSE
###### "

#%%
# Creating the Main module
def main():
    # "Obtain unit array size in terms of array_length (M) and layers (N)""
    welcome() # Calls our first procedure to print a message
    # Hardcode parameter definitions (replace later with user input)
    alpha = 1.0 # "parameter governing steepness of sigmoid transfer function"
    summedInput = 1
    maxNumIterations = 10    # "temporarily set to 10 for testing"
    eta = 0.5                # "training rate"

    # For this first MLP network, we're hardcoding the layers
    # We'll have an Input, Output, and one Hidden layer

    # "This defines the variable arraySizeList, which is a list."
    # "It is initially an empty list. Its purpose is to store the" 
    # size of the array."

    arraySizeList = list() # "empty list *See Note 1 in section below "
    arraySizeList = obtainNeuralNetworkSizeSpecs ()

    # Unpack the list; ascribe the various elements of the list to the 
    # sizes of different network layers    
    inputArrayLength = arraySizeList[0]
    hiddenArrayLength = arraySizeList [1]
    outputArrayLength = arraySizeList [2]

    # See Note 2 below

    # Initialize the training list *See Note 3 below
    trainingDataList = (0,0,0,0,0)

    # Initialize the weight arrays for two sets of weights

    # The wWeightArray is for Input-to-Hidden
    # The vWeightArray is for Hidden-to-Output
    wWeightArraySizeList = (inputArrayLength, hiddenArrayLength)
    vWeightArraySizeList = (hiddenArrayLength, outputArrayLength)
    biasHiddenWeightArraySize = hiddenArrayLength
    biasOutputWeightArraySize = outputArrayLength  

    # The node-to-node connection weights are stored in a 2-D array    

    # Debug parameter for examining results within initializeWeightArray is currently set to False
    debugCallInitializeOff = True
    debugInitializeOff = True
    if not debugCallInitializeOff:
        print(' ')
        print('Calling initializeWeightArray for input-to-hidden weights')

    wWeightArray = initializeWeightArray (wWeightArraySizeList
                                          , debugInitializeOff)
    
    if not debugCallInitializeOff:
        print(' ')
        print('Calling initializeWeightArray for hidden-to-output weights')

    vWeightArray = initializeWeightArray (vWeightArraySizeList
                                          , debugInitializeOff)

    # The bias weights are stored in a 1-D array         
    biasHiddenWeightArray = initializeBiasWeightArray (biasHiddenWeightArraySize)
    biasOutputWeightArray = initializeBiasWeightArray (biasOutputWeightArraySize) 


    initialWWeightArray = wWeightArray[:]
    initialVWeightArray = vWeightArray[:]
    initialBiasHiddenWeightArray = biasHiddenWeightArray[:]   
    initialBiasOutputWeightArray = biasOutputWeightArray[:] 

    print()
    print('The initial weights for this neural network are:')
    print('       Input-to-Hidden                            Hidden-to-Output')
    print('  w(0,0) = %.4f   w(1,0) = %.4f         v(0,0) = %.4f   v(1,0) = %.4f' 
            % (initialWWeightArray[0,0]
            , initialWWeightArray[0,1]
            , initialVWeightArray[0,0]
            , initialVWeightArray[0,1]))
    print('  w(0,1) = %.4f   w(1,1) = %.4f         v(0,1) = %.4f   v(1,1) = %.4f' 
            % (initialWWeightArray[1,0]
            , initialWWeightArray[1,1]
            , initialVWeightArray[1,0]
            , initialVWeightArray[1,1]))
    print(' ')
    print('       Bias at Hidden Layer                          Bias at Output Layer')
    print('       b(hidden,0) = %.4f                           b(output,0) = %.4f' 
            % (biasHiddenWeightArray[0], biasOutputWeightArray[0] ))                  
    print('       b(hidden,1) = %.4f                           b(output,1) = %.4f' 
            % (biasHiddenWeightArray[1], biasOutputWeightArray[1] ))  
  
    epsilon = 0.2
    iteration = 0
    SSE_InitialTotal = 0.0

   # "Next step - Get an initial value for the 
   # Total Summed Squared Error (Total_SSE)"

   # "Initialize an array of SSE values" * See Note 4 below
    SSE_InitialArray = [0,0,0,0,0]
    
    # "Before starting the training run, compute the initial SSE Total 
    #   (sum across SSEs for each training data set)" 
    debugSSE_InitialComputationOff = True

    SSE_InitialArray = computeSSE_Values (alpha, SSE_InitialArray
                                          , wWeightArray, vWeightArray
                                          , biasHiddenWeightArray
                                          , biasOutputWeightArray
                                          , debugSSE_InitialComputationOff)

    # "Start the SSE_Array at the same values as the Initial SSE Array"
    SSE_Array = SSE_InitialArray[:] 
    SSE_InitialTotal = SSE_Array[4] 

    # "Set a local debug print parameter"(this step optional)
    debugSSE_InitialComputationReportOff = True    

    if not debugSSE_InitialComputationReportOff:
        print(' ')
        print('In main, SSE computations completed, Total of all SSEs = %.4f' 
                % SSE_Array[4])
        print('  For input nodes (0,0), SSE_Array[0] = %.4f' % SSE_Array[0])
        print('  For input nodes (0,1), SSE_Array[1] = %.4f' % SSE_Array[1])
        print('  For input nodes (1,0), SSE_Array[2] = %.4f' % SSE_Array[2])
        print('  For input nodes (1,1), SSE_Array[3] = %.4f' % SSE_Array[3])

# Running the model:
# "Next step - Obtain a single set of input values for the X-OR problem; 
# two integers - can be 0 or 1"

    while iteration < maxNumIterations:
         # "Randomly select one of four training sets; the inputs will be 
         # randomly assigned to 0 or 1"
        trainingDataList = obtainRandomXORTrainingValues () 
        input0 = trainingDataList[0]
        input1 = trainingDataList[1] 
        desiredOutput0 = trainingDataList[2]
        desiredOutput1 = trainingDataList[3]
        setNumber = trainingDataList[4]       
        print(' ')
        print('Randomly selecting XOR inputs for XOR, identifying desired outputs for this training pass:')
        print('          Input0 = ', input0,         '            Input1 = ', input1)   
        print(' Desired Output0 = ', desiredOutput0, '   Desired Output1 = ', desiredOutput1)    
        print(' ')

        # "Compute a single feed-forward pass"

        # Initialize the error list
        errorList = (0,0)
    
        # "Initialize the actualOutput list"
        actualAllNodesOutputList = (0,0,0,0)     

        # "Create the inputData list"      
        inputDataList = (input0, input1)         
    
        # "Compute a single feed-forward pass and obtain the Actual Outputs"
        debugComputeSingleFeedforwardPassOff = True
        actualAllNodesOutputList = ComputeSingleFeedforwardPass (alpha
                                        , inputDataList
                                        , wWeightArray
                                        , vWeightArray
                                        , biasHiddenWeightArray
                                        , biasOutputWeightArray
                                        ,debugComputeSingleFeedforwardPassOff)

        # "Assign the hidden and output values to specific different variables"
        actualHiddenOutput0 = actualAllNodesOutputList [0] 
        actualHiddenOutput1 = actualAllNodesOutputList [1] 
        actualOutput0 = actualAllNodesOutputList [2]
        actualOutput1 = actualAllNodesOutputList [3] 
    
        # "Determine the error between actual and desired outputs"

        error0 = desiredOutput0 - actualOutput0
        error1 = desiredOutput1 - actualOutput1
        errorList = (error0, error1)
    
        # "Compute the Summed Squared Error, or SSE"
        SSEInitial = error0**2 + error1**2
        
                            
        debugMainComputeForwardPassOutputsOff = True

        # "Debug print the actual outputs from the two output neurons"
        if not debugMainComputeForwardPassOutputsOff:
            print(' ')
            print('In main; have just completed a feedfoward pass with training set inputs'
                                                                , input0, input1)
            print('  The activations (actual outputs) for the two hidden neurons are:')
            print('    actualHiddenOutput0 = %.4f' % actualHiddenOutput0)
            print('    actualHiddenOutput1 = %.4f' % actualHiddenOutput1)   
            print('  The activations (actual outputs) for the two output neurons are:')
            print('    actualOutput0 = %.4f' % actualOutput0)
            print('    actualOutput1 = %.4f' % actualOutput1)
            print('  Initial SSE (before backpropagation) = %.6f' % SSEInitial)
            print('  Corresponding SSE (from initial SSE determination) = %.6f' 
                                                        % SSE_Array[setNumber])

        # "Perform first part of the backpropagation of weight changes"
        newVWeightArray = BackpropagateOutputToHidden (alpha, eta, errorList
                                                      , actualAllNodesOutputList
                                                      , vWeightArray)

        newBiasOutputWeightArray = BackpropagateBiasOutputWeights (alpha, eta
                                                    , errorList
                                                    , actualAllNodesOutputList
                                                    , biasOutputWeightArray)
        newBiasOutputWeight0 = newBiasOutputWeightArray[0]
        newBiasOutputWeight1 = newBiasOutputWeightArray[1]
        
        newWWeightArray = BackpropagateHiddenToInput (alpha, eta, errorList
                                                    , actualAllNodesOutputList
                                                    , inputDataList, vWeightArray
                                                    , wWeightArray
                                                    , biasHiddenWeightArray
                                                    , biasOutputWeightArray)

        newBiasHiddenWeightArray = BackpropagateBiasHiddenWeights (alpha, eta
                                                , errorList
                                                , actualAllNodesOutputList
                                                , vWeightArray
                                                , biasHiddenWeightArray
                                                , biasOutputWeightArray)
        newBiasHiddenWeight0 = newBiasHiddenWeightArray[0]
        newBiasHiddenWeight1 = newBiasHiddenWeightArray[1]        
        
        newBiasWeightArray = [[newBiasOutputWeight0, newBiasOutputWeight1]
                            , [newBiasHiddenWeight0, newBiasHiddenWeight1]] 

        # "Debug prints on the weight arrays"
        debugWeightArrayOff = False

        if not debugWeightArrayOff:
            print(' ')
            print('    The weights before backpropagation are:')
            print('         Input-to-Hidden                           Hidden-to-Output')
            print('    w(0,0) = %.3f   w(1,0) = %.3f         v(0,0) = %.3f   v(1,0) = %.3f' 
                        % (wWeightArray[0,0], wWeightArray[0,1]
                        , vWeightArray[0,0], vWeightArray[0,1]))
            print('    w(0,1) = %.3f   w(1,1) = %.3f         v(0,1) = %.3f   v(1,1) = %.3f' 
                        % (wWeightArray[1,0], wWeightArray[1,1]
                        , vWeightArray[1,0], vWeightArray[1,1])) 
            print(' ')
            print('    The weights after backpropagation are:')
            print('         Input-to-Hidden                           Hidden-to-Output')
            print('    w(0,0) = %.3f   w(1,0) = %.3f         v(0,0) = %.3f   v(1,0) = %.3f' 
                        % (newWWeightArray[0,0], newWWeightArray[0,1]
                        , newVWeightArray[0,0], newVWeightArray[0,1]))
            print('    w(0,1) = %.3f   w(1,1) = %.3f         v(0,1) = %.3f   v(1,1) = %.3f' 
                        % (newWWeightArray[1,0], newWWeightArray[1,1]
                        , newVWeightArray[1,0], newVWeightArray[1,1]))

        # "Assign the old hidden-to-output weight array to be the same as 
        # what was returned from the BP weight update"
        vWeightArray = newVWeightArray[:]
    
        # "Assign the old input-to-hidden weight array to be the same as 
        # what was returned from the BP weight update"
        wWeightArray = newWWeightArray[:]
    
        # "Run the computeSingleFeedforwardPass again, to compare the 
        # results after just adjusting the hidden-to-output weights"
        newAllNodesOutputList = ComputeSingleFeedforwardPass (alpha, inputDataList, wWeightArray, vWeightArray,
        biasHiddenWeightArray, biasOutputWeightArray, debugComputeSingleFeedforwardPassOff)         
        newOutput0 = newAllNodesOutputList [2]
        newOutput1 = newAllNodesOutputList [3] 

        # "Determine the new error between actual and desired outputs"
        newError0 = desiredOutput0 - newOutput0
        newError1 = desiredOutput1 - newOutput1
        newErrorList = (newError0, newError1)

        # "Compute the new Summed Squared Error, or SSE"
        SSE0 = newError0**2
        SSE1 = newError1**2
        newSSE = SSE0 + SSE1

        # "Print the Summed Squared Error"

        # "Debug print the actual outputs from the two output neurons"
        if not debugMainComputeForwardPassOutputsOff:
            print(' ')
            print('In main; have just completed a single step of backpropagation with inputs'
                                                               , input0, input1)
            print('    The new SSE (after backpropagation) = %.6f' % newSSE)
            print('    Error(0) = %.4f,   Error(1) = %.4f' %(NewError0, NewError1))
            print('    SSE(0) =   %.4f,   SSE(1) =   %.4f' % (SSE0, SSE1))
            deltaSSE = SSEInitial - newSSE
            print('  The difference in initial and the resulting SSEs is: %.4f' 
                                                                    % deltaSSE) 
            
            if deltaSSE >0:
                print(' ')
                print('   The training has resulted in improving the total SSEs')

       # "Assign the SSE to the SSE for the appropriate training set"
        SSE_Array[setNumber] = newSSE

        # "Obtain the previous SSE Total from the SSE array"
        previousSSE_Total = SSE_Array[4]
        print(' ' )
        print('  The previous SSE Total was %.4f' % previousSSE_Total)

        # "Compute the new sum of SSEs (across all the different training sets)
        #   ... this will be different because we've changed one of the SSE's"
        newSSE_Total = SSE_Array[0] + SSE_Array[1] +SSE_Array[2] + SSE_Array[3]

        print('  The new SSE Total was %.4f' % newSSE_Total)
        print('    For node 0: Desired Output = ',desiredOutput0,  ' New Output = %.4f' 
                                                                    % newOutput0) 
        print('    For node 1: Desired Output = ',desiredOutput1,  ' New Output = %.4f' 
                                                                    % newOutput1)  
        print('    Error(0) = %.4f,   Error(1) = %.4f' %(newError0, newError1))
        print('    SSE0(0) =   %.4f,   SSE(1) =   %.4f' % (SSE0, SSE1) )

        # "Assign the new SSE to the final place in the SSE array"
        SSE_Array[4] = newSSE_Total
        deltaSSE = previousSSE_Total - newSSE_Total

        print('  Delta in the SSEs is %.4f' % deltaSSE)
        if deltaSSE > 0:
            print('SSE improvement')
        else: print('NO improvement')

       # "Assign the new errors to the error list"             
        errorList = newErrorList[:]
        
        
        print(' ')
        print('Iteration number ', iteration)
        iteration = iteration + 1

        if newSSE_Total < epsilon:

            
            break
    print('Out of while loop')

    debugEndingSSEComparisonOff = False

    if not debugEndingSSEComparisonOff:
        SSE_Array[4] = newSSE_Total
        deltaSSE = previousSSE_Total - newSSE_Total
        print('  Initial Total SSE = %.4f'  % SSE_InitialTotal)
        print('  Final Total SSE = %.4f'  % newSSE_Total)

        finalDeltaSSE = SSE_InitialTotal - newSSE_Total

        print('  Delta in the SSEs is %.4f' % finalDeltaSSE )
        if finalDeltaSSE > 0:
            print('SSE total improvement')
        else: print('NO improvement in total SSE')

# Conclude specification of the MAIN procedure
if __name__ == "__main__": main()  




#%% [markdown]
# Note 1: "# Notice that I'm using the same variable name, 'arraySizeList' 
# both here in main and in the called procedure, 'obtainNeuralNetworkSizeSpecs.' 
# I don't have to use the same name; the procedure returns a list and I'm 
# assigning it HERE to the list named arraySizeList in THIS 'main' 
# procedure. I could use different names. I'm keeping the same name so 
# that it is easier for us to connect what happens in the called procedure
#    'obtainNeuralNetworkSizeSpecs' with this procedure, 'main.' "
# <br/><br/>
# Note 2: # "I have all sorts of debug statements left in this, so you can 
# trace the code moving into and out of various procedures and functions."
# Try these:    
# <br/><br/>            
#    print('Flow-of-control trace: Back in main')<br/>
#    print('I = number of nodes in input layer is', inputArrayLength)<br/>
#    print('H = number of nodes in hidden layer is', hiddenArrayLength)<br/>         
#    print('O = number of nodes in output layer is', outputArrayLength)<br/>
#  <br/><br/>
# Note 3: "The training list has, in order, the two input nodes, the two 
# output nodes (this is a two-output version of the X-OR problem), and 
# the data set number (0..3), meaning that each data set is numbered.<br/> 
# This helps in going through the entire data set once the initial weights 
# are established to get a total sum (across all data sets) of the 
# Summed Squared Error, or SSE."
# <br/<br/>
# Note 4: "The first four SSE values are the SSE's for specific 
# input/output pairs-- the fifth is the sum of all the SSE's."
# <br/><br/>
# Note 5: 

#%% [markdown]
##### Additional print options for MAIN module
# If desired, all or some of the below code can be used after the final
# print statement in the Main procedure to give stats on weights and SSEs
# before and during training.
# <br/><br/> 
# I left them out above to increase readability. They have not been 
# edited for line length or readability in any other way.



#%%
#    print ' '
#    print 'The initial weights for this neural network are:'
#    print '     Input-to-Hidden                       Hidden-to-Output'
#    print 'w(0,0) = %.3f   w(0,1) = %.3f         v(0,0) = %.3f   v(0,1) = %.3f' % (initialWWeightArray[0,0], 
#    initialWWeightArray[0,1], initialVWeightArray[0,0], initialVWeightArray[0,1])
#    print 'w(1,0) = %.3f   w(1,1) = %.3f         v(1,0) = %.3f   v(1,1) = %.3f' % (initialWWeightArray[1,0], 
#    initialWWeightArray[1,1], initialVWeightArray[1,0], initialVWeightArray[1,1])        

                                                                                    
#    print ' '
#    print 'The final weights for this neural network are:'
#    print '     Input-to-Hidden                       Hidden-to-Output'
#    print 'w(0,0) = %.3f   w(0,1) = %.3f         v(0,0) = %.3f   v(0,1) = %.3f' % (wWeightArray[0,0], 
#    wWeightArray[0,1], vWeightArray[0,0], vWeightArray[0,1])
#    print 'w(1,0) = %.3f   w(1,1) = %.3f         v(1,0) = %.3f   v(1,1) = %.3f' % (wWeightArray[1,0], 
#    wWeightArray[1,1], vWeightArray[1,0], vWeightArray[1,1])        
                                                                                    
   
# Print the SSE's at the beginning of training
#    print ' '
#    print 'The SSE values at the beginning of training were: '
#    print '  SSE_Initial[0] = %.4f' % SSE_InitialArray[0]
#    print '  SSE_Initial[1] = %.4f' % SSE_InitialArray[1]
#    print '  SSE_Initial[2] = %.4f' % SSE_InitialArray[2]
#    print '  SSE_Initial[3] = %.4f' % SSE_InitialArray[3]    
#    print ' '
#    print 'The total of the SSE values at the beginning of training is %.4f' % SSE_InitialTotal 


# Print the SSE's at the end of training
#    print ' '
#    print 'The SSE values at the end of training were: '
#    print '  SSE[0] = %.4f' % SSE_Array[0]
#    print '  SSE[1] = %.4f' % SSE_Array[1]
#    print '  SSE[2] = %.4f' % SSE_Array[2]
#    print '  SSE[3] = %.4f' % SSE_Array[3]    
#    print ' '
#    print 'The total of the SSE values at the end of training is %.4f' % SSE_Array[4]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
# Print comparison of previous and new outputs             
#    print ' ' 
#    print 'Values for the new outputs compared with previous, given only a partial backpropagation training:'
#    print '     Old:', '   ', 'New:', '   ', 'nu*Delta:'
#    print 'Output 0:  Desired = ', desiredOutput0, 'Old actual =  %.4f' % actualOutput0, 'Newactual  %.4f' % newOutput0
#    print 'Output 1:  Desired = ', desiredOutput1, 'Old actual =  %.4f' % actualOutput1, 'Newactual  %.4f' % newOutput1