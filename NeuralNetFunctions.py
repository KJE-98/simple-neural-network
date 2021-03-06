import random
import ast

# Shallow copy
def copy(list):
    result = []
    for element in list:
        result.append(element)
    return result

def activation_ReLU(x):
    if x<0:
        return 0
    if x<1:
        return x
    return 1

# creates a new node in the form of a dictionary
def createNode(inputWeights, activation="ReLU"):
    return {
        'inputs':copy(inputWeights),
        'activation':activation
    }

def activate(prevLayer, node):
    foo = 0
    for i in range(len(node['inputs'])):
        foo += node['inputs'][i]*prevLayer[i]
    result = activation_ReLU(foo)
    return result

# copies network but gives random weights
def createRandomNet(network):
    newNetwork = []
    for i in range(len(network)):
        layer = network[i]
        newNetwork.append([])
        newLayer = newNetwork[-1]
        for j in range(len(layer)):
            currentNode = layer[j]
            newNode = createNode([])
            newLayer.append(newNode)
            for k in range(len(currentNode["inputs"])):
                newInput = random.random()*2-1
                newNode['inputs'].append(newInput)
    return newNetwork

# copies a network and gives all weights value 0
def copyNetStructure(network):
    newNetwork = []
    for i in range(len(network)):
        layer = network[i]
        newNetwork.append([])
        newLayer = newNetwork[-1]
        for j in range(len(layer)):
            currentNode = layer[j]
            newNode = createNode([])
            newLayer.append(newNode)
            for k in range(len(currentNode["inputs"])):
                newInput = 0
                newNode['inputs'].append(newInput)
    return newNetwork

# Adds a node to network
# number_of_inputs only must be specified if adding layer 0 node
def addNode(network, level, defaultWeightForward=.5, defaultWeightBackward = .5, number_of_inputs = 0, activation = "ReLU"):
    node = createNode([],activation)
    network[level].append(node)
    if level>0:
        for aNode in network[level-1]:
            node['inputs'].append(defaultWeightBackward)
    else:
        for i in range(number_of_inputs):
            node['inputs'].append(defaultWeightBackward)
    if level<len(network)-1:
        for aNode in network[level+1]:
            aNode['inputs'].append(defaultWeight)

# Forward propogation
def propogate(network, inputs):
    layer_results = []
    layer_results.append(copy(inputs))
    for layer in network:
        prevLayer = layer_results[-1]
        newLayer = []
        for aNode in layer:
            activation = activate(prevLayer, aNode)
            newLayer.append(activation)
        layer_results.append(newLayer)
    return layer_results

# Creates an array of numbers (rather than nodes) having the same structure as the network
def valueArray(network):
    layer_results = []
    for layer in network:
        newLayer = []
        for aNode in layer:
            newLayer.append(0)
        layer_results.append(newLayer)
    return layer_results

# Modifies network_one and returns it
def combine(network_one, network_two, weight_one, weight_two):
    for i in range(len(network_one)):
        layer_one = network_one[i]
        layer_two = network_two[i]
        for j in range(len(layer_one)):
            currentNode_one = layer_one[j]
            currentNode_two = layer_two[j]
            for k in range(len(currentNode_one["inputs"])):
                currentNode_one["inputs"][k] = currentNode_one["inputs"][k] * weight_one + currentNode_two["inputs"][k] * weight_two
    return network_one

# BackPropogation
def train(data, network, learning_constant):
    data_length = len(data)
    num_layers = len(network)
    delta_weights_final = copyNetStructure(network)
    for datum in data:
        # BackPropogate
        # The structure keeping track of d_error/d_weight values
        delta_weights = copyNetStructure(network)
        # The structure keeping track of the d_error/d_nodeValue values
        delta_nodes = valueArray(network)
        forwardFeed_results = propogate(network, datum['input'])

        guess = forwardFeed_results[-1][0]
        # Change in SquareError over change in final value
        output_derivative = guess-datum['value']

        for reversed_index,layer in enumerate(reversed(delta_weights)):
            index = num_layers-reversed_index-1
            if reversed_index == 0:
                delta_nodes[index][0] = output_derivative
            for node_index,node in enumerate(layer):
                for input_index,input in enumerate(node['inputs']):
                    # We want d_error/d_weight = d_error/d_node * d_node/d_weight
                    # This is equal to d_error/d_node * value_of_prev_node (0 if the node value not in (0,1))
                    if forwardFeed_results[index+1][node_index]>=0 and forwardFeed_results[index+1][node_index]<=1:
                        node['inputs'][input_index] = delta_nodes[index][node_index] * forwardFeed_results[index][input_index]
                        # Also
                        if index > 0:
                            delta_nodes[index-1][input_index] += network[index][node_index]['inputs'][input_index]*delta_nodes[index][node_index]
        weight_two = 1/data_length
        weight_one = 1
        combine(delta_weights_final, delta_weights, weight_one, weight_two)
    combine(network, delta_weights_final, 1, (-1)*learning_constant)
    return network

# Serializes the network and stores it in file named "NeuralNetStorage.txt"
def writeNetToFile(network):
    network_str = repr(network)
    with open("NeuralNetStorage.txt", 'w') as file:
        file.write(network_str)
    return

# Reads from "NeuralNetStorage.txt" and returns a network
def readNetFromFile():
    result = "Error"
    with open("NeuralNetStorage.txt", 'r') as file:
        result = file.read()
    return ast.literal_eval(result)


# How it works

# First initialize a 2-d array of the number of layers you desire

# This network will have 3 Hidden layers
myNetwork = [
[],
[],
[],
[],
]

# Use addNode to create the network
for i in range(5):
    addNode(myNetwork, 0, number_of_inputs = 2)
for i in range(5):
    addNode(myNetwork, 1)
for i in range(5):
    addNode(myNetwork, 2)

addNode(myNetwork, 3)

# Train by batching the data and using the train function
for j in range(100):
    data = []

    for i in range(100):
        x = 2*random.random()/3
        y = x**2-x**3
        data.append({'input':[1,x], 'value':y})

        train(data, myNetwork, 1/200)
    print("Training batch " + str(j))

# Test to see if the Neural Net worked
# Obviously this is not a substitute for a test batch
print("Guess for x = 0")
print(propogate(myNetwork, [1,0])[-1])
print("actual answer:")
print(0**2-0**3)
print()
print("Guess for x = .2")
print(propogate(myNetwork, [1,.2])[-1])
print("actual answer:")
print(.2**2-.2**3)
print()
print("Guess for x = .55")
print(propogate(myNetwork, [1,.55])[-1])
print("actual answer:")
print(.55**2-.55**3)

# Store these weights
writeNetToFile(myNetwork)
print("Net saved into file \'NeuralNetStorage.txt\'")
