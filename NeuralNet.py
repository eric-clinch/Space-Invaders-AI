import random
import math
from functools import reduce
import collections
import time
import numpy as np
import sys
import GetTrainingData
import pickle

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_derivative(WeightedInput):
	return sigmoid(WeightedInput) * (1 - sigmoid(WeightedInput))

class NeuralNet:
	#NeuralNet(i, o, h) creates a neural net that takes a list of length i as input,
	#outputs a list of length o, and has len(h) hidden layers. The ith hidden layer has h[i] neurons.
	#if weights = None, then random weights will be assigned to each neuron
	#Otherwise, weights should be a list of floating point values with length equal to the number of neurons in the layer,
	#which is numInputs + numOutputs + sum(hiddenLayers[i])
	#the weights are inserted into the neuron layer from the left hand side of the input layer upwards
	def __init__(self, numInputs, numOutputs, hiddenLayers, weights = None, biases = None):
		self.numInputs = numInputs
		self.numOutputs = numOutputs
		#create a list of the number of neurons at each layer
		self.numLayerNeurons = hiddenLayers + [numOutputs]
		#create a list of the number of inputs that each layer takes
		self.numLayerInputs = [numInputs] + hiddenLayers
		self.numLayers = len(self.numLayerInputs) + 1
		if(weights is None):
			assert(biases is None)
			self.biases = [np.random.randn(y) for y in self.numLayerNeurons]
			self.weights = [np.random.randn(y, x) for x, y in zip(self.numLayerInputs, self.numLayerNeurons)]
		else:
			assert(not biases is None)
			assert(len(biases) == len(weights) and len(weights) == len(self.numLayerNeurons))
			for i in range(len(weights)):
				assert(len(weights[i]) == len(biases[i]) and len(weights[i]) == self.numLayerNeurons[i])
			self.weights = weights
			self.biases = biases

	#processes the given inputs and returns a list of numbers of length self.numOutputs
	#inputs should be a list of numbers whose length is equal to self.numInputs
	def activate(self, inputs):
		#assert(len(inputs) == self.numInputs)
		a = np.array(inputs)
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a) + b)
		return a

	def getWeights(self):
		return self.weights

	def getBiases(self):
		return self.biases

	def save(self, path):
		np.savez(path, [self.weights, self.biases])

	@staticmethod
	def cost_delta(a, y):
		return a - y

	@staticmethod
	def make2D(x):
		x = x.tolist()
		x = list(map((lambda y: [y]), x))
		return np.array(x)

	#gets the normalized weights of the neural net
	def getLength(self):
		length = 0
		for w in self.weights:
			length += np.linalg.norm(w)
		return length

	def trainWithCutoffRounds(self, training_data, cutOff, cutOffRounds, mini_batch_size, eta, test_data, lmbda = 0.0):
		n_test = len(test_data)
		n = len(training_data)
		epoch = 0
		cutOffRound = 0
		unimprovedEpochs = 0
		minCost = None
		while (cutOffRound < cutOffRounds):
			epoch += 1
			print("epoch %d" % epoch)
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k 
								in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta, lmbda)
			cost = self.cost(test_data)
			correctEvaluations = self.evaluate(test_data)
			print("Epoch %d: %d / %d" % (epoch, correctEvaluations, n_test))
			print("Cost: %f" % cost)
			if(correctEvaluations == n_test): break #classified all correctly, no more training necessary
			if(minCost == None or cost < minCost):
				minCost = cost
				unimprovedEpochs = 0
			else:
				unimprovedEpochs += 1
				if(unimprovedEpochs >= cutOff):
					print("completed round %d\n" % (cutOffRound + 1))
					eta = eta/2
					cutOffRound += 1
					unimprovedEpochs = 0
					minCost = None

	def train(self, training_data, epochs, mini_batch_size, eta, lmbda = 0.0, test_data = None):
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in range(epochs):
			print("epoch %d" % j)
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k 
								in range(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch, eta, lmbda)
			if test_data:
				print("Epoch %d: %d / %d" % (j, self.evaluate(test_data), n_test))
				print("Cost: %f" % self.cost(test_data))

	def update_mini_batch(self, mini_batch, eta, lmbda):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
			nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [(1-eta*(lmbda))*w-(eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

	def backpropagate(self, x, y):
		weightsGradient = [np.zeros(w.shape) for w in self.weights]
		biasesGradient = [np.zeros(b.shape) for b in self.biases]
		#feedforward stage
		activation = x
		activations = [x]
		WeightedInputs = []
		for b, w in zip(self.biases, self.weights):
			WeightedInput = np.dot(w, activation) + b
			WeightedInputs.append(WeightedInput)
			activation = sigmoid(WeightedInput)
			activations.append(activation)
		delta = self.cost_delta(activations[-1], y)
		weightsGradient[-1] = np.dot(self.make2D(delta), self.make2D(activations[-2]).transpose())
		biasesGradient[-1] = delta
		for l in range(2, self.numLayers):
			WeightedInput = WeightedInputs[-l]
			sp = sigmoid_derivative(WeightedInput)
			delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
			biasesGradient[-l] = delta
			weightsGradient[-l] = np.dot(self.make2D(delta), self.make2D(activations[-l-1]).transpose())
		return (biasesGradient, weightsGradient)

	def getMove(self, x):
		maxValue = max(x)
		index = x.index(maxValue)
		res = [0 for _ in x]
		res[index] = 1
		return res

	def evaluate(self, test_data):
		correct = 0
		for x, y in test_data:
			res = self.activate(x.tolist()).tolist()
			res = self.getMove(res)
			diff = res - y
			mag = np.linalg.norm(diff)
			if mag < .0001: correct += 1
		return correct

	def cost(self, test_data):
		cost = 0
		for x, y in test_data:
			a = self.activate(x.tolist())
			c = np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
			cost += c
		return cost / len(test_data)

def getNN(path):
	loadedFile = np.load(path)
	NNInfo = loadedFile[loadedFile.files[0]]
	weights = NNInfo[0]
	biases = NNInfo[1]
	numInputs = len(weights[0][0])
	hiddenLayers = list(map(len, weights[:-1]))
	numOutputs = len(weights[-1])
	NN = NeuralNet(numInputs, numOutputs, hiddenLayers,
							 weights = weights, biases = biases)
	return NN

if __name__ == "__main__":
	# trainingDataPath = sys.argv[1]
	# outputPath = sys.argv[2]

	trainingDataPath = "data/trainingData0.p"
	outputPath = "NNs/AlphaBetaNN.npz"

	NN = NeuralNet(768, 4, [600])
	data = pickle.load(open(trainingDataPath, 'rb'))
	random.shuffle(data)
	testDataSize = len(data) // 5
	trainingData = data[:-testDataSize]
	testingData = data[-testDataSize:]

	NN.trainWithCutoffRounds(trainingData, 5, 6, 1, .02, test_data = testingData, lmbda = .0001)
	NN.save(outputPath)