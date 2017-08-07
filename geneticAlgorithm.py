#a genetic algorithm for evolving neural nets

import os
import random
import copy
import NeuralNet
from functools import reduce

#an individual neural net in the population
class individual:
	def __init__(self, neuralNet):
		self.NN = neuralNet
		self.fitness = None

	#accepts a function that takes a neural net and returns a fitness value
	#sets the fitness value for the individual
	def setFitness(self, fitnessFunction):
		self.fitness = fitnessFunction(self.NN)

	#returns a child individual created by "breeding" this individual with the given other individual
	def reproduce(self, other, crossRate, mutRate):
		weights1, weights2 = self.NN.getFlattenedValues(), other.NN.getFlattenedValues()
		#assert(len(weights1) == len(weights2))
		childWeights = copy.copy(weights1)
		#crossover stage
		if(random.random() < crossRate):
			pos = random.randint(0, len(childWeights) - 1)
			childWeights[pos:] = weights2[pos:]

		#mutation stage
		for i in range(len(childWeights)):
			if(random.random() < mutRate):
				weight = childWeights[i]
				changePercent = (2*random.random()) - 1
				weight = weight * (1 + changePercent) #mutate
				childWeights[i] = weight

		childNumInputs = self.NN.numInputs
		childNumOutputs = self.NN.numOutputs
		childNumHiddenLayerNeurons = self.NN.numLayerNeurons[:-1]
		childNN = NeuralNet.NeuralNet(childNumInputs, childNumOutputs, 
							childNumHiddenLayerNeurons, childWeights)
		return individual(childNN)

class population:
	#constants used for reproducing individuals
	crossRate = .3
	mutRate = .001

	#takes a list of neural nets, all with the same structure
	def __init__(self, NNs, generation = 0, storeInitialGeneration = False):
		self.generation = generation
		self.populationSize = len(NNs)
		self.individuals = []
		self.totalFitness = 0
		for NN in NNs:
			self.individuals.append(individual(NN))
		if(generation == 0 and storeInitialGeneration): self.storeGenerationWeights()

	#accepts a function that takes a neural net and returns a fitness value.
	#Sets the fitness values for each individual in the population
	def setFitnesses(self, fitnessFunction):
		print("getting fitnesses")
		self.totalFitness = 0
		for individ in self.individuals:
			individ.setFitness(fitnessFunction)
			self.totalFitness += individ.fitness

	def getMaxFitness(self):
		return reduce((lambda x, y : max(x, y.fitness)), self.individuals, 0)

	def getAverageFitness(self):
		return self.totalFitness / self.populationSize

	def storeGenerationFitnesses(self):
		path = "generations/generationFitness%d.txt" % self.generation
		F = open(path, "w")
		F.write("Generation Average Fitness: %f\n" % self.getAverageFitness())
		F.write("Generation Max Fitness: %d" % self.getMaxFitness())
		F.close()

	def storeGenerationWeights(self):
		path = "generations/generationWeights%d.txt" % self.generation
		F = open(path, "w")
		for individ in self.individuals:
			weights = individ.NN.getFlattenedValues()
			F.write(str(weights))
			F.write("\n\n")
		F.close()

	#randomly selects an individual from the population. An individuals chance
	#of being selected is proportional to its fitness
	def selectIndividual(self):
		#if everything has a fitness of 0, return a random individual
		if(self.totalFitness == 0): 
			return random.choice(self.individuals)

		index = random.randint(0, self.totalFitness - 1)
		sum = 0
		for individual in self.individuals:
			sum += individual.fitness
			if(sum > index): return individual
		return self.individuals[random.randint(0, self.populationSize - 1)]

	#accepts a function that takes a neural net and returns a fitness value.
	#runs the fitness function on each individual and 
	#creates the next generation of the population accordingly
	def evolve(self, fitnessFunction):
		print("evolving generation %d" % self.generation)
		self.setFitnesses(fitnessFunction)
		self.storeGenerationFitnesses()
		print("creating new generation")
		newGen = []
		for i in range(len(self.individuals)):
			parent1 = self.selectIndividual()
			parent2 = self.selectIndividual()
			child = parent1.reproduce(parent2, self.crossRate, self.mutRate)
			newGen.append(child)
		self.individuals = newGen
		self.generation += 1
		print("storing new generation weights")
		self.storeGenerationWeights()