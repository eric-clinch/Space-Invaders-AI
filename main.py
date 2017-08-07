import ScreenReader
import getScore
import controls
import NeuralNet
import NeuralNetParser
import geneticAlgorithm 
import copy
import inputEditing

screenRows = ScreenReader.screenRows
screenCols = ScreenReader.screenCols

def play(NN):
	controls.setup()
	start = True
	initialInput = None
	while(start == True or ScreenReader.averageBlue(pixels) > .1):
		pixels = ScreenReader.getPixels()
		NNinputs = ScreenReader.getPixelInputs(pixels)
		NNinputs = inputEditing.addShipFocusing(NNinputs)
		NNoutputs = NN.process(NNinputs)
		controls.doCommands(NNoutputs)
		start = False
	score = getScore.getScore()
	return score

if __name__ == "__main__":
	NN = NeuralNet.getNN("NNs/Data0NN.npz")
	play(NN)