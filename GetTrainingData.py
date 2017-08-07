from pynput import keyboard
import ScreenReader
import time
import numpy as np

def on_press(data):
	def on_press2(key):
		if (key == keyboard.Key.left or
				key == keyboard.Key.right or
				key == keyboard.Key.space):
			screen = ScreenReader.getInputs()
			data.content += str(screen) + "\n"
			data.content += '{0}\n\n'.format(key)
	return on_press2

def on_release(key):
	if key == keyboard.Key.esc:
		# Stop listen
		return False

def testFn():
	while(True):
		print("test")

def parseStrList(Str):
	nums = Str.split(",")

	#remove the bracket from the first element
	nums[0] = nums[0][1:]
	#remove the bracket and newline character from the last element
	nums[-1] = nums[-1][:-2]
	#convert each string to an int
	for i in range(len(nums)):
		nums[i] = int(nums[i])
	return nums

def parseStrArray(Str):
	return np.array(parseStrList(Str))

def getData(path):
	F = open(path)
	lines = F.readlines()
	ins = lines[::3]
	outs = lines[1::3]
	ins = list(map(parseStrArray, ins))
	outs = list(map(parseStrArray, outs))
	return list(zip(ins, outs))

class struct: pass

if __name__ == "__main__":
	data = struct()
	data.content = ""

	with keyboard.Listener(
	        on_press=on_press(data),
	        on_release=on_release) as listener:
	    listener.join()

	data.content = data.content.replace("Key.space", "[1, 0, 0]")
	data.content = data.content.replace("Key.left", "[0, 1, 0]")
	data.content = data.content.replace("Key.right", "[0, 0, 1]")
	path = "data/moreData.txt"
	F = open(path, "w")
	F.write(data.content)
	F.close()