import time
from pynput import keyboard

def on_press(data):
	def on_press2(key):
		if (key == keyboard.Key.left or
				key == keyboard.Key.right or
				key == keyboard.Key.space):
			data.content += '{0} Pressed '.format(key)
			data.content += str(time.time()) + "\n"
	return on_press2

def on_release(data):
	def on_release2(key):
		if (key == keyboard.Key.left or
				key == keyboard.Key.right or
				key == keyboard.Key.space):
			data.content += '{0} Released '.format(key)
			data.content += str(time.time()) + "\n"
		if key == keyboard.Key.esc:
			# Stop listener
			return False
	return on_release2

def updatePressed(currentPressed, key, action):
	i = None
	if(key == "Key.space"): i = 0
	elif(key == "Key.left"): i = 1
	elif(key == "Key.right"): i = 2
	currentPressed[i] = 1 if action == "Pressed" else 0

def getPressedKeys(inPath, outPath):
	F = open(inPath)
	lines = F.readlines()
	F.close()
	currentPressed = [0, 0, 0] #space, left, right
	currentTime = None
	content = ""
	for line in lines:
		words = line.split()
		key = words[0]
		action = words[1]
		t = int(float(words[2]))
		if(currentTime != None and t > currentTime):
			content += "%s %d\n" % (str(currentPressed), currentTime + 1)
		updatePressed(currentPressed, key, action)
		currentTime = t
	F = open(outPath, "w")
	F.write(content)
	F.close()

def combineFiles():
	content = ""
	for i in range(fileNum + 1):
		filePath = path.replace(".txt", str(i) + ".txt")
		F = open(filePath)
		content += F.read()
		F.close()
	F = open(path)
	F.write(content)
	F.close()

class struct:
	pass

if __name__ == "__main__":
	data = struct()
	data.content = ""
	# Collect events until released
	with keyboard.Listener(
	        on_press=on_press(data),
	        on_release=on_release(data)) as listener:
	    listener.join()

	path = "keyboardData.txt"
	F = open(path, "w")
	F.write(data.content)
	F.close()
	getPressedKeys(path, path)