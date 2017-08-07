import GetTrainingData
import ScreenReader

screenRows = ScreenReader.screenRows
screenCols = ScreenReader.screenCols

def getShipArea(screen):
	screen = [screen[i : i+screenCols] for i in range(0, screenCols*screenRows, screenCols)]
	bottomRow = screen[-1]
	rangeWidth = 7
	for i in range(1, len(bottomRow)-2):
		if(bottomRow[i : i + 3] == [1,1,1]):
			return (i - 1, i - 1 + rangeWidth)
	return (0, rangeWidth)

def getShipSection(screen, area):
	left = area[0]
	right = area[1]
	screen = [screen[i : i+screenCols] for i in range(0, screenCols*screenRows, screenCols)]
	res = []
	for row in screen:
		section = row[left:right]
		res += section
	return res

#takes a screen in the form of a 1D list, adds ship focusing to the list
def addShipFocusing(screen):
	area = getShipArea(screen)
	section = getShipSection(screen, area)
	return screen + section

#takes a path to a txt file of data, adds ship focusing to each screen input
def addShipFocusingToData(path):
	data = GetTrainingData.getData(path)
	F = open(path, "w")
	for point in data:
		screen = point[0].tolist()
		keys = point[1].tolist()
		inputs = addShipFocusing(screen)
		F.write(str(inputs) + "\n")
		F.write(str(keys) + "\n\n")
	F.close()

if __name__ == "__main__":
	pass