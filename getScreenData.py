import time
import ScreenReader

#returns a tuple of the form (d, min, max) where
#d is a dictionary mapping integer times to 'screen' strings,
#min is the minimum time in the screenpath, and max is the
#maximum time in the screenpath
def getScreenInfo(screenPath):
    F = open(screenPath)
    lines = F.readlines()
    F.close()
    screens = lines[::3]
    times = lines[1::3]
    times = list(map(lambda x: int(x), times))
    d = dict()
    for i in range(len(times)):
        t = times[i]
        screen = screens[i]
        d[t] = screen
    mini = times[0]
    maxi = times[-1]
    return (d, mini, maxi)

def getKeyboardInfo(keyPath):
    F = open(keyPath)
    lines = F.readlines()
    lines = list(map(lambda x: x.split("] "), lines))
    F.close()
    keys = list(map(lambda x: x[0] + "]", lines))
    times = list(map(lambda x: int(x[1][:-1]), lines))
    d = dict()
    for i in range(len(times)):
        t = times[i]
        k = keys[i]
        d[t] = k
    mini = times[0]
    maxi = times[-1]
    return (d, mini, maxi)

def mergeData(screenPath, keyPath, mergedPath):
    screenDict, screenMin, screenMax = getScreenInfo(screenPath)
    keyDict, keyMin, keyMax = getKeyboardInfo(keyPath)
    mini = max(screenMin, keyMin)
    maxi = min(screenMax, keyMax)
    content = ""
    for i in range(mini, maxi + 1):
        screen = screenDict[i]
        keys = keyDict.get(i, None)
        if(keys != None):
            content += "%s%s\n\n" % (screen, keys)
    F = open(mergedPath, "w")
    F.write(content)
    F.close()

def getScreenData(path, maxTime):
    F = open(path, "w")
    content = ""
    start = time.time()
    while(time.time() - start < maxTime):
        t = time.time()
        if(t % 1 < .1):
            content += str(ScreenReader.getInputs()) + "\n"
            content += str(round(t)) + "\n\n"
    F.write(content)
    F.close()

def filterZeroes(path):
    F = open(path)
    lines = F.readlines()
    F.close()
    screens = lines[::3]
    keys = lines[1::3]
    zipped = zip(screens, keys)
    zipped = list(filter((lambda x : x[1] != "[0, 0, 0]\n"), zipped))
    F = open(path, "w")
    for screen, keys in zipped:
        F.write(screen)
        F.write(keys+"\n")
    F.close()

if __name__ == "__main__":
	# path = "screenData.txt"
	# getScreenData(path, 600)
    
    mergeData("screenData.txt", "keyboardData.txt", "mergedData.txt")
    filterZeroes("mergedData.txt")