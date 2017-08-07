import pyscreenshot as ImageGrab
import time
from functools import reduce
import operator

from tkinter import *

screenRows = 22
screenCols = 87

#from the given image and bounds, returns an (r,g,b) approximately
#representing the average pixel in the box
def getRBGBox(img, left, top, right, bot):
    sumR, sumG, sumB = 0, 0, 0
    total = 0
    for x in range(left, right, 3):
        for y in range(top, bot, 6):
            (r,g,b) = img.getpixel((x,y))
            sumR += r
            sumG += g
            sumB += b
            total += 1
    return (sumR // total, sumG // total, sumB // total)

def getPixels():
    left = 610
    top = 330
    screenWidth = 350
    screenHeight = 270
    pixelWidth = 4
    pixelHeight = 12

    image = ImageGrab.grab(bbox = (left, top, left + screenWidth, top + screenHeight))
    pixels = []
    for y in range(0, screenHeight - pixelHeight, pixelHeight):
        rowPixels = []
        for x in range(0, screenWidth - pixelWidth, pixelWidth):
            rowPixels.append(getRBGBox(image, x, y, x + pixelWidth, y + pixelHeight))
        pixels.append(rowPixels)
    return pixels

def averageBlue(pixels):
    rows, cols = len(pixels), len(pixels[0])
    blueSum = 0
    total = 0
    for row in range(rows):
        for col in range(cols):
            blueSum += pixels[row][col][2]
            total += 1
    return blueSum / total

def getBrightness(pixel):
    return reduce(operator.add, pixel)

#gets the inputs that will be fed to the neural network
def getPixelInputs(pixels):
    res = []
    rows, cols = len(pixels), len(pixels[0])
    for row in range(rows):
        for col in range(cols):
            pixel = pixels[row][col]
            res.append(1 if getBrightness(pixel) > 10 else 0)
    return res

def getInputs():
    pixels = getPixels()
    return getPixelInputs(pixels)

####################################
# Used to debug
####################################

def getPixelInputs2(pixels):
    res = []
    rows, cols = len(pixels), len(pixels[0])
    for row in range(rows):
        resRow = []
        for col in range(cols):
            pixel = pixels[row][col]
            resRow.append(1 if getBrightness(pixel) > 10 else 0)
        res.append(resRow)
    return res

#taken from http://www.cs.cmu.edu/~112/notes/notes-graphics.html
def rgbString(red, green, blue):
    return "#%02x%02x%02x" % (red, green, blue)

def init(data):
    data.left = 610
    data.top = 330
    data.screenWidth = 350
    data.screenHeight = 270
    data.pixelWidth = 4
    data.pixelHeight = 12
    data.inputPixels = None

def mousePressed(event, data):
    # use event.x and event.y
    pass

def keyPressed(event, data):
    # use event.char and event.keysym
    pass

def timerFired(data):
    image = ImageGrab.grab(bbox = (data.left, data.top,
                data.left + data.screenWidth, data.top + data.screenHeight))
    pixels = getPixels()
    data.inputPixels = getPixelInputs2(pixels)

def redrawAll(canvas, data):
    if(data.inputPixels == None): pass

    rows, cols = len(data.inputPixels), len(data.inputPixels[0])
    print(rows, cols)
    deltaX = data.width / cols
    deltaY = data.height / rows
    for row in range(rows):
        for col in range(cols):
            color = "white" if data.inputPixels[row][col] else "black"
            yTop = row * deltaY
            yBot = yTop + deltaY
            xLeft = col * deltaX
            xRight = xLeft + deltaX
            canvas.create_rectangle(xLeft, yTop, xRight, yBot, fill = color)

####################################
# use the run function as-is
####################################

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds
    init(data)
    # create the root and the canvas
    root = Tk()
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

if __name__ == "__main__":
	run(400, 600)

    # start = time.time()
    # for i in range(20):
    #     p = getPixels()
    #     rows = len(p)
    #     cols = len(p[0])
    #     print(rows, cols)