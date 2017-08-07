import pytesseract
import pyscreenshot as ImageGrab
from PIL import Image
import string

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'

def charToDigit(c):
	if (c in string.digits): return c
	if (c == 'a' or c == 'c'): return '0'
	if (c == 'i'): return '1'
	if (c == 's'): return '4'
	return '0'

def scoreStrToInt (scoreStr):
	try:
		res = int(''.join(map(charToDigit, scoreStr)))
		if(res > 5000): return res % 1000 #if it's greater than 5000, the first digit is probably incorrect, so remove it
		return res % 10000
	except:
		return 0

def getScore():
	left = 610
	top = 260
	height = 35
	width = 120
	image = ImageGrab.grab((left,top,left + width,top + height))
	scoreStr = pytesseract.image_to_string(image, config="digits")
	score = scoreStrToInt(scoreStr)
	return score

if __name__ == "__main__":
	print(getScore())