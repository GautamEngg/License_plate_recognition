import cv2
import imutils
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

img = cv2.imread('car9.jpg', cv2.IMREAD_COLOR)

img = cv2.resize(img, (600, 400))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to grey scale
cv2.imshow('Resized & Gray scaled Image', gray)

gray = cv2.bilateralFilter(gray, 13, 15, 15)
cv2.imshow('Blurr Background', gray)

edged = cv2.Canny(gray, 0, 200)
cv2.imshow('Edge Detection', edged)

contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None
# cv2.imshow('Edge Detected', gray)

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break
if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
new_image = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('New Masked Image', new_image)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

text = pytesseract.image_to_string(Cropped, config='--psm 11')
print("Detected license plate Number is:", text)
img = cv2.resize(img, (500, 300))
Cropped = cv2.resize(Cropped, (400, 200))
cv2.imshow('car', img)
cv2.imshow('Cropped', Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
