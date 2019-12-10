import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#   This alghoritm was created to detect buildings on a satelite picture by threshold and cv.findContours
#   You can modify it to work properly with any picture by removing usun_zielone function
#   It works pretty bad if object and background are in similiar colour

def srodek(contour):
#   Finds the middle of the contour

    M = cv.moments(contour)
    cx = int(M['m10'] /M['m00'])
    cy = int(M['m01'] /M['m00'])
    return([cy,cy])

def odl_xy(x, y):
#   Finds the distance between x and y points

    dx = x[0] - y[0]
    dy = x[1] - y[1]
    D = np.math.sqrt(dx*dx+dy*dy)
    return D

def kat(a, b, c):
#   Returns angle in degrees between points a, b and c

    v0 = np.array(a) - np.array(b)
    v1 = np.array(c) - np.array(b)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)
    

def proste_katy(contour):
#   Checks if contour has angles near 90 or 180 degrees

    a=80
    b=110
    c=170
    k = abs( kat(contour[len(contour)-1][0],contour[0][0],contour[1][0]) )
    if k < a or (k > b and k < c):
        return False

    k = abs( kat(contour[len(contour)-2][0],contour[len(contour)-1][0],contour[0][0]) )
    if k < a or (k > b and k < c):
        return False

    for i in range(0, len(contour)-1):
        k = abs( kat(contour[i-1][0],contour[i][0],contour[i+1][0]) )
        if k < a or (k > b and k < c):
            return False
    return True


def selekcja_kontorow(contours, min, max):
#   Deletes contours not matching area or angle requirements

    a = len(contours)
    k = 0
    while (k < a):
        r = cv.minAreaRect(contours[k])[1]
        if cv.contourArea( contours[k] ) < min or cv.contourArea( contours[k] ) > max or not proste_katy(contours[k]) or r[0]/r[1] > 6 or r[1]/r[0] > 6:
            del contours[k]
            a = len(contours)
            k = k - 1
        k = k + 1
    return contours


def wyostrz(img):
#   Sharpens or blurs the picture

    blur = cv.GaussianBlur(img, (5,5), 3)
    img = cv.addWeighted(img, 1.5, blur, -0.5, 0)
    return blur


def usun_zielone(img):
#   Removes green areas (grass)

    lower_green = np.array([30,30,10])
    upper_green = np.array([80,255,220])

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = cv.GaussianBlur(hsv,(5,5),0)
    mask = cv.inRange(hsv, lower_green, upper_green)

    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j] == 255:
                mask[i][j] = 0
            else:
                mask[i][j] = 255
    img = cv.bitwise_and(img, img, mask= mask)
    return img


def rysuj_kontury(img, contours):
#   Draws contours and their corners

    cv.drawContours(img, contours, -1, (0,255,0), 2)

    for i in range(len(contours)):
        for j in range(len(contours[i])):
            cv.drawMarker(img, (contours[i][j][0][0],contours[i][j][0][1]),
            [255,0,0], cv.MARKER_SQUARE, 0, 1)
    return img

def usun_duplikaty(contours):
#   Deletes contours detected several times

    a = len(contours)
    i = 0
    while i < a:
        x = srodek(contours[i])
        j = i + 1
        while j < a:
            y = srodek(contours[j])
            if odl_xy(x,y) < 3:
                del contours[j]
                a = len(contours)
                j = j - 1
            j = j + 1
        i = i + 1
    return contours

def kontury_budynkow(img):
    
#   Returns an array of contours detected on the img file
    result = []
    img = usun_zielone(img)
    imgSplit = cv.split(img)

#   Contour search process is executed separately for R, G and B channels and for different threshold options
    for y in range(0,2):
        for b in range(11, 511, 50):
            for g in range(-20,-6,2):
                imgray = wyostrz(imgSplit[y])

                # Applies threshold, morph and detects contours
                thresh  = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,  b, g)
                kernel  = np.ones((3,3),np.uint8)
                morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
                morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)

                contours, hierarchy = cv.findContours(morph, cv.RETR_TREE, cv.CHAIN_APPROX_NONE )

                for i in range(len(contours)):
                    contours[i] = cv.approxPolyDP(contours[i], 0.02*cv.arcLength(contours[i],True), True)

                contours = selekcja_kontorow(contours, 100, 10000)

                for x in contours:
                    result.append(x)

    return result

#   You can see how the program works by using included test.jpg files
img = cv.imread('test2.jpg')
contours = kontury_budynkow(img)
contours = usun_duplikaty(contours)
img = rysuj_kontury(img, contours)

plt.imshow(img)
plt.show()