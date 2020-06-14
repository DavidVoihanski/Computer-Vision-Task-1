"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np

global imgGamma


def gammaDisplay(img_path: str, rep: int) -> None:
    global imgGamma
    imgGamma = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        imgGamma = cv2.cvtColor(imgGamma, cv2.COLOR_RGB2GRAY)
    imgGamma = imgGamma/255
    cv2.namedWindow("Gamma correction")
    cv2.createTrackbar("Gamma", "Gamma correction", 1, 200, onTrackBar)
    cv2.imshow("Gamma correction", imgGamma)
    onTrackBar(100)
    cv2.waitKey()


def onTrackBar(val):
    global imgGamma
    val = val/100
    newImg = np.power(imgGamma, val)
    cv2.imshow("Gamma correction", newImg)


def main():
    gammaDisplay("/home/david/Downloads/beach.jpg", 2)


if __name__ == '__main__':
    main()
