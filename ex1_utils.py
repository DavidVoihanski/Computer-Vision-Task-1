import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List

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
global imgGamma
global LOAD_RGB
LOAD_RGB = 2
global LOAD_GRAY_SCALE
LOAD_GRAY_SCALE = 1


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 315873455


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    img = plt.imread(filename)
    if representation == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img / 255
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    if representation == LOAD_GRAY_SCALE:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    h, w, _ = imgRGB.shape
    helper = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.275, -0.321],
                       [0.212, -0.523, 0.311]])
    imgYIQ = imgRGB.reshape((-1, 3)).T
    imgYIQ = helper.dot(imgYIQ).T
    imgYIQ = imgYIQ.reshape((h, w, 3))
    return imgYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    h, w, _ = imgYIQ.shape
    helper = np.array([[0.299, 0.587, 0.114],
                       [0.596, -0.275, -0.321],
                       [0.212, -0.523, 0.311]])
    imgYIQ = imgYIQ.reshape((-1, 3)).T
    helper = np.linalg.inv(helper)
    imgRGB = helper.dot(imgYIQ).T
    imgRGB = imgRGB.reshape((h, w, 3))
    return imgRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    plt.gray()
    plt.imshow(imgOrig)
    plt.show()
    numDim = imgOrig.ndim
    convImage = imgOrig.copy()
    if numDim == 3:
        YIQimage = transformRGB2YIQ(imgOrig)
        convImage = YIQimage[:, :, 0]
    histOrg, bins = np.histogram(convImage * 255, bins=range(0, 257))
    cumSum = np.cumsum(histOrg)
    cumSum = (cumSum / cumSum[-1])
    convImage = convImage * 255
    convImage = convImage.astype("uint8")
    imgEq = cumSum[convImage]
    imgEq = ((imgEq - np.min(imgEq)) / (np.max(imgEq) - np.min(imgEq)))
    histEQ, bins = np.histogram(imgEq*255, bins=range(0, 257))
    if numDim == 3:
        YIQimage[:, :, 0] = imgEq
        imgEq = transformYIQ2RGB(YIQimage)
    plt.gray()
    plt.imshow(imgEq)
    plt.show()
    return imgEq, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # plt.gray()
    # plt.imshow(imOrig)
    # plt.show()
    imgCopy = imOrig.copy()
    numDim = imOrig.ndim
    if numDim == 3:
        w, h, _ = imOrig.shape
        imgYIQ = transformRGB2YIQ(imOrig)
        imgCopy = imgYIQ[:, :, 0]
    else:
        w, h = imOrig.shape
    hist, bins = np.histogram(imgCopy * 255, bins=range(0, 256))
    means = np.zeros(nQuant)
    jumps = np.zeros(nQuant + 1).astype("uint")
    jumps[0] = 0
    jumps[-1] = 255
    pixelsInEach = w*h/nQuant
    sum_ = 0
    index = 1
    imgCopy = imgCopy * 255
    newImage = imgCopy.copy()
    errors = []
    images = []
    for i in range(0, 255):  # border init
        sum_ += hist[i]
        if sum_ >= pixelsInEach:
            jumps[index] = i
            index += 1
            sum_ = 0
            if index == nQuant:
                break

    for j in range(0, nIter):
        for i in range(1, nQuant + 1):  # calc means
            start = jumps[i - 1]
            end = jumps[i]
            mean = np.sum(hist[start:end]*bins[start:end]) / np.sum(hist[start:end])
            means[i - 1] = mean

        for i in range(1, nQuant):  # calc borders
            newJump = (means[i - 1] + means[i]) / 2
            newJump = round(newJump)
            jumps[i] = newJump

        index = nQuant - 1
        while index >= 0:  # apply changes
            start = jumps[index + 1]
            newVal = means[index]
            newImage[imgCopy <= start] = newVal
            index -= 1
        newImage = newImage / 255
        imageToAppend = newImage.copy()
        if numDim == 3:
            imgYIQ[:, :, 0] = newImage
            imageToAppend = transformYIQ2RGB(imgYIQ)
        images.append(imageToAppend)
        mse = np.mean(np.power(imOrig*255 - imageToAppend*255, 2))
        errors.append(mse)
    # plt.gray()
    # plt.imshow(imageToAppend)
    # plt.show()
    # plt.plot(errors)
    # plt.show()
    return images, errors


if __name__ == "__main__":
#     # img = imReadAndConvert("/home/david/Downloads/beach.jpg", 2)
#     # plt.imshow(img)
#     # plt.show()
#     # plt.gray()
#     # img = transformRGB2YIQ(img)
#     # plt.imshow(img[:, :, 0])
#     # plt.show()
#     # img = transformYIQ2RGB(img)
#     # plt.imshow(img)
#     # plt.show()
    img = imReadAndConvert("testImg2.jpg", 2)
     # plt.gray()
     # plt.imshow(img)
#     eqimg, hist, newhist = hsitogramEqualize(img)
# #     # # plt.imsave("test.jpg", eqimg)
# #     # # print(newhist)
#     plt.figure()
#     plt.imshow(eqimg)
#     plt.show()
    quantizeImage(img, 5, 17)
#     gammaDisplay("/home/david/Downloads/beach.jpg", 2)
