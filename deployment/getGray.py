import numpy as np
import cv2 as cv


def get_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def test_get_gray():
    # create a color image (red)
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:, :, 2] = 255

    # convert to grayscale using the function
    gray_img = get_gray(color_img)
    # verify that the output is a grayscale image
    assert gray_img.ndim == 2
    assert np.array_equal(gray_img, np.zeros((100, 100), dtype=np.uint8) + 76)

    # create a color image (green)
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:, :, 1] = 255
    # convert to grayscale using the function
    gray_img = get_gray(color_img)

    # verify that the output is a grayscale image
    assert gray_img.ndim == 2
    assert np.array_equal(gray_img, np.zeros((100, 100), dtype=np.uint8) + 150)

    # create a color image (blue)
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:, :, 0] = 255

    # convert to grayscale using the function
    gray_img = get_gray(color_img)
    cv.imshow("Gray Image",gray_img)
    cv.waitKey()
    # verify that the output is a grayscale image
    assert gray_img.ndim == 2
    assert np.array_equal(gray_img, np.zeros((100, 100), dtype=np.uint8) + 29)


test_get_gray()
