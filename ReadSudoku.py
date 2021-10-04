import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


class SudokuReader():

    def __init__(self):
        self.solution_field = np.zeros((9, 9))
        self.image = np.empty((1, 1, 3), dtype=np.uint8)
        self.image_gray = np.empty((1, 1), dtype=np.uint8)
        self.edges = np.empty((1, 1), dtype=np.uint8)
        self.image_binary = np.empty((1, 1), dtype=np.uint8)

    def read_image_from_source(self, path_src):
        self.image = cv.imread(path_src)
        long_side = max(self.image.shape[0], self.image.shape[1])
        scale_factor = 800 / long_side
        self.image = cv.resize(self.image, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
        self.image_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        return None

    def show_all_images(self):
        plt.subplot(131)
        plt.imshow(self.image)
        plt.subplot(132)
        plt.imshow(self.image_gray)
        plt.subplot(133)
        plt.imshow(self.image_binary)
        #cv.waitKey(0)
        plt.show()
        return None

    def show_images(self):
        plt.subplot(121)
        plt.imshow('Sudoku image gray', self.image_gray)
        plt.subplot(122)
        plt.imshow('Sudoku image edges', self.edges)
        plt.show()
        #cv.waitKey(0)
        return None

    def save_image(self, path_dest):
        cv.imwrite(path_dest, self.image)
        return None

    def compute_binary_image(self, thres):
        self.image_binary = cv.adaptiveThreshold(self.image_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, C=thres)
        return None

    def open_image(self, kernel_size):
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (kernel_size, kernel_size))
        self.image_binary = cv.morphologyEx(self.image_binary, cv.MORPH_OPEN, kernel)
        return None

    def otsu_thresholding(self, kernel_size):
        blur = cv.GaussianBlur(self.image_gray, (kernel_size, kernel_size), 0)
        _, self.image_binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return None

    def canny_edge_detection(self, thres_low, thres_upper):
        self.edges = cv.Canny(self.image_gray, thres_low, thres_upper)
        return None

    def harris_corner(self):
        self.image_gray = np.float32(self.image_gray)
        blockSize = 2
        ksize = 3
        k = 0.15
        dst = cv.cornerHarris(self.image_gray, blockSize, ksize, k)
        dst = cv.dilate(dst, None)
        self.image[dst > 0.01 * dst.max()] = [0, 0, 255]
        return None
