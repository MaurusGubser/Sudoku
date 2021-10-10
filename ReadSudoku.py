import numpy as np
import matplotlib
import cv2 as cv

import matplotlib.pyplot as plt

matplotlib.use('tkagg')


class SudokuReader():

    def __init__(self):
        self.solution_field = np.zeros((9, 9))
        self.image = np.empty((1, 1, 3), dtype=np.uint8)
        self.image_gray = np.empty((1, 1), dtype=np.uint8)
        self.edges = np.empty((1, 1), dtype=np.uint8)
        self.image_binary = np.empty((1, 1), dtype=np.uint8)
        self.lines = np.empty((1, 1), dtype=np.uint8)

    def read_image_from_source(self, path_src):
        self.image = cv.imread(path_src)
        long_side = max(self.image.shape[0], self.image.shape[1])
        scale_factor = 800 / long_side
        self.image = cv.resize(self.image, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
        self.image_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        return None

    def show_all_images(self):
        fig, axs = plt.subplots(nrows=1, ncols=3)
        axs[0].imshow(self.image)
        axs[1].imshow(self.image_gray, cmap='gray')
        axs[2].imshow(self.image_binary, cmap='gray')
        plt.show()
        return None

    def show_all_images_opencv(self):
        cv.imshow('Sudoku image', self.image)
        cv.imshow('Sudoku image gray', self.image_gray)
        cv.imshow('Sudoku image gray', self.image_binary)
        cv.waitKey(0)
        return None

    def show_edge_image(self):
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(self.image_gray, cmap='gray')
        axs[1].imshow(self.edges, cmap='gray')
        plt.show()
        return None

    def show_hough_line(self):
        line_img = self.image.copy()
        len_x, len_y, _ = self.image.shape
        long_side = max(len_x, len_y)
        for line in self.lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + long_side * (-b))
            y1 = int(y0 + long_side * (a))
            x2 = int(x0 - long_side * (-b))
            y2 = int(y0 - long_side * (a))
            cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        plt.imshow(line_img)
        plt.show()

    def save_images(self, path_dest):
        cv.imwrite(path_dest, self.image)
        cv.imwrite(path_dest + '_gray', self.image_gray)
        cv.imwrite(path_dest + '_binary', self.image_binary)
        return None

    def compute_binary_image(self, thres=2.0, block_size=5):
        self.image_binary = cv.adaptiveThreshold(self.image_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                                 blockSize=block_size, C=thres)
        return None

    def open_image(self, kernel_shape=cv.MORPH_CROSS, kernel_size=3):
        kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        self.image_binary = cv.morphologyEx(self.image_binary, cv.MORPH_OPEN, kernel)
        return None

    def otsu_thresholding(self, kernel_size=3):
        blur = cv.GaussianBlur(self.image_gray, (kernel_size, kernel_size), 0)
        _, self.image_binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return None

    def canny_edge_detection(self, thres_low=100, thres_upper=200):
        self.edges = cv.Canny(self.image_gray, thres_low, thres_upper)
        return None

    def hough_line_detection(self, rho=1, theta=np.pi / 180, thres=200):
        self.lines = cv.HoughLines(self.edges, rho=rho, theta=theta, threshold=thres)
        return None

    def watershed_detection(self, kernel_size=3, mask_size=5):
        # noise removal
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv.morphologyEx(thresh, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
        # sure background area
        sure_bg = cv.dilate(opening, kernel=kernel, iterations=1)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, distanceType=cv.DIST_L2, maskSize=mask_size)
        ret, sure_fg = cv.threshold(dist_transform, thresh=0.7 * dist_transform.max(), maxval=255, type=0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        fig, axs = plt.subplots(nrows=2, ncols=3)
        axs[0, 0].imshow(self.image)
        axs[0, 1].imshow(thresh)
        axs[0, 2].imshow(sure_bg)
        axs[1, 0].imshow(dist_transform)
        axs[1, 1].imshow(sure_fg)
        axs[1, 2].imshow(unknown)
        plt.show()
        return None

    def harris_corner(self, block_size=2, ksize=3, k=0.15):
        self.image_gray = np.float32(self.image_gray)
        dst = cv.cornerHarris(self.image_gray, blockSize=block_size, ksize=ksize, k=k)
        dst = cv.dilate(dst, None)
        self.image[dst > 0.01 * dst.max()] = [0, 0, 255]
        return None
