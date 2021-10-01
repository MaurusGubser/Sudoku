import numpy as np
import cv2 as cv


class SudokuReader():

    def __init__(self):
        self.solution_field = np.zeros((9, 9))
        self.image = np.empty((1, 1, 3), dtype=np.uint8)
        self.image_gray = np.empty((1, 1), dtype=np.uint8)

    def read_image_from_source(self, path_src):
        self.image = cv.imread(path_src)
        fx = 1024 // self.image.shape[0]
        fy = 768 // self.image.shape[1]
        self.image = cv.resize(self.image, dsize=(1024, 768))
        self.image_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        return None

    def show_image(self):
        cv.imshow('Sudoku image', self.image)
        cv.waitKey(0)
        return None

    def save_image(self, path_dest):
        cv.imwrite(path_dest, self.image)
        return None

    def harris_corner(self):
        self.image_gray = np.float32(self.image_gray)
        blockSize = 5
        ksize = 5
        k = 0.01
        dst = cv.cornerHarris(self.image_gray, blockSize, ksize, k)
        dst = cv.dilate(dst, None)
        self.image[dst > 0.01 * dst.max()] = [0, 0, 255]
        return None
