import numpy as np
import cv2 as cv


class SudokuReader():

    def __init__(self):
        self.solution_field = np.zeros((9, 9))
        self.image = np.empty(dtype=np.uint8)

