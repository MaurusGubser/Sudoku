import cv2
import numpy as np

from SudokuSolver import Sudoku
from SudokuReader import SudokuReader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    reader = SudokuReader()
    path = 'Sudoku_Images/Sudoku3.jpg'
    reader.read_image_from_source(path)

    reader.find_contour_sudoku()
    reader.compute_binary_image(thres=2.3, block_size=5)
    # reader.show_all_images()

    reader.find_candidates()
    reader.load_model('NumberClassifierLarge')
    reader.fill_in_numbers()
    reader.show_sudoku()
    # reader.canny_edge_detection(kernel_size=5, thres_low=100, thres_upper=200)
    # reader.otsu_thresholding(kernel_size=3)
    # reader.show_edge_image()
    # reader.open_image(kernel_shape=cv2.MORPH_CROSS, kernel_size=3)
    # reader.hough_line_detection(rho=1.0, theta=np.pi/30, thres=120)
    # reader.show_hough_line()
    # reader.watershed_detection()
    # reader.harris_corner(block_size=2, ksize=3, k=0.15)
    """
    sudoku = Sudoku()
    sudoku.read_field_from_csv('Sudoku_Examples/Example6_Redundant')
    sudoku.solve()
    """
