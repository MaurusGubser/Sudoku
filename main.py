from Sudoku import Sudoku
from ReadSudoku import SudokuReader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    reader = SudokuReader()
    path = 'Sudoku_Images/Sudoku4.jpg'
    reader.read_image_from_source(path)
    #reader.show_image()
    #reader.harris_corner()
    reader.compute_binary_image(thres=2.3)
    #reader.otsu_thresholding(kernel_size=3)
    reader.show_all_images()
    reader.open_image(3)
    reader.show_all_images()

    """
    sudoku = Sudoku()
    sudoku.read_field_from_csv('Sudoku_Examples/Example6_Redundant')
    sudoku.solve()
    """
