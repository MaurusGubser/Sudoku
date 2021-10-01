from Sudoku import Sudoku
from ReadSudoku import SudokuReader

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    reader = SudokuReader()
    path = 'Sudoku_Images/Sudoku7.jpg'
    reader.read_image_from_source(path)
    #reader.show_image()
    reader.harris_corner()
    reader.show_image()

    """
    sudoku = Sudoku()
    sudoku.read_field_from_csv('Sudoku_Examples/Example6_Redundant')
    sudoku.solve()
    """
