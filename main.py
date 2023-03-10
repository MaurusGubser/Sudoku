import sys

from SudokuReader import SudokuReader
from SudokuSolver import SudokuSolver


def read_and_solve_sudoku(path_sudoku_img):
    """
    Reads an image, processes it and searches for a sudoku field and its numbers. If found, the sudoku is solved and
    the solution gets drawn onto the input image.
    :param path_sudoku_img: string
        path to the sudoku field
    """
    path_clf = 'model-OCR.h5'
    reader = SudokuReader(path_img=path_sudoku_img, path_clf=path_clf, show_steps=True, show_predictions=False)
    if reader.get_sudoku_field_from_image():
        input_field = reader.sudoku_field
        solver = SudokuSolver()
        if solver.set_sudoku_field(input_field):
            if solver.find_one_solution():
                print('Found a solution.')
                reader.write_solution_to_image(solver.field_sudoku, path_sudoku_img)
            else:
                raise RuntimeError('Did not find a solution')
    return None


if __name__ == "__main__":
    """
    sys.argv is used to take arbitrary parameters from the command line. In this case a parameter is expected which
    describes the path to a sudoku image.
    """
    read_and_solve_sudoku(sys.argv[1])
