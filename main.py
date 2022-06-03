from SudokuSolver import SudokuSolver
from SudokuReader import SudokuReader


def read_and_solve_sudoku(path_sudoku_img):
    path_clf = 'model-OCR.h5'
    reader = SudokuReader(path_img=path_sudoku_img, path_clf=path_clf, show_steps=True)
    if reader.get_sudoku_field_from_image():
        input_field = reader.sudoku_field
        solver = SudokuSolver()
        if solver.set_sudoku_field(input_field):
            if solver.find_one_solution():
                print('Found a solution.')
                reader.show_solution_on_sudoku(solver.solutions[0])
                # reader.save_solution_on_image(solver.field_sudoku, path_sudoku_img)
            else:
                raise RuntimeError('Did not find a solution')
    return None

path = 'Sudoku_Images/Test3.jpg'
read_and_solve_sudoku(path)
