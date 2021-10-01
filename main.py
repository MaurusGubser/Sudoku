from Sudoku import Sudoku

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sudoku = Sudoku()
    sudoku.read_field_from_csv('Sudoku_Examples/Example4_Intermediate')
    sudoku.solve_sudoku()
