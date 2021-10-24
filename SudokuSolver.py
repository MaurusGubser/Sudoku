import numpy as np
import pandas as pd


class Sudoku:
    def __init__(self):
        self.field_solution = np.zeros((9, 9), dtype=int)
        self.field_possible = np.reshape([np.arange(1, 10, dtype=int) for i in range(0, 81)], (9, 9, 9))

    def read_field_from_csv(self, path_to_csv):
        df = pd.read_csv(path_to_csv, header=None)
        self.field_solution = df.to_numpy()
        return None

    def show_field(self):
        print(self.field_solution)
        return None

    def check_possibility(self, x, y, number):
        for row in range(0, 9):
            if self.field_solution[x, row] == number:
                return False
        for column in range(0, 9):
            if self.field_solution[column, y] == number:
                return False
        box_col = x // 3
        box_row = y // 3
        for i in range(3*box_col, 3*box_col + 3):
            for j in range(3*box_row, 3*box_row + 3):
                if self.field_solution[i, j] == number:
                    return False
        return True

    def solve(self):
        for x in range(0, 9):
            for y in range(0, 9):
                if self.field_solution[x, y] == 0:
                    for n in range(1, 10):
                        if self.check_possibility(x, y, n):
                            self.field_solution[x, y] = n
                            self.solve()
                            self.field_solution[x, y] = 0
                    return None
        self.show_field()
        input('Show more solutions?')
