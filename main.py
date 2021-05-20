import numpy as np
import matplotlib.pyplot as plt


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class Sudoku:
    def __init__(self):
        self.field_solution = np.zeros((9, 9), dtype=int)
        self.field_possible = np.reshape([np.arange(1, 10, dtype=int) for i in range(0, 81)], (9, 9, 9))

    def set_number(self, x, y, number):
        self.field_solution[x, y] = number
        return None

    def show_field(self):
        print(self.field_solution)
        return None

    def show_possibilities(self):
        print(self.field_possible)
        return None

    def update_small_box(self, box_x, box_y):
        changed = False
        temp = self.field_possible
        remove_numbers = set([])
        for i in range(box_x*3, box_x*3 + 3):
            for j in range(box_y*3, box_y*3 + 3):
                remove_numbers.add(self.field_solution[i, j])
        remove_numbers.remove(0)
        for num in remove_numbers:
            for i in range(box_x*3, box_x*3 + 3):
                for j in range(box_y*3, box_y*3+ 3):
                    self.field_possible[i, j, int(num)] = 0
        if np.any(abs(temp - self.field_possible)) > 0:
            changed = True
        return changed

    def update_row(self, row_idx):
        changed = False
        temp = self.field_possible
        remove_numbers = set([])
        for j in range(0, 9):
            remove_numbers.add(self.field_solution[row_idx, j])
        remove_numbers.remove(0)
        for num in remove_numbers:
            for j in range(0, 9):
                self.field_possible[row_idx, j, int(num)] = 0
        if np.any(abs(temp - self.field_possible)) > 0:
            changed = True
        return changed

    def update_col(self, col_idx):
        changed = False
        temp = self.field_possible
        remove_numbers = set([])
        for i in range(0, 9):
            remove_numbers.add(self.field_solution[i, col_idx])
        remove_numbers.remove(0)
        for num in remove_numbers:
            for i in range(0, 9):
                self.field_possible[col_idx, i, int(num)] = 0
        if np.any(abs(temp - self.field_possible)) > 0:
            changed = True
        return changed

    def check_for_solved(self):
        if np.min(self.field_solution) == 0:
            return False
        else:
            return True


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sudoku_exple = Sudoku()
    sudoku_exple.show_field()
    sudoku_exple.set_number(8, 8, 1)
    sudoku_exple.show_field()
    sudoku_exple.show_possibilities()
    sudoku_exple.update_small_box(2, 2)
    sudoku_exple.update_row(8)
    sudoku_exple.update_col(8)
    print('Update')
    sudoku_exple.show_possibilities()

