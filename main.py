import numpy as np
import matplotlib.pyplot as plt


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


class Sudoku():
    def __init__(self):
        self.field_solution = np.zeros((9, 9))
        self.field_possible = np.reshape([np.arange(1, 10) for i in range(0, 81)], shape=(9, 9, 9))

    def set_number(self, x, y, number):
        self.field_solution[x, y] = number
        return None

    def show_field(self):
        print(self.field_solution)
        return None

    def check_small_box(self, box_x, box_y):
        changed = False
        temp = self.field_possible
        remove_numbers = set([])
        for i in range(box_x, box_x + 3):
            for j in range(box_y, box_y + 3):
                remove_numbers.add(self.field_solution[i, j])
        remove_numbers.remove(0)
        for num in remove_numbers:
            for i in range(box_x, box_x + 3):
                for j in range(box_y, box_y + 3):
                    self.field_possible[i, j, num] = 0
        if abs(temp - self.field_possible) > 0:
            changed = True
        return changed

    def check_row(self, row_idx):
        changed = False
        temp = self.field_possible
        remove_numbers = set([])
        for j in range(0, 9):
            remove_numbers.add(self.field_solution[row_idx, j])
        remove_numbers.remove(0)
        for num in remove_numbers:
            for j in range(0, 9):
                self.field_possible[row_idx, j, num] = 0
        if abs(temp - self.field_possible) > 0:
            changed = True
        return changed

    def check_col(self, col_idx):
        changed = False
        temp = self.field_possible
        remove_numbers = set([])
        for i in range(0, 9):
            remove_numbers.add(self.field_solution[i, col_idx])
        remove_numbers.remove(0)
        for num in remove_numbers:
            for i in range(0, 9):
                self.field_possible[col_idx, i, num] = 0
        if abs(temp - self.field_possible) > 0:
            changed = True
        return changed


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
