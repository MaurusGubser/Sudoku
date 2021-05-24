import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


class Sudoku:
    def __init__(self):
        self.field_solution = np.zeros((9, 9), dtype=int)
        self.field_possible = np.reshape([np.arange(1, 10, dtype=int) for i in range(0, 81)], (9, 9, 9))

    def set_number(self, i, j, number):
        self.field_solution[i, j] = number
        return None

    def read_field_from_csv(self, path_to_csv):
        df = pd.read_csv(path_to_csv, header=None)
        self.field_solution = df.to_numpy()
        return None

    def show_field(self):
        print(self.field_solution)
        return None

    def show_possibilities(self):
        print(self.field_possible)
        return None

    def update_single_possible(self):
        changed = False
        for i in range(0, 9):
            for j in range(0, 9):
                if self.field_solution[i, j] > 0:
                    number = self.field_solution[i, j]
                    box_i, box_j = self.get_small_box(i, j)
                    if len(set(self.field_possible[i, j, :]).difference([0])) > 1:
                        changed = True
                    self.field_possible[i, j, :] = 0
                    self.field_possible[i, :, number - 1] = 0
                    self.field_possible[:, j, number - 1] = 0
                    self.field_possible[box_i * 3:box_i * 3 + 3, box_j * 3:box_j * 3 + 3, number - 1] = 0
        return changed

    def update_small_box_possible(self, box_i, box_j):
        changed = False
        temp = copy.copy(self.field_possible)
        remove_numbers = set([])
        for i in range(box_i * 3, box_i * 3 + 3):
            for j in range(box_j * 3, box_j * 3 + 3):
                remove_numbers.add(self.field_solution[i, j])
        remove_numbers = remove_numbers.difference([0])
        for num in remove_numbers:
            for i in range(box_i * 3, box_i * 3 + 3):
                for j in range(box_j * 3, box_j * 3 + 3):
                    self.field_possible[i, j, num - 1] = 0
        if np.any(abs(temp - self.field_possible)) > 0:
            changed = True
        return changed

    def update_row_possible(self, row_idx):
        changed = False
        temp = copy.copy(self.field_possible)
        remove_numbers = set([])
        for j in range(0, 9):
            remove_numbers.add(self.field_solution[row_idx, j])
        remove_numbers = remove_numbers.difference([0])
        for num in remove_numbers:
            for j in range(0, 9):
                self.field_possible[row_idx, j, num - 1] = 0
        if np.any(abs(temp - self.field_possible)) > 0:
            changed = True
        return changed

    def update_col_possible(self, col_idx):
        changed = False
        temp = copy.copy(self.field_possible)
        remove_numbers = set([])
        for i in range(0, 9):
            remove_numbers.add(self.field_solution[i, col_idx])
        remove_numbers = remove_numbers.difference([0])
        for num in remove_numbers:
            for i in range(0, 9):
                self.field_possible[i, col_idx, num - 1] = 0
        if np.any(abs(temp - self.field_possible)) > 0:
            changed = True
        return changed

    def update_field_possible(self):
        temp = copy.copy(self.field_possible)
        changed = self.update_single_possible()
        for i in range(0, 9):
            change = True
            while change:
                change = self.update_row_possible(i)
        for j in range(0, 9):
            change = True
            while change:
                change = self.update_col_possible(j)
        for box_i in range(0, 3):
            for box_j in range(0, 3):
                change = True
                while change:
                    change = self.update_small_box_possible(box_i, box_j)
        if np.any(abs(temp - self.field_possible)) > 0:
            changed = True
        return changed

    def get_small_box(self, i, j):
        box_i = i // 3
        box_j = j // 3
        return box_i, box_j

    def update_single_solution(self):
        changed = False
        for i in range(0, 9):
            for j in range(0, 9):
                possible_num = set(self.field_possible[i, j, :]).difference([0])
                if len(possible_num) == 1 and self.field_solution[i, j] == 0:
                    number = possible_num.pop()
                    box_i, box_j = self.get_small_box(i, j)
                    self.field_solution[i, j] = number
                    self.field_possible[i, j, :] = 0
                    self.field_possible[i, :, number - 1] = 0
                    self.field_possible[:, j, number - 1] = 0
                    self.field_possible[box_i * 3:box_i * 3 + 3, box_j * 3:box_j * 3 + 3, number - 1] = 0
                    changed = True
        return changed

    def update_small_box_solution(self, box_i, box_j):
        changed = False
        for num in range(1, 10):
            candidates_idxs = np.argwhere(
                self.field_possible[box_i * 3:box_i * 3 + 3, box_j * 3:box_j * 3 + 3, num - 1] == num)
            if candidates_idxs.size == 1:
                self.field_solution[box_i + candidates_idxs[0], box_j + candidates_idxs[1]] = num
                self.field_possible[box_i * 3:box_i * 3 + 3, box_j * 3:box_j * 3 + 3, num - 1] = 0
                self.field_possible[box_i + candidates_idxs[0], :] = 0
                self.field_possible[:, box_j + candidates_idxs[1]] = 0
                changed = True
        return changed

    def update_row_solution(self, row_idx):
        changed = False
        for num in range(1, 10):
            candidates_idxs = np.argwhere(self.field_possible[row_idx, :, num - 1] == num).flatten()
            if candidates_idxs.size == 1:
                self.field_solution[row_idx, candidates_idxs[0]] = num
                self.field_possible[row_idx, :, num - 1] = 0
                self.field_possible[:, candidates_idxs[0], num - 1] = 0
                box_i, box_j = self.get_small_box(row_idx, candidates_idxs[0])
                self.field_possible[box_i * 3:box_i * 3 + 3, box_j * 3:box_j * 3 + 3, num - 1] = 0
                changed = True
        return changed

    def update_column_solution(self, column_idx):
        changed = False
        for num in range(1, 10):
            candidates_idxs = np.argwhere(self.field_possible[:, column_idx, num - 1] == num).flatten()
            if candidates_idxs.size == 1:
                self.field_solution[candidates_idxs[0], column_idx] = num
                self.field_possible[:, column_idx, num - 1] = 0
                self.field_possible[candidates_idxs[0], :, num - 1] = 0
                box_i, box_j = self.get_small_box(candidates_idxs[0], column_idx)
                self.field_possible[box_i * 3:box_i * 3 + 3, box_j * 3:box_j * 3 + 3, num - 1] = 0
                changed = True
        return changed

    def update_field_solution(self):
        temp = copy.copy(self.field_solution)
        changed = self.update_single_solution()
        for i in range(0, 9):
            change = True
            while change:
                change = self.update_row_solution(i)
        for j in range(0, 9):
            change = True
            while change:
                change = self.update_column_solution(j)
        for box_i in range(0, 3):
            for box_j in range(0, 3):
                change = True
                while change:
                    change = self.update_small_box_solution(box_i, box_j)
        if np.any(abs(temp - self.field_solution)) > 0:
            changed = True
        return changed

    def check_for_solved(self):
        if np.min(self.field_solution) == 0:
            return False
        else:
            return True

    def check_for_error(self):
        found_error = False
        for i in range(0, 9):
            row = self.field_solution[i, :]
            to_check = row[row > 0]
            if len(set(to_check)) < len(to_check):
                print('Found error in row {}'.format(i))
                found_error = True
        for j in range(0, 9):
            column = self.field_solution[:, j]
            to_check = column[column > 0]
            if len(set(to_check)) < len(to_check):
                print('Found error in column {}'.format(j))
                found_error = True
        for box_i in range(0, 3):
            for box_j in range(0, 3):
                box = self.field_solution[box_i * 3:box_i * 3 + 3, box_j * 3:box_j * 3 + 3]
                to_check = box[box > 0].flatten()
                if len(set(to_check)) < len(to_check):
                    print('Found error in box ({}, {})'.format(box_i, box_j))
                    found_error = True
        return found_error

    def solve_sudoku(self):
        print('Searching for a solution of:')
        self.show_field()
        changed_possible = True
        changed_solution = True
        iter_nb = 0
        while changed_possible or changed_solution:
            changed_possible = self.update_field_possible()
            changed_solution = self.update_field_solution()
            if self.check_for_error():
                self.show_field()
                print('Found error in iteraion {}'.format(iter_nb))
                return False
            iter_nb += 1
        if self.check_for_solved():
            print('Sudoku solved after {} iterations:'.format(iter_nb))
            self.show_field()
            return True
        else:
            print('Solver stuck at:')
            self.show_possibilities()
            self.show_field()
            return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sudoku = Sudoku()
    sudoku.read_field_from_csv('Sudoku_Examples/Example1')
    sudoku.solve_sudoku()

