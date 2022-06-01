# Sudoku
Scripts to read a sudoku from an image, detect the filled in numbers and complete the sudoku.

### SudokuReader
Class for reading a sudoku from a file. The largest quadrilateral contour is chosen as sudoku. In this contour, connected components are searched for. If the connected component satisfies certain shape criteria, it is assumed to be a digit. The digit is then classified using a neural network. 

### SudokuSolver
Class for solving a sudoku. A backtrack algorithm is used.

### NumberClassifier
A script for training a CNN for number classification. We use the pre-trained CNN `model-OCR.h5`. It can be used using the tensorflow/keras library:

```
import tensorflow as tf
my_model = tf.keras.models.load_model('model-OCR.h5')
```

### Usage
To use the script, use the main.py script. That is define a path to a sudoku image and execute the `read_and_solve_sudoku` function:

```
path = 'path/to/sudoku'
read_and_solve_sudoku(path)
```
