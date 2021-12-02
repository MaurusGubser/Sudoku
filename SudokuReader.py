import numpy as np
import matplotlib
import cv2 as cv

import matplotlib.pyplot as plt

matplotlib.use('tkagg')


class SudokuReader():

    def __init__(self):
        self.solution_field = np.zeros((9, 9), dtype=np.uint8)
        self.input_image = np.empty((1, 1, 3), dtype=np.uint8)
        self.input_edges = np.empty((1, 1), dtype=np.uint8)
        self.sudoku_img = np.empty((1, 1), dtype=np.uint8)
        self.sudoku_gray = np.empty((1, 1), dtype=np.uint8)
        self.sudoku_binary = np.empty((1, 1), dtype=np.uint8)

        self.img_sudoku_gray = np.empty((1, 1), dtype=np.uint8)
        self.lines = np.empty((1, 1), dtype=np.uint8)
        self.height_img = 0  # row
        self.width_img = 0  # column
        self.x0_sudoku = 0
        self.y0_sudoku = 0
        self.height_sudoku = 0  # row
        self.width_sudoku = 0   # column
        self.number_candidates = []

    def read_image_from_source(self, path_src):
        self.input_image = cv.imread(path_src)
        long_side = max(self.input_image.shape[0], self.input_image.shape[1])
        scale_factor = 800 / long_side
        self.input_image = cv.resize(self.input_image, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
        self.height_img, self.width_img = self.input_image.shape[0], self.input_image.shape[1]
        return None

    def show_all_images(self):
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0, 0].imshow(self.input_image)
        axs[0, 0].set_title('Input image')
        axs[0, 1].imshow(self.input_edges)
        axs[0, 1].set_title('Edge image')
        axs[1, 0].imshow(self.sudoku_gray, cmap='gray')
        axs[1, 0].set_title('Sudoku gray')
        axs[1, 1].imshow(self.sudoku_binary, cmap='gray')
        axs[1, 1].set_title(label='Sudoku binary')
        fig.suptitle('All (relevant) images')
        plt.show()
        return None

    def show_all_images_opencv(self):
        cv.imshow('Input image', self.input_image)
        cv.imshow('Input edges', self.input_edges)
        cv.imshow('Sudoku image gray', self.sudoku_gray)
        cv.imshow('Sudoku image gray', self.sudoku_binary)
        cv.waitKey(0)
        return None

    def show_edge_image(self):
        fig, axs = plt.subplots(nrows=1, ncols=2)
        input_gray = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)
        axs[0].imshow(input_gray, cmap='gray')
        axs[1].imshow(self.input_edges, cmap='gray')
        plt.show()
        return None

    def show_hough_line(self):
        line_img = self.input_image.copy()
        len_x, len_y, _ = self.input_image.shape
        long_side = max(len_x, len_y)
        for line in self.lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + long_side * (-b))
            y1 = int(y0 + long_side * (a))
            x2 = int(x0 - long_side * (-b))
            y2 = int(y0 - long_side * (a))
            cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        plt.imshow(line_img)
        plt.show()
        return None

    def save_images(self, path_dest):
        cv.imwrite(path_dest, self.input_image)
        cv.imwrite(path_dest + '_sudoku', self.sudoku_img)
        cv.imwrite(path_dest + '_gray', self.sudoku_gray)
        cv.imwrite(path_dest + '_binary', self.sudoku_binary)
        return None

    def compute_binary_image(self, gaussian_kernel_size=5, thres=1.0, block_size=5):
        blur = cv.GaussianBlur(self.sudoku_gray, (gaussian_kernel_size, gaussian_kernel_size), sigmaX=3.0)
        self.sudoku_binary = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
                                                  blockSize=block_size, C=thres)
        return None

    def otsu_thresholding(self, kernel_size=7):
        blur = cv.GaussianBlur(self.sudoku_gray, (kernel_size, kernel_size), 0)
        _, self.sudoku_binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return None

    def open_image(self, kernel_shape=cv.MORPH_RECT, kernel_size=3):
        kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        self.sudoku_binary = cv.morphologyEx(self.sudoku_binary, cv.MORPH_OPEN, kernel)
        return None

    def close_image(self, kernel_shape=cv.MORPH_RECT, kernel_size=5):
        kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        self.sudoku_binary = cv.morphologyEx(self.sudoku_binary, cv.MORPH_CLOSE, kernel)
        return None

    def canny_edge_detection(self, kernel_size=5, thres_low=100, thres_upper=200):
        input_gray = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)
        input_blur = cv.GaussianBlur(input_gray, ksize=(kernel_size, kernel_size), sigmaX=1.0)
        self.input_edges = cv.Canny(input_blur, thres_low, thres_upper)
        return None

    def hough_line_detection(self, rho=1, theta=np.pi / 180, thres=200):
        self.lines = cv.HoughLines(self.input_edges, rho=rho, theta=theta, threshold=thres)
        return None

    def watershed_detection(self, kernel_size=5, mask_size=5):
        # noise removal
        gray = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opening = cv.morphologyEx(thresh, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
        # sure background area
        sure_bg = cv.dilate(opening, kernel=kernel, iterations=1)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening, distanceType=cv.DIST_L2, maskSize=mask_size)
        ret, sure_fg = cv.threshold(dist_transform, thresh=0.7 * dist_transform.max(), maxval=255, type=0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg, sure_fg)
        fig, axs = plt.subplots(nrows=2, ncols=3)
        axs[0, 0].imshow(self.input_image)
        axs[0, 1].imshow(thresh)
        axs[0, 2].imshow(sure_bg)
        axs[1, 0].imshow(dist_transform)
        axs[1, 1].imshow(sure_fg)
        axs[1, 2].imshow(unknown)
        plt.show()
        return None

    def find_contour_sudoku(self):
        # self.otsu_thresholding()
        self.canny_edge_detection()
        contours, _ = cv.findContours(self.input_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        areas_candidates = np.array([cv.contourArea(contours[i]) for i in range(0, len(contours))])
        idx_sudoku = np.argmax(areas_candidates)
        poly_sudoku = cv.approxPolyDP(contours[idx_sudoku], epsilon=1.0, closed=True)
        y0, x0, h, w = [int(i) for i in cv.boundingRect(poly_sudoku)]
        self.x0_sudoku = x0
        self.y0_sudoku = y0
        self.height_sudoku = h
        self.width_sudoku = w
        self.sudoku_img = self.input_image[y0:y0 + h, x0:x0 + w]
        self.sudoku_gray = cv.cvtColor(self.sudoku_img, cv.COLOR_BGR2GRAY)
        return None

    def find_number_candidates(self):
        self.otsu_thresholding()
        self.close_image()
        thresh = self.sudoku_binary
        nb_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv.CV_32S)

        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(thresh, cmap='gray')
        axs[0].set_title('Binary image')
        axs[1].imshow(labels)
        axs[1].set_title('Connected components')
        plt.show()

        output = self.sudoku_img.copy()
        for i in range(0, nb_labels):
            if self.is_candidate_size_realistic(stats[i]):
                self.number_candidates.append({'y': stats[i, cv.CC_STAT_TOP],
                                               'x': stats[i, cv.CC_STAT_LEFT],
                                               'h': stats[i, cv.CC_STAT_HEIGHT],
                                               'w': stats[i, cv.CC_STAT_WIDTH],
                                               'y_center': centroids[i, 0],
                                               'x_center': centroids[i, 1]})
            else:
                continue
        """
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            cx = centroids[i, 1]
            cy = centroids[i, 0]
            cand_img = self.crop_candidate(stats[i])
            cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.circle(output, (int(cy), int(cx)), 1, (255, 0, 0), 3)

        plt.imshow(output)
        plt.title('Candidates found')
        plt.show()
        """
        return None

    def harris_corner(self, block_size=2, ksize=3, k=0.15):
        self.sudoku_gray = np.float32(self.sudoku_gray)
        dst = cv.cornerHarris(self.sudoku_gray, blockSize=block_size, ksize=ksize, k=k)
        dst = cv.dilate(dst, None)
        self.input_image[dst > 0.01 * dst.max()] = [0, 0, 255]
        return None

    def is_candidate_size_realistic(self, stats):
        # assuming area_total \approx A_sudoku = s**2
        # assuming 1/10 * 1/81 * s**2 <= A_cand <= 1/81 * s**2
        # s being the side length of the sudoku square
        area_total = self.height_sudoku * self.width_sudoku
        w = stats[cv.CC_STAT_WIDTH]
        h = stats[cv.CC_STAT_HEIGHT]
        area_cand = w * h
        if h / w < 1.0 / 3.0 or 3.0 < h / w:
            return False
        elif area_cand / area_total < 0.0012 or 0.012 < area_cand / area_total:
            return False
        else:
            return True

    def crop_candidate(self, stats):
        x = stats[cv.CC_STAT_LEFT]
        y = stats[cv.CC_STAT_TOP]
        w = stats[cv.CC_STAT_WIDTH]
        h = stats[cv.CC_STAT_HEIGHT]
        delta_h = h // 5
        delta_w = w // 5

        img_cand = self.sudoku_gray[y - delta_h:y + delta_h + h, x - delta_w:x + delta_w + w]
        img_cand = cv.resize(img_cand, dsize=(28, 28))

        return img_cand
