import pytesseract
import numpy as np
import os
import cv2


class ReadTableImage:

    def __init__(self, image_path: str, tesseract_path=None, show=False):
        self.image_path = image_path
        self.tesseract_path = tesseract_path
        self.show = show
        try:
            self.img = cv2.imread(cv2.samples.findFile(self.image_path))
            self.cImage = np.copy(self.img)
            self.img_height, self.img_width, self.img_channels = self.img.shape
        except cv2.error:
            raise cv2.error('Incorrect path to image')
        self.lines = None
        self.chart_lines = []
        self.horizontal_lines = []
        self.vertical_lines = []
        self.results = []

    def pre_process_image(self) -> None:
        self.dilate_img()
        self.gaussian_blur()
        self.grayscale()
        self.canny_img()
        self.morph_img()

    def auto_resize_image(self):
        scalar_options = [int(6000/self.img_height), int(6000/self.img_width)]
        scalar = min(scalar_options)
        if scalar <= 0:
            scalar = 1
        self.img = cv2.resize(self.img, (self.img_width * scalar, self.img_height * scalar))
        self.cImage = self.img
        self.img_height, self.img_width, self.img_channels = self.img.shape
        if self.show:
            cv2.imshow("resize", self.img)
            cv2.waitKey(0)
            cv2.destroyWindow("resize")

    def dilate_img(self) -> None:
        # dilated
        k_size = 3
        kernel = np.ones((k_size, k_size), np.uint8)
        src = self.img
        img_dilated = cv2.dilate(src, kernel, iterations=1)
        self.img = img_dilated
        if self.show:
            cv2.imshow("dilate", self.img)
            cv2.waitKey(0)
            cv2.destroyWindow("dilate")

    def gaussian_blur(self) -> None:
        # gaussian blur
        img_blurred = cv2.GaussianBlur(self.img, (0, 0), 3)
        self.img = img_blurred
        if self.show:
            cv2.imshow("blur", self.img)
            cv2.waitKey(0)
            cv2.destroyWindow("blur")

    def grayscale(self) -> None:
        # grayscale
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img = img_gray
        if self.show:
            cv2.imshow("gray", self.img)
            cv2.waitKey(0)
            cv2.destroyWindow("gray")

    def canny_img(self) -> None:
        # canny alteration
        img_canny = cv2.Canny(self.img, 50, 150)
        self.img = img_canny
        if self.show:
            cv2.imshow("canny", self.img)
            cv2.waitKey(0)
            cv2.destroyWindow("canny")

    def morph_img(self) -> None:
        # morph image
        kernel = np.ones((5, 5), np.uint8)
        img_morph = cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel, iterations=2)
        self.img = img_morph
        if self.show:
            cv2.imshow("canny", self.img)
            cv2.waitKey(0)
            cv2.destroyWindow("canny")

    def create_lines(self, rho=1, threshold=50, min_line_length=200, theta=(np.pi/180), max_line_gap=20) -> None:

        self.lines = cv2.HoughLinesP(self.img, rho, theta, threshold, None, min_line_length, max_line_gap)
        img_line = np.copy(self.cImage)

        if self.lines is not None:
            for i in range(0, len(self.lines)):
                ln = self.lines[i][0]
                self.chart_lines.append(ln)

        for i, line in enumerate(self.chart_lines):
            cv2.line(img_line, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 3, cv2.LINE_AA)

        if self.show:
            cv2.imshow("with_line", img_line)
            cv2.waitKey(0)
            cv2.destroyWindow("with_line")

    @staticmethod
    def __is_vertical(line, vert_tolerance: int) -> bool:
        diff = line[0] - line[2]
        if diff in range(int((vert_tolerance*-1)/2), int(vert_tolerance/2)):
            return True
        else:
            return False

    @staticmethod
    def __is_horizontal(line, horizontal_tolerance: int) -> bool:
        diff = line[1] - line[3]
        if diff in range(int((horizontal_tolerance*-1)/2), int(horizontal_tolerance/2)):
            return True
        else:
            return False

    def __combine_horizontal_lines(self, lst: list) -> list:
        min_x = self.img_width
        min_y = self.img_height
        max_x = 0
        max_y = 0

        for line in lst:
            if line[0] < min_x:
                min_x = line[0]
                min_y = line[1]
            if line[2] > max_x:
                max_x = line[2]
                max_y = line[3]

        return [min_x, min_y, max_x, max_y]

    def __combine_vertical_lines(self, lst: list) -> list:
        min_x = self.img_width
        min_y = self.img_height
        max_x = 0
        max_y = 0

        for line in lst:
            if line[1] < min_y:
                min_x = line[0]
                min_y = line[1]
            if line[3] > max_y:
                max_x = line[2]
                max_y = line[3]

        return [min_x, min_y, max_x, max_y]

    def __overlapping_filter(self, lines: list, sorting_index: int, overlapping_tolerance=None) -> list:
        filtered_lines = []
        combined_lines = []
        line_dict = {}

        lines = sorted(lines, key=lambda x: x[sorting_index])
        separation = overlapping_tolerance
        for i in range(len(lines)):
            l_curr = lines[i]
            if i > 0:
                l_prev = lines[i - 1]

                if (l_curr[sorting_index] - l_prev[sorting_index]) > separation:
                    filtered_lines.append(l_curr)
                    line_dict.update({len(filtered_lines): [l_curr]})

                else:
                    array = line_dict.get(len(filtered_lines))
                    array.append(list(l_curr))
            else:
                filtered_lines.append(l_curr)
                line_dict.update({len(filtered_lines): [l_curr]})

        for key, value in line_dict.items():
            if sorting_index == 1:
                comb = self.__combine_horizontal_lines(value)
            else:
                comb = self.__combine_vertical_lines(value)
            combined_lines.append(np.asarray(comb))

        return combined_lines

    def merge_lines(self, vert_tolerance=100, horizontal_tolerance=100, overlap_tolerance=50) -> None:
        merge_lines = np.copy(self.cImage)
        if self.lines is not None:
            for i in range(0, len(self.lines)):
                ln = self.lines[i][0]
                if self.__is_vertical(ln, vert_tolerance):
                    self.vertical_lines.append(ln)
                elif self.__is_horizontal(ln, horizontal_tolerance):
                    self.horizontal_lines.append(ln)
            self.horizontal_lines = self.__overlapping_filter(
                lines=self.horizontal_lines,
                sorting_index=1,
                overlapping_tolerance=overlap_tolerance
            )
            self.vertical_lines = self.__overlapping_filter(
                lines=self.vertical_lines,
                sorting_index=0,
                overlapping_tolerance=overlap_tolerance
            )

        for i, line in enumerate(self.horizontal_lines):
            cv2.line(merge_lines, (line[0], line[1]), (line[2], line[3]), (0, 255, 20), 3, cv2.LINE_AA)
            cv2.putText(merge_lines, str(i) + "h", (line[0] + 5, line[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)

        for i, line in enumerate(self.vertical_lines):
            cv2.line(merge_lines, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(merge_lines, str(i) + "v", (line[0], line[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)
        if self.show:
            cv2.imshow("with_map", merge_lines)
            cv2.waitKey(0)
            cv2.destroyWindow("with_map")

    def create_merge_lines(self, vert_tolerance=100, horizontal_tolerance=100, overlap_tolerance=50) -> None:
        self.create_lines()
        self.merge_lines(
            vert_tolerance=vert_tolerance,
            horizontal_tolerance=horizontal_tolerance,
            overlap_tolerance=overlap_tolerance
        )

    @staticmethod
    def __read_box(img) -> str:

        kernel = np.ones((3, 3), np.uint8)
        img_dilated = cv2.dilate(img, kernel, iterations=1)
        img_blurred = cv2.GaussianBlur(img_dilated, (0, 0), 3)
        gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        result = 255 - close
        txt = pytesseract.image_to_string(result, lang='eng', config='--psm 12 --oem 3').strip()

        return txt

    def read_chart(self) -> list:

        # tesseract path
        if self.tesseract_path is not None:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        for i in range(len(self.horizontal_lines) - 1):
            y_up = self.horizontal_lines[i][1]
            y_down = self.horizontal_lines[i + 1][1]

            lst = []
            if len(self.vertical_lines) == 0:
                x_left = 0 + int(0.1*self.img_width)
                x_right = self.img_width

                try:

                    cropped_img = self.cImage[y_up:y_down, x_left:x_right]
                    if self.show:
                        cv2.imshow("crop", cropped_img)
                        cv2.waitKey(0)
                        cv2.destroyWindow("crop")
                except AttributeError:
                    raise Exception(
                        'Please preprocess the image before creating lines. use the pre_process_image() function')

                result = self.__read_box(cropped_img)
                lst.append(result)

            else:
                for j in range(len(self.vertical_lines) - 1):
                    x_left = self.vertical_lines[j][0]
                    x_right = self.vertical_lines[j + 1][0]

                    try:

                        cropped_img = self.cImage[y_up:y_down, x_left:x_right]
                        if self.show:
                            cv2.imshow("crop", cropped_img)
                            cv2.waitKey(0)
                            cv2.destroyWindow("crop")
                    except AttributeError:
                        raise Exception('Please preprocess the image before creating lines. use the pre_process_image() function')

                    result = self.__read_box(cropped_img)
                    lst.append(result)

            self.results.append(lst)

        return self.results

    def create_csv(self):
        dir_path = os.path.dirname(__file__)
        csv_results = os.path.join(dir_path, 'results.csv')
        with open(csv_results, 'w') as file:
            for line in self.results:
                row = ''
                for item in line:
                    row += str(item) + ','
                file.write(row + '\n')




