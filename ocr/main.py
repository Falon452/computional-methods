import matplotlib.pyplot as plt
import cv2
import numpy as np


class OCR:
    def __init__(self, img_path, font='SegoeUI'):
        self.letters = ['g', 'z', 'x', 'b', 'm', 'h', 'd', 'y', 'v', 'a', 's', 'p', 'q', 'o', 'e', 'c', 'f', 'n', 'j',
                        'k', 'l',
                        'w', 'r', 't', 'u', 'i']
        self.numbers = []
        self.special = []
        self.scores = {}
        self.img = 255 - self.read_img(img_path)
        # self.img = self.get_img_approx(self.img, 40)
        self.font_path = "Fonts/" + font + "/"
        self.pattern_imgs = self.get_pattern_imgs()
        self.pattern_positions = {}

    def run_ocr(self):
        self.get_max_score_for_each_pattern()
        # self.show_img(self.img)
        for p in self.pattern_imgs.keys():
            self.single_pattern(p, 0.9, 1.1)
        result = self.convert_to_text()

        for pattern_key, positions in self.pattern_positions.items():
            print(f"{pattern_key} detected {len(positions)} times")
        print('\n' + result, end='\n')
        return result

    def read_img(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def get_pattern_imgs(self):
        pattern_imgs = {}
        # numbers = [str(i) for i in range(10)]
        # special = ['dot', 'comma', 'exclamation', 'question']

        for l in self.letters:
            pattern_imgs[l] = 255 - self.read_img(self.font_path + l + '.PNG')
        for n in self.numbers:
            pattern_imgs[n] = 255 - self.read_img(self.font_path + str(n) + '.PNG')
        for s in self.special:
            pattern_imgs[s] = 255 - self.read_img(self.font_path + s + '.PNG')

        return pattern_imgs

    def get_pattern_corr(self, pattern, thresh, upper_thresh, key):
        x = np.fft.fft2(self.img)
        y = np.fft.fft2(np.rot90(pattern, 2), x.shape)
        res = np.multiply(x, y)
        corr = np.abs(np.fft.ifft2(res)).astype(float)

        corr[corr < thresh * self.scores[key]] = 0
        corr[corr > upper_thresh * self.scores[key]] = 0
        corr[corr != 0] = 255
        return corr

    def get_pattern_positions(self, pattern, corr):
        positions = []

        w, h = pattern.shape
        x_prev, y_prev = -100, -100
        for (i, j), v in np.ndenumerate(corr):
            if corr[i][j] == 255 and (x_prev + w < i or y_prev + h < j):
                positions.append((i, j))
                x_prev, y_prev = i, j

        return positions

    def single_pattern(self, pattern_key, thresh, upper_thresh):
        pattern = self.pattern_imgs[pattern_key]
        pattern_corr = self.get_pattern_corr(pattern, thresh, upper_thresh, pattern_key)

        self.pattern_positions[pattern_key] = self.get_pattern_positions(pattern, pattern_corr)

        # print(pattern_key)
        # print(self.pattern_positions)
        # self.show_img(pattern_corr)
        # self.show_img(np.rot90(pattern_corr,2))

        for x, y in self.pattern_positions[pattern_key]:
            w, h = pattern.shape[:2]
            self.img[x - w: x, y - h: y] = 0

        # self.show_img(self.img)

    def convert_to_text(self):
        positions = []
        for key, value in self.pattern_positions.items():
            for val in value:
                positions.append((key, val[1], val[0]))
        positions_y = sorted(positions, key=lambda e: (e[2], e[1]))

        line_height = 60

        lines = []
        lowest_y = positions_y[0][2]
        lines.append(([], lowest_y))

        current_line = 0
        for pos in positions_y:
            letter, x, y = pos
            if y > lines[current_line][1] + line_height:
                lines.append(([], y))
                current_line += 1

            lines[current_line][0].append((letter, x))

        for i in range(len(lines)):
            lines[i][0].sort(key=lambda e: e[1])

        result = ""
        for line in lines:
            line_text = ""
            for i in range(len(line[0])):
                if i != 0:
                    prev_letter, prev_x = line[0][i - 1]
                    letter, x = line[0][i]
                    if x - prev_x > 55:
                        line_text += " "
                else:
                    letter, x = line[0][i]
                line_text += letter

            result += line_text
            result += '\n'

        return result

    def show_img(self, img):
        plt.imshow(img, cmap='gray')
        plt.show()

    def get_max_score_for_each_pattern(self):
        scores = {}
        for l in self.letters:
            pattern = self.pattern_imgs[l]
            x = np.fft.fft2(pattern)
            y = np.fft.fft2(np.rot90(pattern, 2), x.shape)
            res = np.multiply(x, y)
            corr = np.abs(np.fft.ifft2(res)).astype(float)
            scores[l] = np.max(corr)
        self.scores = scores

    def deskew(self, img):
        angle = self._getSkewAngle(img)
        return self._rotateImage(img, -1.0 * angle)

    def _getSkewAngle(self, img):
        blur = cv2.GaussianBlur(img, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=5)

        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        largest_contour = contours[0]
        min_area_react = cv2.minAreaRect(largest_contour)

        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = min_area_react[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    def _rotateImage(self, img, angle):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return img

    def get_img_approx(self, img, k):
        U, s, V = np.linalg.svd(img)
        S = np.zeros((img.shape[0], img.shape[1]))
        S[:img.shape[0], :img.shape[0]] = np.diag(s)
        n_component = k
        S = S[:, :n_component]
        VT = V[:n_component, :]
        A = U.dot(S.dot(VT))
        return A


"""
wnioski
c zamazalo b  
v zamazalo n i m
c zamazalo n
b zamzało g
g poprawnie działa dla rotacji o 180 stopni
n zamazało h
c zamzało d
v zamazało y
c zamazało o i trochę p
"""
if __name__ == '__main__':
    mutilinerotated = 'Fonts/SegoeUI/mutilinerotated.PNG'
    oneline_rotated = 'Fonts/SegoeUI/oneline_rotated.PNG'
    oneline_rotated_corr = 'this is one line text'
    special_chars = 'Fonts/SegoeUI/special_chars.PNG'
    with_numbers = 'Fonts/SegoeUI/with_numbers.PNG'
    test_file = 'Fonts/SegoeUI/test_file.PNG'
    all_letters = 'Fonts/SegoeUI/all_letters.PNG'
    all_letters_cutted = 'Fonts/SegoeUI/all_letters_cutted.PNG'
    c_letter = 'Fonts/SegoeUI/c.PNG'
    qwerty = 'Fonts/SegoeUI/qwertyuiop.PNG'
    zxcvbnmasdfghjkl = 'Fonts/SegoeUI/zxcvbnmasdfghjkl.PNG'
    all_oneline = 'Fonts/SegoeUI/all_oneline.PNG'
    oneline_close = 'Fonts/SegoeUI/oneline_close.PNG'

    hello_mownit = 'Fonts/SegoeUI/hello_mownit.PNG'
    hello_mownit_ocr_test_multiline = 'Fonts/SegoeUI/hello_ocr_test_multiline.PNG'

    proceeding_img = hello_mownit_ocr_test_multiline

    ocr = OCR(proceeding_img)
    ocr.get_max_score_for_each_pattern()
    # deskew
    # img = ocr.read_img(proceeding_img)
    # ocr.show_img(img)

    # deskewd_img = ocr.deskew(img)

    # ocr.show_img(deskewd_img)
    #######
    ocr.run_ocr()
