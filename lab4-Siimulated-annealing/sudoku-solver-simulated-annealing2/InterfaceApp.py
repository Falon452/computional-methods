from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import Config
import cv2
import numpy as np
from random import randint

"""

Simulated annealing using arbitrary swap at each iteration

Current energy and temperature are printed on the console

--- installation
pip install opencv-python
--- and run main



---
At the beginning I fill missing points such that each square has unique numbers
Arbitrary swap works inside square
---

-----
It works randomly, sometimes it will find solution sometimes don't
-----

The preset values worked for me most of the times
"""


Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '800')

PATH = 'sudoku_images/sudoku2.txt'
width = 400
height = 400

TEMPERATURE = 1500
COOLING = 0.99
NOF_ITERATIONS_TO_COOL = 5
MAX_ITERATIONS = 50000


INITIAL_SUDOKU = np.array([['' for _ in range(9)] for _ in range(9)])
sudoku = np.array([['' for _ in range(9)] for _ in range(9)])


ITERATION = 0
ENERGY = float('inf')


class StartScreen(Screen):
    def start_program(self):
        self.manager.current = 'image'


class ImageScreen(Screen):
    def solve(self):
        self.img.source = self.path.text
        self.img.source = solve(PATH)

    def upload(self):
        global PATH
        PATH = self.path.text
        print(f"updated PATH {PATH}")

    def set_temperature(self):
        global TEMPERATURE
        TEMPERATURE = int(self.temp.text)
        print(f"updated TEMPERATURE {TEMPERATURE}")

    def set_cooling(self):
        global COOLING
        COOLING = float(self.cool.text)
        print(f"updated COOLING {COOLING}")

    def set_iterations_to_cool(self):
        global NOF_ITERATIONS_TO_COOL
        NOF_ITERATIONS_TO_COOL = int(self.iter_to_cool.text)
        print(f"updated NOF_ITERATIONS_TO_COOL{NOF_ITERATIONS_TO_COOL}")

    def set_max_iterations(self):
        global MAX_ITERATIONS
        MAX_ITERATIONS = int(self.max_iter.text)
        print(f"updated {PATH}")
        print(MAX_ITERATIONS)

    def reset(self):
        global sudoku, ITERATION, ENERGY, TEMPERATURE
        sudoku = np.array([['' for _ in range(9)] for _ in range(9)])
        ITERATION = 0
        ENERGY = float('inf')
        TEMPERATURE = 1500


class InterfaceApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(StartScreen(name='start'))
        sm.add_widget(ImageScreen(name='image'))

        return sm


def solve(filepath):
    global sudoku
    read_sudoku(filepath)
    fill_squares()
    for _ in range(MAX_ITERATIONS):
        iterate()
        if ENERGY == 0:
            return draw_sudoku(sudoku)

    return 'sudoku_clean.png'


def read_sudoku(path):
    global INITIAL_SUDOKU
    with open(path, 'r') as f:
        for i in range(9):
            line = f.readline()
            n = line.split(',')
            for j in range(9):
                INITIAL_SUDOKU[i][j] = n[j][0]


def iterate():
    global TEMPERATURE, ITERATION, ENERGY, sudoku
    new_sudoku = sudoku.copy()

    arbitrary_swap(new_sudoku)

    new_energy = energy(new_sudoku)
    if new_energy > ENERGY:
        if accept_worse(ENERGY, new_energy):
            sudoku = new_sudoku
            ENERGY = new_energy
    else:
        ENERGY = new_energy
        sudoku = new_sudoku

    print(f"({ITERATION}) Energy: {ENERGY}  Temperature: {TEMPERATURE}")
    if ITERATION % NOF_ITERATIONS_TO_COOL == 0:
        TEMPERATURE *= COOLING
    ITERATION += 1


def accept_worse(prev, next):
    if prev > next:
        print("prev cannot be higher than next value in accept_wrose")
        exit(1)

    probability =  1/ (1 + np.exp((next - prev) / TEMPERATURE))
    if np.random.ranf() < probability:
        return True
    return False


def energy(sudoku):
    total = 0
    for row in sudoku:
        dict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
        for number in row:
            dict[number] += 1
        for value in dict.values():
            total += max(value - 1, 0)

    for i in range(9):
        dict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
        for number_in_column in sudoku[:, i]:
            dict[number_in_column] += 1
        for value in dict.values():
            total += max(value - 1, 0)

    for k in range(3):
        for l in range(3):
            dict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
            for i in range(3):
                for j in range(3):
                    dict[sudoku[3 * k + i, 3 * l + j]] += 1
            for value in dict.values():
                total += max(value - 1, 0)

    return total


def fill_squares():
    global sudoku, INITIAL_SUDOKU
    sudoku = INITIAL_SUDOKU.copy()
    for k in range(3):
        for l in range(3):
            dict = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0}
            for i in range(3):
                for j in range(3):
                    if sudoku[3 * k + i, 3 * l + j] != 'x':
                        dict[sudoku[3 * k + i, 3 * l + j]] += 1

            for key, value in dict.items():
                if value > 1:
                    print("Invalid sudoku")
                    exit(1)

                break_above = False
                if value == 0:
                    for i in range(3):
                        if break_above:
                            break
                        for j in range(3):
                            if sudoku[3 * k + i, 3 * l + j] == 'x':
                                sudoku[3 * k + i, 3 * l + j] = key
                                break_above = True
                                break


def arbitrary_swap(sudoku):
    global INITIAL_SUDOKU
    i = randint(0, 8)
    j = randint(0, 8)
    while INITIAL_SUDOKU[i][j] != 'x':  # check if it was preset
        i = randint(0, 8)
        j = randint(0, 8)

    # find square

    if i < 3:
        k = 0
    elif i < 6:
        k = 1
    else:
        k = 2
    if j < 3:
        l = 0
    elif j < 6:
        l = 1
    else:
        l = 2

    # check if the square where i and j is not all preset
    total_preset = 0
    for m in range(3):
        for n in range(3):
            if INITIAL_SUDOKU[3*k + m, 3*l + n] == 'x':
                total_preset += 1
    if total_preset == 9:
        return

    # find other to swap

    v = randint(0, 2)
    u = randint(0, 2)
    while INITIAL_SUDOKU[3*k + v, 3*l + u] != 'x':
        v = randint(0, 2)
        u = randint(0, 2)

    sudoku[i][j], sudoku[3*k +v, 3*l +u] = sudoku[3*k +v, 3*l +u], sudoku[i][j]


def draw_sudoku(sudoku):
    sudoku_clean = cv2.imread("sudoku_clean.png")
    sudoku_clean = cv2.resize(sudoku_clean, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX
    step = width // 9
    fontScale = 1
    color = (0, 0, 0)
    thickness = 1
    for i in range(9):
        i_err = 0
        if i > 2:
            i_err = 2
        if i > 5:
            i_err = 4
        for j in range(9):
            pos = (14 + i * step + i_err, 32 + j * step + i_err)
            sudoku_clean = cv2.putText(sudoku_clean, str(sudoku[j][i]), pos, font,  # check j, i after
                                       fontScale, color, thickness, cv2.LINE_AA)

    filename = 'sudoku_solved.png'
    cv2.imwrite(filename, sudoku_clean)
    return filename


if __name__ == '__main__':
    InterfaceApp().run()