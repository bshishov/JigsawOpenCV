from typing import List

import cv2 as cv
import numpy as np


from pieces_utils import gen_piece, gen_piece_lite, NUM_PARAMS, NUM_PARAMS_LITE
from utils import arrange4points, benchmark
from cv_tools import MeanFrameOverTime, ColoredMarkerDetector, InversePerspective, PieceDetector


def align(contour, canvas_size=(512, 512), dst=None):
    """ Draws contour at a center of a canvas """
    dst = dst or np.zeros(canvas_size, dtype=np.uint8)

    contour = np.float32(contour)
    center = np.squeeze(np.mean(contour, axis=0))

    # Position contour at the center of canvas
    contour += np.float32(canvas_size) * 0.5 - center

    cv.fillPoly(dst, pts=[np.int0(contour)], color=(255, 255, 255), lineType=cv.LINE_AA)
    return dst


class Optimization:
    def __init__(self, target):
        self.target = target

    def norm(self, img):
        img = np.float32(img)
        return img / np.max(img)

    def dst1(self, i1, i2):
        i1 = self.norm(i1)
        i2 = self.norm(i2)

        return np.sum(np.logical_xor(i1, i2)) / i1.size

        #return np.sqrt(np.sum((i1 - i2) ** 2)) / i1.size
        #return np.sum(np.abs(i1 - i2))
        #dist_manhattan = np.sum(np.abs(i1 - i2)) / i1.size
        #return np.sqrt(np.sum((i1 - i2) ^ 2)) / i1.size

    def minimize(self, args) -> float:
        polygon = gen_piece(args)
        sample = align(polygon)
        cv.imshow('align1', sample)
        cv.waitKey(1)

        return self.dst1(sample, self.target)

    def minimize_lite(self, args) -> float:
        polygon = gen_piece_lite(args)
        sample = align(polygon)
        cv.imshow('align1', sample)
        cv.waitKey(1)

        return self.dst1(sample, self.target)

    def minimize_lite_kw(self, x, y, r1, r2, r3, r4, rot):
        polygon = gen_piece_lite([
            x, y, r1, r2, r3, r4, rot
        ])
        sample = align(polygon)
        cv.imshow('align1', sample)
        cv.waitKey(1)

        return 10.-self.dst1(sample, self.target)

    def plot(self, args):
        polygon = gen_piece(args)
        cv.imshow('align1', align(polygon))


def piece_generation_test(filename='sample_piece.png'):
    wnd_name = 'Frame'
    cv.namedWindow(wnd_name)
    cv.namedWindow('align1')
    cv.namedWindow('align2')

    sample = cv.imread(filename)

    sample_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    sample_in_range = cv.inRange(sample_hsv, (0, 0, 220), (10, 10, 255))

    contours, hierarchy = cv.findContours(sample_in_range,
                                          mode=cv.RETR_EXTERNAL,
                                          method=cv.CHAIN_APPROX_SIMPLE)
    sample_contour = contours[0]
    target = align(sample_contour)
    cv.imshow('align2', target)

    from scipy.optimize import differential_evolution, basinhopping, minimize

    while True:
        opt = Optimization(target)

        with benchmark('Evolution'):
            opt_res = differential_evolution(opt.minimize_lite,
                                             bounds=[(-1, 1)] * NUM_PARAMS_LITE,
                                             tol=0.05)
            print(opt_res.message)
            cv.imshow('align1', align(gen_piece_lite(opt_res.x)))

        cv.waitKey(1)

        lite_args = opt_res.x

        # Default bounds
        bounds = np.zeros((NUM_PARAMS, 2), dtype=np.float32)
        bounds[:, 0] = -1
        bounds[:, 1] = +1

        delta = 0.05

        # Size
        bounds[0] = [lite_args[0] - delta, lite_args[0] + delta]
        bounds[1] = [lite_args[1] - delta, lite_args[1] + delta]

        # Rotation
        bounds[2] = [lite_args[2] - delta, lite_args[2] + delta]

        # Radius constraints
        bounds[27] = [lite_args[3] - delta, lite_args[3] + delta]
        bounds[28] = [lite_args[4] - delta, lite_args[4] + delta]
        bounds[29] = [lite_args[5] - delta, lite_args[5] + delta]
        bounds[30] = [lite_args[6] - delta, lite_args[6] + delta]

        with benchmark('Second pass'):
            opt_res = differential_evolution(opt.minimize, bounds=list(bounds), tol=0.05)
            #opt_res = differential_evolution(opt.minimize, bounds=[(-1, 1)] * 23)
            polygon = gen_piece(opt_res.x)

        cv.imshow('align1', align(polygon))
        cv.waitKey(0)


def detection():
    denoiser = MeanFrameOverTime()
    markers_detector = ColoredMarkerDetector(window_name='Markers')
    perspective = InversePerspective(window_name='Perspective')
    pieces_detector = PieceDetector('Pieces')

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print('Error opening')
        return

    # cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    cv.namedWindow('Capture')

    while cap.isOpened():
        ret, raw = cap.read()
        if ret:
            raw = denoiser.process(raw)
            markers = markers_detector.find_markers(raw)
            # markers_detector.draw_markers(raw, markers)
            board = perspective.process(raw, markers)
            pieces_detector.find_pieces(board)

            cv.imshow('Capture', raw)

            # Press Q on keyboard to exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


def main():
    piece_generation_test()
    #detection()


if __name__ == '__main__':
    main()

