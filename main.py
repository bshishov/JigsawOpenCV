from typing import List

import cv2 as cv
import numpy as np

from scipy.optimize import differential_evolution, basinhopping, minimize

from pieces_utils import gen_piece, gen_piece_lite, gen_piece_options, get_options_number, NUM_PARAMS, NUM_PARAMS_LITE, NUM_PARAMS_CORNERS
from utils import benchmark, center_contour, iter_curve_sides, resample_curve, curve_corners
from cv_tools import MeanFrameOverTime, ColoredMarkerDetector, InversePerspective, PieceDetector

# ['size', 'rotation', 'neck_heights', 'corner_offset', 'middle_offset', 'radius', 'ellipse']
OPTIONS = ['size', 'rotation']

VIS_SHAPE_FIT = False
VIS_LITE_SHAPE_FIT = False
VIS_OPTIONS_FIT = False


def contour_at_canvas_center(contour, canvas_size=(256, 256), dst=None):
    """ Draws contour at a center of a canvas """
    dst = dst or np.zeros(canvas_size, dtype=np.uint8)

    # Position contour at the center of canvas
    contour = center_contour(contour) + np.float32(canvas_size) * 0.5

    cv.fillPoly(dst, pts=[np.int0(contour)], color=(255, 255, 255), lineType=cv.LINE_AA)
    return dst


class Optimization:
    def __init__(self, target):
        self.target = self.normalize(target)

    def normalize(self, img):
        img = np.float32(img)
        return img / np.max(img)

    def distance(self, i1, i2):
        return np.sum(np.logical_xor(i1, i2)) / i1.size
        #return np.sqrt(np.sum((i1 - i2) ** 2)) / i1.size

    def minimize(self, args) -> float:
        polygon, _ = gen_piece(args)
        sample = contour_at_canvas_center(polygon)
        if VIS_SHAPE_FIT:
            cv.imshow('fit', sample)
            cv.waitKey(1)

        return self.distance(self.normalize(sample), self.target)

    def minimize_lite(self, args) -> float:
        polygon, _ = gen_piece_lite(args)
        sample = contour_at_canvas_center(polygon)
        if VIS_SHAPE_FIT:
            cv.imshow('shape-fit', sample)
            cv.waitKey(1)

        return self.distance(self.normalize(sample), self.target)

    def minimize_options(self, args) -> float:
        polygon, _ = gen_piece_options(args, OPTIONS)
        sample = contour_at_canvas_center(polygon)
        if VIS_OPTIONS_FIT:
            cv.imshow('options-fit', sample)
            cv.waitKey(1)

        return self.distance(self.normalize(sample), self.target)


def find_contour_corners(piece_contour):
    piece_contour = np.squeeze(np.asarray(piece_contour, dtype=np.float32))
    return curve_corners(piece_contour)


def find_contour_corners_opt(piece_contour):
    piece_contour = np.squeeze(np.asarray(piece_contour, dtype=np.float32))
    piece_center = np.squeeze(np.mean(piece_contour, axis=0))

    target_canvas = contour_at_canvas_center(piece_contour)

    opt = Optimization(target_canvas)

    with benchmark('Evolution'):
        opt_res = differential_evolution(opt.minimize_options,
                                         bounds=[(-1, 1)] * get_options_number(OPTIONS),
                                         tol=0.05)

    generated, corners = gen_piece_options(opt_res.x, OPTIONS)
    generated_offset = np.squeeze(np.mean(generated, axis=0))
    return np.float32(corners) - generated_offset + piece_center


def find_piece_sides(piece_contour, debug=False):
    piece_contour = np.squeeze(np.asarray(piece_contour, dtype=np.float32)) * 6
    with benchmark('Corners'):
        corners = find_contour_corners(piece_contour)
    sides = list(map(resample_curve, iter_curve_sides(piece_contour, corners)))
    #sides = list(iter_curve_sides(piece_contour, corners))

    if debug:
        w, h = 1024, 1024
        canvas_center = np.float32((w, h)) * 0.5
        piece_center = np.squeeze(np.mean(piece_contour, axis=0))
        dst = np.zeros((w, h, 3), np.uint8)

        for corner in corners:
            cv.drawMarker(dst, tuple(np.int0(corner + canvas_center - piece_center)), (255, 0, 0))

        colors = [
            (255, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ]
        for side, color in zip(sides, colors):
            if side.shape[0] > 2:
                cv.polylines(dst,
                             #[np.int0(resample_curve(side) + np.float32([512, 512] - piece_center))],
                             [np.int0(side + canvas_center - piece_center)],
                             isClosed=False,
                             color=color,
                             thickness=1)

        cv.namedWindow('find_piece_sides')
        cv.imshow('find_piece_sides', dst)
        cv.waitKey(0)

    return sides


def piece_sides_test(filename='sample_piece.png'):
    sample = cv.imread(filename)

    sample_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    sample_in_range = cv.inRange(sample_hsv, (0, 0, 220), (10, 10, 255))

    if cv.__version__ == '3.4.4':
        _, contours, _ = cv.findContours(sample_in_range,
                                         mode=cv.RETR_EXTERNAL,
                                         method=cv.CHAIN_APPROX_TC89_KCOS)
    else:
        contours, _ = cv.findContours(sample_in_range,
                                      mode=cv.RETR_EXTERNAL,
                                      method=cv.CHAIN_APPROX_TC89_KCOS)
    target_contour = contours[0]
    cv.imshow('Piece', contour_at_canvas_center(target_contour))
    cv.waitKey(1)
    find_piece_sides(target_contour, debug=True)

    cv.destroyAllWindows()


def detection():
    denoiser = MeanFrameOverTime()
    markers_detector = ColoredMarkerDetector(window_name='Markers')
    perspective = InversePerspective(window_name='Perspective')
    pieces_detector = PieceDetector('Pieces')

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not cap.isOpened():
        print('Error opening')
        return

    #cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

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
    piece_sides_test()
    #detection()


if __name__ == '__main__':
    main()

