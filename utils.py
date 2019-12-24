import time
from contextlib import contextmanager

import numpy as np


def arrange4points(points: np.ndarray):
    """ Arranges array of 4 points into specific order"""
    if points.shape[0] != 4:
        return points
    min_p = np.min(points, 0)
    max_p = np.max(points, 0)

    a = np.float32([min_p[0], min_p[1]])
    b = np.float32([min_p[0], max_p[1]])
    c = np.float32([max_p[0], min_p[1]])
    d = np.float32([max_p[0], max_p[1]])

    def _nearest_p(pts, to):
        dist = np.sum((pts - to) ** 2, axis=1)
        return pts[np.argsort(dist)[0]]

    return np.float32([
        _nearest_p(points, a),
        _nearest_p(points, b),
        _nearest_p(points, c),
        _nearest_p(points, d),
    ])


@contextmanager
def benchmark(name: str):
    print(f'Started: {name}')
    start = time.time()

    try:
        yield None
    finally:
        end = time.time()
        print(f'Finished: {name} time: {end - start}')


def center_contour(contour):
    contour = np.float32(contour)
    return contour - np.squeeze(np.mean(contour, axis=0))


def index_of_nearest_to_point(curve, point):
    distances = np.squeeze(np.sqrt(np.sum((curve - point) ** 2, axis=-1)))
    return np.argmin(distances)


def iter_curve_sides(contour, corners):
    corner_indexes = sorted(index_of_nearest_to_point(contour, corner) for corner in corners)
    yield np.concatenate((contour[corner_indexes[-1]:], contour[:corner_indexes[0] + 1]))
    start = corner_indexes[0]
    for end in corner_indexes[1:]:
        yield contour[start:end+1]
        start = end


def resample_curve(curve, samples=20):
    curve = np.asarray(curve)

    space_old = np.linspace(0, 1, curve.shape[0])
    space_new = np.linspace(0, 1, samples)

    res = np.zeros((samples, 2), dtype=np.float32)
    res[:, 0] = np.interp(space_new, space_old, curve[:, 0])
    res[:, 1] = np.interp(space_new, space_old, curve[:, 1])
    return res


def angle3(a, b, c):
    """ Find angle between a-b and b-c """
    angle = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
            np.arctan2(a[1] - b[1], a[0] - b[0])
    if angle < np.pi:
        angle += 2 * np.pi
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


def calc_angles(curve):
    n = len(curve)
    yield angle3(curve[-1], curve[0], curve[1])
    for i in range(1, n-1):
        yield angle3(curve[i-1], curve[i], curve[i+1])
    yield angle3(curve[-2], curve[-1], curve[0])


def curve_corners(curve, num_corners=4, corner_threshold=0.7 * np.pi):
    if len(curve) < num_corners * 2:
        return []

    center = np.squeeze(np.mean(curve, axis=0))
    angles = np.abs(list(calc_angles(curve)))

    # distances to center
    d = np.squeeze(np.sqrt(np.sum((curve - center) ** 2, axis=-1)))
    max_d = np.max(d)
    d = d / max_d

    loss = angles / np.pi + 1 - d

    indices = np.argsort(loss)[:num_corners]
    return curve[indices]


def test():
    print(angle3(
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([0, 0]),
    ))

    curve = [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 0),
    ]
    curve = np.asarray(curve)
    print(list(calc_angles(curve)))

    reversed_curve = np.asarray(list(reversed(curve)))
    print(list(calc_angles(reversed_curve)))


if __name__ == '__main__':
   test()
