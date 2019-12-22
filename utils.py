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

