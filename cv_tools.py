import cv2 as cv
import numpy as np

from utils import arrange4points


class TrackbarControl:
    def __init__(self, name: str, min_val: float, max_val: float, value=None):
        self.name = name
        self.value = value or min_val
        self.min_val = min_val
        self.max_val = max_val

    def setup(self, window: str):
        def _on_changed(val):
            self.value = val

        cv.createTrackbar(self.name, window, self.value, self.max_val, _on_changed)


class MeanFrameOverTime:
    """ Calculates a mean frame over specified time-buffer """

    def __init__(self, buffer_size=5):
        assert buffer_size > 1, 'Buffer size must be > 1'
        self.buffer_size = buffer_size
        self.buffer = None

    def process(self, img: np.ndarray) -> np.ndarray:
        if self.buffer is None:
            self.buffer = np.zeros((self.buffer_size, *img.shape), dtype=img.dtype)

        # Shift buffer
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = img.copy()

        if self.buffer.shape[0] < self.buffer_size:
            return img

        return np.clip(np.mean(self.buffer, axis=0), 0, 255).astype(img.dtype)


class InversePerspective:
    def __init__(self, size=(512, 512), time_smooth: float = 0.3, window_name: str = None):
        self.size = size
        self.corners = np.float32([[0, 0], [0, self.size[1]], [self.size[0], 0], self.size])
        self.corners = arrange4points(self.corners)
        self.marker_points = self.corners.copy()
        self.window_name = window_name
        self.time_smooth = time_smooth

    def process(self, img, marker_positions):
        if len(marker_positions) != 4:
            return img

        marker_points = arrange4points(np.float32(marker_positions))

        self.marker_points += self.time_smooth * (marker_points - self.marker_points)

        transformation = cv.getPerspectiveTransform(self.marker_points, self.corners)
        dst = cv.warpPerspective(img, transformation, self.size)

        if self.window_name:
            cv.imshow(self.window_name, dst)
        return dst


class ColoredMarkerDetector:
    MARKER_SIZE = 10
    MARKER_COLOR = (0, 255, 0)
    MARKER_TYPE = cv.MARKER_STAR
    MARKER_THICKNESS = 1

    FIND_BLUR = 5

    def __init__(self, window_name: str = None):
        self.window_name = window_name

        # Hue-Saturation-Value thresholds
        self.h_min = TrackbarControl('H min', min_val=0, max_val=360 // 2, value=0)
        self.h_max = TrackbarControl('H max', min_val=0, max_val=360 // 2, value=18)
        self.s_min = TrackbarControl('S min', min_val=0, max_val=255, value=158)
        self.s_max = TrackbarControl('S max', min_val=0, max_val=255, value=255)
        self.v_min = TrackbarControl('V min', min_val=0, max_val=255, value=169)
        self.v_max = TrackbarControl('V max', min_val=0, max_val=255, value=255)
        self.blur = TrackbarControl('Blur', min_val=0, max_val=10, value=7)

        if self.window_name:
            cv.namedWindow(self.window_name)

            self.h_min.setup(self.window_name)
            self.h_max.setup(self.window_name)
            self.s_min.setup(self.window_name)
            self.s_max.setup(self.window_name)
            self.v_min.setup(self.window_name)
            self.v_max.setup(self.window_name)
            self.blur.setup(self.window_name)

    def find_markers(self, img: np.ndarray, limit: int = 4):
        if img is None:
            return []

        if self.blur.value > 1 and self.blur.value % 2 == 1:
            img = cv.medianBlur(img, self.blur.value)

        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img = cv.inRange(img,
                         (self.h_min.value, self.s_min.value, self.v_min.value),
                         (self.h_max.value, self.s_max.value, self.v_max.value))
        contours, hierarchy = cv.findContours(img,
                                              mode=cv.RETR_EXTERNAL,
                                              method=cv.CHAIN_APPROX_SIMPLE)

        dst = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawContours(dst, contours=contours[:4], contourIdx=-1, color=(255, 0, 0))

        markers = []
        for contour in contours[:limit]:
            x, y = np.round(np.squeeze(np.mean(contour, axis=0)))
            markers.append((x, y))

        self.draw_markers(dst, markers)

        if self.window_name:
            cv.imshow(self.window_name, dst)

        return markers

    def draw_markers(self, img, markers):
        for x, y in markers:
            cv.drawMarker(img, (int(x), int(y)),
                          self.MARKER_COLOR,
                          markerSize=self.MARKER_SIZE,
                          markerType=self.MARKER_TYPE,
                          thickness=self.MARKER_THICKNESS)


class PieceDetector:
    MARKER_SIZE = 10
    MARKER_COLOR = (0, 255, 0)
    MARKER_TYPE = cv.MARKER_STAR
    MARKER_THICKNESS = 1

    def __init__(self, window_name: str):
        self.window_name = window_name

        self.h_min = TrackbarControl('H min', min_val=0, max_val=360 // 2, value=0)
        self.h_max = TrackbarControl('H max', min_val=0, max_val=360 // 2, value=180)
        self.s_min = TrackbarControl('S min', min_val=0, max_val=255, value=0)
        self.s_max = TrackbarControl('S max', min_val=0, max_val=255, value=82)
        self.v_min = TrackbarControl('V min', min_val=0, max_val=255, value=216)
        self.v_max = TrackbarControl('V max', min_val=0, max_val=255, value=255)
        self.blur = TrackbarControl('Blur', min_val=0, max_val=10, value=0)

        cv.namedWindow(self.window_name)

        if self.window_name:
            cv.namedWindow(self.window_name)

            self.h_min.setup(self.window_name)
            self.h_max.setup(self.window_name)
            self.s_min.setup(self.window_name)
            self.s_max.setup(self.window_name)
            self.v_min.setup(self.window_name)
            self.v_max.setup(self.window_name)
            self.blur.setup(self.window_name)

    def find_pieces(self, img: np.ndarray, max_contours: int = 20):
        if img is None:
            return []

        if self.blur.value > 1 and self.blur.value % 2 == 1:
            img = cv.medianBlur(img, self.blur.value)

        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img = cv.inRange(img,
                         (self.h_min.value, self.s_min.value, self.v_min.value),
                         (self.h_max.value, self.s_max.value, self.v_max.value))
        contours, hierarchy = cv.findContours(img,
                                              mode=cv.RETR_LIST,
                                              method=cv.CHAIN_APPROX_SIMPLE)

        contours = contours[:max_contours]

        dst = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        if self.window_name:
            cv.drawContours(dst, contours=contours, contourIdx=-1, color=(255, 0, 0))

        for contour in contours:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)

            if self.window_name:
                cv.drawContours(dst, contours=[box], contourIdx=-1, color=(0, 255, 0))

        if self.window_name:
            cv.imshow(self.window_name, dst)

        return contours
