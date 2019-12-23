from typing import Tuple
import numpy as np


NUM_PARAMS = 31
NUM_PARAMS_LITE = 7
NUM_PARAMS_CORNERS = 3


def map_to_range(x, target_range, source_range=(-1, 1)):
    """ maps x from [-1, 1] to [a, b]"""
    src_min, src_max = source_range
    tgt_min, tgt_max = target_range
    p = (x - src_min) / (src_max - src_min)
    return tgt_min + (tgt_max - tgt_min) * p


def _angle_to_position(c, r, a, ex=1.0, ey=1.0):
    return c + r * np.float32([np.sin(a) * ex, np.cos(a) * ey])


def arc(c, r, arc_min=0, arc_max=np.pi * 2, div=10, ex=1, ey=1):
    if r < 0:
        arc_min, arc_max = arc_max, arc_min

    for a in np.linspace(arc_min, arc_max, div):
        yield _angle_to_position(c, r, -a, ex=ex, ey=ey)


def gen_piece(args):
    """ Assuming all args in [-1, 1] range"""
    args = np.float32(args)

    # Size parameters
    p_size_x = map_to_range(args[0], (50, 400))
    p_size_y = map_to_range(args[1], (50, 400))
    p_rot = map_to_range(args[2], (-np.pi, np.pi))
    p_size = np.float32((p_size_x, p_size_y))

    max_corner_offset = p_size * 0.1
    max_middle_offset = p_size * 0.3
    min_radius = 0.1 * min(p_size_x, p_size_y)
    max_radius = 0.3 * min(p_size_x, p_size_y)

    # Corner offset parameter
    p_p1 = args[3:5]
    p_p2 = args[5:7]
    p_p3 = args[7:9]
    p_p4 = args[9:11]

    # Middle edge offset
    p_e1 = args[11:13]
    p_e2 = args[13:15]
    p_e3 = args[15:17]
    p_e4 = args[17:19]

    # Radius
    r1, r2, r3, r4 = map_to_range(args[19:23], (min_radius, max_radius))

    # Ellipse params
    p_el1, p_el2, p_el3, p_el4 = map_to_range(args[23:27], (0.25, 1.75))

    # Neck heights
    p_nh1, p_nh2, p_nh3, p_nh4 = map_to_range(args[27:31], (-max_radius, max_radius))

    # Neck widths
    p_nw1, p_nw2, p_nw3, p_nw4 = np.abs([r1, r2, r3, r4]) * 0.75

    r1 *= np.sign(p_nh1)
    r2 *= np.sign(p_nh2)
    r3 *= np.sign(p_nh3)
    r4 *= np.sign(p_nh4)

    # Base corners
    c1 = np.float32((0, 0))
    c2 = np.float32((p_size_x, 0))
    c3 = np.float32((p_size_x, p_size_y))
    c4 = np.float32((0, p_size_y))

    p1 = c1 + p_p1 * max_corner_offset
    p2 = c2 + p_p2 * max_corner_offset
    p3 = c3 + p_p3 * max_corner_offset
    p4 = c4 + p_p4 * max_corner_offset

    # middle points
    p1p2c = 0.5 * (p2 + p1) + p_e1 * max_middle_offset
    p2p3c = 0.5 * (p3 + p2) + p_e2 * max_middle_offset
    p3p4c = 0.5 * (p4 + p3) + p_e3 * max_middle_offset
    p4p1c = 0.5 * (p1 + p4) + p_e4 * max_middle_offset

    # Circle-center points
    c1 = p1p2c + np.float32([0, -r1 - p_nh1]) * 0.7
    c2 = p2p3c + np.float32([r2 + p_nh2, 0]) * 0.7
    c3 = p3p4c + np.float32([0, r3 + p_nh3]) * 0.7
    c4 = p4p1c + np.float32([-r4 - p_nh4, 0]) * 0.7

    k1 = np.float32([p_nw1, 0])
    k2 = np.float32([0, p_nw2])
    k3 = np.float32([-p_nw3 * 0.75, 0])
    k4 = np.float32([0, -p_nw4 * 0.75])

    polygon = [
        p1,
        p1p2c - k1,
        *arc(c1, r1, np.pi * 0.25, (2 - 0.25) * np.pi, ex=p_el1),
        p1p2c + k1,
        p2,
        p2p3c - k2,
        *arc(c2, r2, np.pi * 0.75, (2 + 0.25) * np.pi, ey=p_el2),
        p2p3c + k2,
        p3,
        p3p4c - k3,
        *arc(c3, r3, np.pi * 1.25, (2 + 0.75) * np.pi, ex=p_el3),
        p3p4c + k3,
        p4,
        p4p1c - k4,
        *arc(c4, r4, np.pi * 1.75, (2 + 1.25) * np.pi, ey=p_el4),
        p4p1c + k4
    ]

    # Perform rotation
    s, c = np.sin(p_rot), np.cos(p_rot)
    rot_matrix = np.array(
        ((c, -s), (s, c))
    )
    polygon = np.matmul(polygon, rot_matrix)
    corners = np.matmul([p1, p2, p3, p4], rot_matrix)
    return np.int0(polygon), corners


def gen_piece_lite(args_lite):
    """ Assuming all args in [-1, 1] range"""
    args = np.zeros(NUM_PARAMS, dtype=np.float32)

    # Size
    args[0] = args_lite[0]
    args[1] = args_lite[1]

    # Rotation
    args[2] = args_lite[2]

    # Radiuses
    args[27:31] = args_lite[3:7]

    return gen_piece(args)


def gen_piece_options(args_lite, options):
    args = np.zeros(NUM_PARAMS, dtype=np.float32)
    next_index = 0
    if 'size' in options:
        args[0] = args_lite[0]
        args[1] = args_lite[1]
        next_index += 2

    if 'rotation' in options:
        args[2] = args_lite[next_index]
        next_index += 1

    if 'corner_offset' in options:
        for i in range(0, 4, 2):
            args[3+i:5+i] = args_lite[next_index+i:next_index+i+2]
        next_index += 8

    if 'middle_offset' in options:
        for i in range(0, 4, 2):
            args[11+i:13+i] = args_lite[next_index+i:next_index+i+2]
        next_index += 8

    if 'radius' in options:
        args[19:23] = args_lite[next_index:next_index+4]
        next_index += 4

    if 'ellipse' in options:
        args[23:27] = args_lite[next_index:next_index + 4]
        next_index += 4

    if 'neck_heights' in options:
        args[27:31] = args_lite[next_index:next_index+4]
        next_index += 4

    return gen_piece(args)


def get_options_number(options):
    next_index = 0
    if 'size' in options:
        next_index += 2
    if 'rotation' in options:
        next_index += 1
    if 'corner_offset' in options:
        next_index += 8
    if 'middle_offset' in options:
        next_index += 8
    if 'radius' in options:
        next_index += 4
    if 'ellipse' in options:
        next_index += 4
    if 'neck_heights' in options:
        next_index += 4
    return next_index
