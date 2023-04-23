import numpy as np


def apply_bayer_filter(image):
    h, w, _ = image.shape
    red = np.zeros(shape=(h, w), dtype=np.float64)
    green = red.copy()
    blue = red.copy()
    for i in range(0, h, 2):
        for j in range(0, w, 2):
            green[i, j] = image[i, j, 1]
        for j in range(1, w, 2):
            red[i, j] = image[i, j, 0]
    for i in range(1, h, 2):
        for j in range(0, w, 2):
            blue[i, j] = image[i, j, 2]
        for j in range(1, w, 2):
            green[i, j] = image[i, j, 1]
    return red, green, blue


def take(array2d, i, j):
    if 0 <= i < array2d.shape[0] and 0 <= j < array2d.shape[1]:
        return array2d[i, j]
    return np.float64(0)


def red_blue_positions(height, width):
    first_non_green = 1
    for i in range(height):
        for j in range(first_non_green, width, 2):
            yield i, j
        first_non_green = 1 - first_non_green


def directional_green_interpolation(bayer_data):
    height, width = bayer_data.shape
    green_h = bayer_data.copy()  # green positions are copied
    green_v = bayer_data.copy()  # other values will be replaced

    for i, j in red_blue_positions(height, width):
        r = lambda k: take(bayer_data, i, j + k)  # r - relative indexing
        green_h[i, j] = (r(1) + r(-1) + r(0)) / 2 - (r(2) + r(-2)) / 4
        r = lambda k: take(bayer_data, i + k, j)
        green_v[i, j] = (r(1) + r(-1) + r(0)) / 2 - (r(2) + r(-2)) / 4

    return green_h, green_v


def green_decision(bayer_data, green_h, green_v,
                   cardinal_directions_improvement=True):
    height, width = bayer_data.shape

    # "chrominance" is R - G in red locations, B - G in blue locations
    # and 0 in green locations
    chrominance_h = bayer_data - green_h
    chrominance_v = bayer_data - green_v

    # also 0 in green locations, this will be useful
    gradient_h = chrominance_h.copy()
    gradient_v = chrominance_v.copy()

    for i, j in red_blue_positions(height, width):
        gradient_h[i, j] -= take(chrominance_h, i, j + 2)
        gradient_v[i, j] -= take(chrominance_v, i + 2, j)
    gradient_h = np.abs(gradient_h)
    gradient_v = np.abs(gradient_v)
    # could be easily rewritten without loops

    window = np.ones(shape=(5, 5), dtype=np.float64)
    if cardinal_directions_improvement:
        window[2, :] = 3
        window[:, 2] = 3

    delta_h = np.zeros(shape=(height, width), dtype=np.float64)
    delta_v = delta_h.copy()
    padded_grad_h = np.zeros(shape=(height + 4, width + 4), dtype=np.float64)
    padded_grad_v = padded_grad_h.copy()
    padded_grad_h[2 : height + 2, 2 : width + 2] = gradient_h
    padded_grad_v[2 : height + 2, 2 : width + 2] = gradient_v
    green = green_h.copy()
    for i, j in red_blue_positions(height, width):
        delta_h[i, j] = np.sum(window * padded_grad_h[i : i + 5, j : j + 5])
        delta_v[i, j] = np.sum(window * padded_grad_v[i : i + 5, j : j + 5])
        if delta_v[i, j] < delta_h[i, j]:
            green[i, j] = green_v[i, j]

    return green, delta_h, delta_v


def red_blue_interpolation(bayer_data, green, delta_h, delta_v):
    height, width = bayer_data.shape
    red = bayer_data.copy()
    blue = bayer_data.copy()

    # green positions first
    for i in range(0, height, 2):  # green-red rows
        for j in range(0, width, 2):
            red[i, j] = (take(bayer_data, i, j - 1) +
                         take(bayer_data, i, j + 1)) / 2
            blue[i, j] = (take(bayer_data, i - 1, j) +
                          take(bayer_data, i + 1, j)) / 2
    for i in range(1, height, 2):  # green-blue rows
        for j in range(1, width, 2):
            blue[i, j] = (take(bayer_data, i, j - 1) +
                          take(bayer_data, i, j + 1)) / 2
            red[i, j] = (take(bayer_data, i - 1, j) +
                         take(bayer_data, i + 1, j)) / 2
    
    # now red in blue positions, blue in red positions
    red_minus_blue = red - blue
    for i in range(1, height, 2):
        for j in range(0, width, 2):
            if delta_v[i, j] < delta_h[i, j]:
                red[i, j] = blue[i, j] + (take(red_minus_blue, i - 1, j) +
                                          take(red_minus_blue, i + 1, j)) / 2
            else:
                red[i, j] = blue[i, j] + (take(red_minus_blue, i, j - 1) +
                                          take(red_minus_blue, i, j + 1)) / 2
    for i in range(0, height, 2):
        for j in range(1, width, 2):
            if delta_v[i, j] < delta_h[i, j]:
                blue[i, j] = red[i, j] - (take(red_minus_blue, i - 1, j) +
                                          take(red_minus_blue, i + 1, j)) / 2
            else:
                blue[i, j] = red[i, j] - (take(red_minus_blue, i, j - 1) +
                                          take(red_minus_blue, i, j + 1)) / 2

    return red, blue


def replace_high_1d(dst, src, i, j, vertical: bool):
    if vertical:
        low_pass = lambda arr, i, j: (take(arr, i - 1, j) + arr[i, j] +
                                      take(arr, i + 1, j)) / 3
    else:
        low_pass = lambda arr, i, j: (take(arr, i, j - 1) + arr[i, j] +
                                      take(arr, i, j + 1)) / 3
    dst[i, j] = low_pass(dst, i, j) + src[i, j] - low_pass(src, i, j)


def high_frequency_refining(red, green, blue, delta_h, delta_v):
    height, width = green.shape

    for i in range(0, height, 2):  # red locations
        for j in range(1, width, 2):
            replace_high_1d(green, red, i, j, delta_v[i, j] < delta_h[i, j])

    for i in range(1, height, 2):  # blue locations
        for j in range(0, width, 2):
            replace_high_1d(green, blue, i, j, delta_v[i, j] < delta_h[i, j])

    for i in range(0, height, 2):  #  green locations - red rows
        for j in range(0, width, 2):
            replace_high_1d(red, green, i, j, vertical=False)
            replace_high_1d(blue, green, i, j, vertical=True)

    for i in range(1, height, 2):  #  green locations - blue rows
        for j in range(1, width, 2):
            replace_high_1d(red, green, i, j, vertical=True)
            replace_high_1d(blue, green, i, j, vertical=False)

    for i in range(0, height, 2):  # red locations
        for j in range(1, width, 2):
            replace_high_1d(blue, red, i, j, delta_v[i, j] < delta_h[i, j])

    for i in range(1, height, 2):  # blue locations
        for j in range(0, width, 2):
            replace_high_1d(red, blue, i, j, delta_v[i, j] < delta_h[i, j])

    return red, green, blue


def demosaicing_algorithm(bayer_data):
    green_h, green_v = directional_green_interpolation(bayer_data)
    green, delta_h, delta_v = green_decision(bayer_data, green_h, green_v)
    red, blue = red_blue_interpolation(bayer_data, green, delta_h, delta_v)
    red, green, blue = high_frequency_refining(red, green, blue,
                                               delta_h, delta_v)
    return np.clip(np.dstack((red, green, blue)), 0, 1)
