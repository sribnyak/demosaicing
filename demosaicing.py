import numpy as np


class BayerMasks:
    def __init__(self, height, width):
        assert(height % 2 == 0 and width % 2 == 0)
        tiles = np.array([
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[0, 1], [0, 0]],
            [[0, 0], [1, 0]]
        ])
        self.green_red_row, self.green_blue_row, self.red, self.blue = (
            np.tile(tiles, (1, height // 2, width // 2)))
        self.green = self.green_red_row + self.green_blue_row
        self.red_blue = self.red + self.blue


def apply_bayer_filter(image):
    masks = BayerMasks(image.shape[0], image.shape[1])
    red, green, blue = np.squeeze(np.dsplit(image, 3), axis=-1)
    return red * masks.red, green * masks.green, blue * masks.blue


def _directional_green_interpolation(bayer_data, masks, intermediate):
    kernel = np.array([-0.25, 0.5, 0.5, 0.5, -0.25])
    conv = lambda arr: np.convolve(arr, kernel, mode='same')

    horizontal = np.apply_along_axis(conv, axis=1, arr=bayer_data)
    vertical = np.apply_along_axis(conv, axis=0, arr=bayer_data)
    green_h = bayer_data * masks.green + horizontal * masks.red_blue
    green_v = bayer_data * masks.green + vertical * masks.red_blue

    if intermediate is not None:
        intermediate.append(('step1_green_h', green_h))
        intermediate.append(('step1_green_h', green_v))

    return green_h, green_v


def _weighted_window_sum_2d(array, coeff):
    padding = coeff.shape[0] // 2
    padded_array = np.pad(array, ((padding, padding), (padding, padding)))
    view = np.lib.stride_tricks.sliding_window_view(padded_array, coeff.shape)
    return np.einsum('ijkl,kl->ij', view, coeff)


def _green_decision(bayer_data, green_h, green_v, masks, intermediate):
    chrominance_h = bayer_data - green_h
    chrominance_v = bayer_data - green_v

    gradient_h = np.abs(chrominance_h -
                        np.pad(chrominance_h, ((0, 0), (0, 2)))[:, 2:])
    gradient_v = np.abs(chrominance_v -
                        np.pad(chrominance_v, ((0, 2), (0, 0)))[2:])

    window = np.ones(shape=(5, 5), dtype=np.float64)
    window[2, :] = 3
    window[:, 2] = 3

    delta_h = _weighted_window_sum_2d(gradient_h, window)
    delta_v = _weighted_window_sum_2d(gradient_v, window)

    classifier_h = (delta_v >= delta_h).astype(masks.red.dtype)
    green = green_h * classifier_h + green_v * (1 - classifier_h)

    if intermediate is not None:
        intermediate.append(('step2_green', green))
        amp = min(1, max(delta_h.max(), delta_v.max()))
        intermediate.append(('step2_delta_h', delta_h / amp))
        intermediate.append(('step2_delta_v', delta_v / amp))

    return green, classifier_h


def _red_blue_interpolation(bayer_data, green, classifier_h,
                            masks, intermediate):
    means_h = (np.pad(bayer_data, ((0, 0), (0, 2))) +
               np.pad(bayer_data, ((0, 0), (2, 0))))[:, 1:-1] / 2
    means_v = (np.pad(bayer_data, ((0, 2), (0, 0))) +
               np.pad(bayer_data, ((2, 0), (0, 0))))[1:-1] / 2

    red = (bayer_data * masks.red + means_h * masks.green_red_row +
           means_v * masks.green_blue_row)
    blue = (bayer_data * masks.blue + means_v * masks.green_red_row +
            means_h * masks.green_blue_row)

    red_minus_blue = red - blue

    means_h = (np.pad(red_minus_blue, ((0, 0), (0, 2))) +
               np.pad(red_minus_blue, ((0, 0), (2, 0))))[:, 1:-1] / 2
    means_v = (np.pad(red_minus_blue, ((0, 2), (0, 0))) +
               np.pad(red_minus_blue, ((2, 0), (0, 0))))[1:-1] / 2

    red += masks.blue * (blue + (1 - classifier_h) * means_v
                         + classifier_h * means_h)
    blue += masks.red * (red - (1 - classifier_h) * means_v
                         - classifier_h * means_h)

    if intermediate is not None:
        intermediate.append(('step3_no_refining',
                             np.dstack((red, green, blue))))

    return red, blue


def low_freq(src):
    return ((np.pad(src, ((0, 0), (0, 2))) + np.pad(src, ((0, 0), (1, 1))) +
             np.pad(src, ((0, 0), (2, 0))))[:, 1:-1] / 3,
            (np.pad(src, ((0, 2), (0, 0))) + np.pad(src, ((1, 1), (0, 0))) +
             np.pad(src, ((2, 0), (0, 0))))[1:-1] / 3)


def _high_frequency_refining(red, green, blue, classifier_h,
                             masks, intermediate):
    red_low_h, red_low_v = low_freq(red)
    green_low_h, green_low_v = low_freq(green)
    blue_low_h, blue_low_v = low_freq(blue)

    green = masks.green * green + masks.red * (
        (1 - classifier_h) * (green_low_v + red - red_low_v) +
        classifier_h * (green_low_h + red - red_low_h)
    ) + masks.blue * (
        (1 - classifier_h) * (green_low_v + blue - blue_low_v) +
        classifier_h * (green_low_h + blue - blue_low_h)
    )

    if intermediate is not None:
        intermediate.append(('step4_refining_1',
                             np.dstack((red, green, blue))))

    red_low_h, red_low_v = low_freq(red)
    green_low_h, green_low_v = low_freq(green)
    blue_low_h, blue_low_v = low_freq(blue)

    red = (red * masks.red_blue +
           (red_low_h + green - green_low_h) * masks.green_red_row +
           (red_low_v + green - green_low_v) * masks.green_blue_row)
    blue = (blue * masks.red_blue +
            (blue_low_v + green - green_low_v) * masks.green_red_row +
            (blue_low_v + green - green_low_h) * masks.green_blue_row)

    if intermediate is not None:
        intermediate.append(('step4_refining_2',
                             np.dstack((red, green, blue))))
    
    red_low_h, red_low_v = low_freq(red)
    blue_low_h, blue_low_v = low_freq(blue)

    blue = blue * (1 - masks.red) + masks.red * (
        (1 - classifier_h) * (blue_low_v + red - red_low_v) +
        classifier_h * (blue_low_h + red - red_low_h))
    red = red * (1 - masks.blue) + masks.blue * (
        (1 - classifier_h) * (red_low_v + blue - blue_low_v) +
        classifier_h * (red_low_h + blue - blue_low_h))

    if intermediate is not None:
        intermediate.append(('step4_refining_3',
                             np.dstack((red, green, blue))))

    return red, green, blue


def demosaicing_algorithm(bayer_data, save_intermediate=False):
    intermediate = [] if save_intermediate else None
    masks = BayerMasks(bayer_data.shape[0], bayer_data.shape[1])

    green_h, green_v = _directional_green_interpolation(bayer_data,
                                                        masks, intermediate)
    green, classifier_h = _green_decision(bayer_data, green_h, green_v,
                                          masks, intermediate)
    red, blue = _red_blue_interpolation(bayer_data, green, classifier_h,
                                        masks, intermediate)
    red, green, blue = _high_frequency_refining(red, green, blue, classifier_h,
                                                masks, intermediate)
    result = np.clip(np.dstack((red, green, blue)), 0, 1)
    if save_intermediate:
        return result, intermediate
    return result
