import numpy as np
import imageio.v3 as iio
from demosaicing import apply_bayer_filter, demosaicing_algorithm

if __name__ == '__main__':
    name = 'house'
    image = iio.imread(f'test_images/{name}.png').astype(np.float64) / 255
    red, green, blue = apply_bayer_filter(image)
    reconstructed_image = demosaicing_algorithm(red + green + blue)
    iio.imwrite(f'{name}_reconstruction.png', (reconstructed_image * 255).astype(np.uint8))
