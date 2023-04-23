import numpy as np
import imageio.v3 as iio
from demosaicing import apply_bayer_filter, demosaicing_algorithm


def save_scaled(name, image, scale):
    if image.ndim == 2:
        image = np.dstack((image, image, image))
    image = (image * 255).astype(np.uint8)
    h, w, _ = image.shape
    big_img = np.zeros(shape=(h * scale, w * scale, 3), dtype = np.uint8)
    for i in range(h * scale):
        for j in range(w * scale):
            big_img[i, j] = image[i // scale, j // scale]
    iio.imwrite(name, big_img)


if __name__ == '__main__':
    name = 'house'
    image = iio.imread(f'test_images/{name}.png').astype(np.float64) / 255
    red, green, blue = apply_bayer_filter(image)
    reconstructed_image, intermediate = demosaicing_algorithm(
        red + green + blue, save_intermediate=True)
    iio.imwrite(f'{name}_reconstruction.png', (reconstructed_image * 255).astype(np.uint8))
    for i, (nm, img) in enumerate(intermediate):
        save_scaled(f'debug/{i}_{nm}.png', np.clip(img, 0, 1), 4)
