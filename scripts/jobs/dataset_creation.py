import numpy as np
import rasterio


def ReadRasterFile(input_fn):
    with rasterio.open(input_fn) as src:
        return src.read(1)


def CreateStridedArray(raster, window_size = 28):
    step = 0
    result = np.zeros(((raster.shape[0] - window_size + 1) * (raster.shape[1] - window_size + 1), window_size, window_size))

    for i in range(0, raster.shape[0] - window_size + 1):
        for j in range(0, raster.shape[1] - window_size + 1):
            result[step][:][:] = raster[i:i + window_size, j:j + window_size]
            step += 1

    return result
