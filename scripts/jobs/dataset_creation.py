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


def CreateTFDataset(input_array):

    full_dataset = tf.data.Dataset.from_tensor_slices((train_examples))
    DATASET_SIZE = train_examples.shape[0]
    train_size = int(0.8 * DATASET_SIZE)
    test_size = DATASET_SIZE - train_size

    full_dataset = full_dataset.shuffle()
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    return train_dataset, test_dataset