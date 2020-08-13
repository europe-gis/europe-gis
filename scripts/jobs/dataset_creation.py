import numpy as np
import h5py
import rasterio
import tensorflow as tf


hdf5_dir = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/'


def ReadRasterFile(input_fn):
    with rasterio.open(input_fn) as src:
        return src.read(1)


def CreateStridedArray(raster, window_size = 28):
    step = 0
    result = np.zeros(((raster.shape[0] - window_size + 1) * (raster.shape[1] - window_size + 1), window_size, window_size, 1))
    limit = 10000000
    bad_value = -1

    for i in range(0, raster.shape[0] - window_size + 1):
        for j in range(0, raster.shape[1] - window_size + 1):
            if np.amin(raster[i:i + window_size, j:j + window_size]) > bad_value:
                result[step, :, :, 0] = raster[i:i + window_size, j:j + window_size]
                step += 1
        if step > limit:
            break
    result = result[0:step - 1]
    train_size = int(0.8 * step)
    shuffle = result[np.random.permutation(result.shape[0])]
    train = shuffle[0:train_size]
    test = shuffle[train_size:]
    return train, test


def PreProcessPopulationRaster(a):
    a[a < 2] = 1
    a = np.log(a)
    a[a > 10] = 10
    a = a / 10
    return a


def CreateTFDataset(input_array):

    full_dataset = tf.data.Dataset.from_tensor_slices((input_array))
    DATASET_SIZE = input_array.shape[0]
    train_size = int(0.8 * DATASET_SIZE)
    test_size = DATASET_SIZE - train_size

    full_dataset = full_dataset.shuffle(10000)
    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    return train_dataset, test_dataset


def StoreDataHDF5(input_array, fn):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir + fn + ".h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(input_array), h5py.h5t.IEEE_F32LE, data=input_array
    )
    file.close()


class HDF5Generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf["images"]:
                yield im


def CreateTFDatasetFromGenerator(fn):

    gen = HDF5Generator(hdf5_dir + fn + ".h5")

    full_dataset = tf.data.Dataset.from_generator(gen, tf.float32)
    full_dataset = full_dataset.shuffle(1000)
    return full_dataset
