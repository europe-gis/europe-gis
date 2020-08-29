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


limit = 10000000


def CreateCompositeStridedArray(rasters, window_size = 28):
    # rasters = {
    #   'pop': {
    #           'type': 'input',
    #           'data': raster,
    #           'bad_value_threshold': 0
    #           }
    # }
    for key in rasters:
        if rasters[key]['type'] == 'input':
            raster_shape = rasters[key]['data'].shape
            break
    result_x = np.zeros(((raster_shape[0] - window_size + 1) * (raster_shape[1] - window_size + 1), window_size, window_size, len(rasters) - 1))
    result_y = np.zeros(((raster_shape[0] - window_size + 1) * (raster_shape[1] - window_size + 1), 1))

    step = 0
    for i in range(0, raster_shape[0] - window_size + 1):
        for j in range(0, raster_shape[1] - window_size + 1):
            sum_check = 0
            for (key, raster), i in zip(rasters.items(), range(len(rasters))):
                if raster['type'] == 'input':
                    if np.amin(raster['data'][i:i + window_size, j:j + window_size]) > raster['bad_value_threshold']:
                        result_x[step, :, :, i] = raster['data'][i:i + window_size, j:j + window_size]
                        sum_check += 1
                else:
                    if np.amin(raster['data'][i:i + int(round(window_size / 2, 0)), j:j + int(round(window_size / 2, 0))]) > raster['bad_value_threshold']:
                        result_y[step, 0] = raster['data'][i + int(round(window_size / 2, 0)), j + int(round(window_size / 2, 0))]
                        sum_check += 1
            if sum_check == len(rasters):
                step += 1
        if step > limit:
            break
    result_x = result_x[0:step - 1]
    result_y = result_y[0:step - 1]
    shuffle = np.random.permutation(result_x.shape[0])

    result_x = result_x[shuffle]
    result_y = result_y[shuffle]

    train_size = int(0.8 * step)
    return (result_x[0:train_size], result_y[0:train_size]), (result_x[train_size:], result_y[train_size:])


def PreProcessLogarithmPopulationRaster(a):
    a[a < 2] = 1
    a = np.log(a)
    a[a > 10] = 10
    a = a / 10
    return a


def PreProcessBorderRaster(a, bad_value = -1000):
    a[a < -bad_value] = -bad_value
    a[(a < 3) & (a > -bad_value)] = 0
    a[a > 2] = 1
    return a


def PreProcessPopulationRaster(a):
    a[a < 0] = 0
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
    file = h5py.File(hdf5_dir + fn + ".h5", "w")

    dataset = file.create_dataset(
        "images", np.shape(input_array), h5py.h5t.IEEE_F32LE, data=input_array
    )
    file.close()


def StoreCompositeDataHDF5(input_array, labels, fn):
    file = h5py.File(hdf5_dir + fn + ".h5", "w")

    dataset = file.create_dataset(
        "images", np.shape(input_array), h5py.h5t.IEEE_F32LE, data=input_array
    )
    meta_set = file.create_dataset(
        "labels", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()


class HDF5Generator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for im in hf["images"]:
                yield im


class HDF5CompositeGenerator:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for image, label in zip(hf["images"], hf["labels"]):
                yield image, label  # yield {"input_1": s1, "input_2": s2}, l


def CreateTFDatasetFromGenerator(fn):

    batch_size = 64

    gen = HDF5Generator(hdf5_dir + fn + ".h5")

    full_dataset = tf.data.Dataset.from_generator(gen, tf.float32, output_shapes=(28, 28, 1))
    full_dataset = full_dataset.batch(batch_size)
    # full_dataset = full_dataset.shuffle(1000)
    full_dataset = full_dataset.repeat()
    return full_dataset


def CreateTFDatasetFromCompositeGenerator(fn, class_n, window_size = 28, batch_size = 64, input_n = 1):

    output_shapes = (1 * (window_size, window_size, 1), (1))
    output_types = (tf.float32, tf.int8)
    generator_list = [tf.data.Dataset.from_generator(HDF5CompositeGenerator(hdf5_dir + fn + '_' + str(i) + ".h5"), output_types = output_types, output_shapes = output_shapes) for i in range(class_n)]
    full_dataset = tf.data.experimental.sample_from_datasets(generator_list, weights=class_n * [1 / class_n])

    full_dataset = full_dataset.batch(batch_size)
    full_dataset = full_dataset.repeat()
    return full_dataset
