import random
import itertools
from os import path
import numpy as np
import h5py
import rasterio
import tensorflow as tf
from tqdm import tqdm
# from IPython.display import display, clear_output


hdf5_dir = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/'



def CreateStridedArray(raster, window_size = 28, limit = 10000000, bad_value = -1):
    step = 0
    result = np.zeros(((raster.shape[0] - window_size + 1) * (raster.shape[1] - window_size + 1), window_size, window_size, 1))

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


def CreateCompositeStridedArray(rasters, gen, window_size = 28, limit = 500000):
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
    # result_x = np.zeros(((raster_shape[0] - window_size + 1) * (raster_shape[1] - window_size + 1), window_size, window_size, len(rasters) - 1), dtype = np.float32)
    # result_y = np.zeros(((raster_shape[0] - window_size + 1) * (raster_shape[1] - window_size + 1), 1))
    result_x = np.zeros((limit, window_size, window_size, len(rasters) - 1), dtype = np.float32)
    result_y = np.zeros((limit, 1), dtype = np.int8)

    step = 0
    with tqdm(total=limit) as pbar:
        for i, j in gen:
            # clear_output(wait=True)
            # display('Row: ' + str(i) + ' Column: ' + str(j) + ' Step: ' + str(step))
            sum_check = 0
            for (key, raster), r in zip(rasters.items(), range(len(rasters))):
                if raster['type'] == 'input':
                    if np.amin(raster['data'][i:i + window_size, j:j + window_size]) > raster['bad_value_threshold']:
                        result_x[step, :, :, r] = raster['data'][i:i + window_size, j:j + window_size]
                        sum_check += 1
                else:
                    if np.amin(raster['data'][i:i + window_size, j:j + window_size]) > raster['bad_value_threshold']:
                        result_y[step, 0] = raster['data'][i + int(round(window_size / 2, 0)), j + int(round(window_size / 2, 0))]
                        sum_check += 1
            if sum_check == len(rasters):
                step += 1
                pbar.update(1)
            if step >= limit:
                break
    result_x = result_x[0:step - 1]
    result_y = result_y[0:step - 1]

    train_size = int(0.8 * step)
    return (result_x[0:train_size], result_y[0:train_size]), (result_x[train_size:], result_y[train_size:])


class StridedArrayGenerator:
    def __init__(self, rasters, window_size = 28, file_row_n = 100000):
        self.window_size = window_size
        self.file_row_n = file_row_n
        self.rasters = rasters
        for key in rasters:
            if rasters[key]['type'] == 'input':
                raster_shape = rasters[key]['data'].shape

        self.gen = (n for n in random.sample(list(itertools.product(range(0, raster_shape[0] - window_size + 1), range(0, raster_shape[1] - window_size + 1))), (raster_shape[0] - window_size + 1) * (raster_shape[1] - window_size + 1)))

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return CreateCompositeStridedArray(self.rasters, self.gen, window_size = self.window_size, limit = self.file_row_n)

    def __call__(self):
        yield CreateCompositeStridedArray(self.rasters, self.gen, window_size = self.window_size, limit = self.file_row_n)








class InMemoryStridedArrayGeneratorForLogisticRegression:
    def __init__(self, rasters, window_size = 28, generator_sequence = None, batch_size = 1000):
        self.window_size = window_size
        self.generator_sequence = generator_sequence
        input_rasters = []
        for key in rasters:
            if rasters[key]['type'] == 'input':
                raster_shape = rasters[key]['data'].shape
                input_rasters.append(rasters[key]['data'])
            else:
                self.output_raster = rasters[key]['data']
        self.input_data = np.stack(input_rasters, axis = -1)
        self.restart_all_generators()
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def restart_all_generators(self):
        self.generator = (n for n in random.sample(self.generator_sequence, len(self.generator_sequence)))

    def next(self):
        images = []
        labels = []
        for step in range(self.batch_size):
            i, j = next(self.generator)
            images.append(self.input_data[i:i + self.window_size, j:j + self.window_size].flatten())
            labels.append(self.output_raster[i + int(round(self.window_size / 2, 0)), j + int(round(self.window_size / 2, 0))])
        return np.array(images), np.array(labels)

    def __call__(self):
        try:
            yield self.next()
        except StopIteration:
            self.restart_all_generators()
            yield self.next()




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



def get_hdf5_dataset_size(file):
    with h5py.File(file, 'r') as hf:
        return len(hf["images"])


def CreateTFDatasetFromCompositeGenerator(fn, class_n, window_size = 28, batch_size = 64, channel_n = 1):

    file_count = 0
    while True:
        if path.exists(hdf5_dir + fn + '_0_' + str(file_count) + ".h5"):
            file_count += 1
        else:
            break

    output_shapes = ((window_size, window_size, channel_n), (1))
    output_types = (tf.float32, tf.int8)

    merge_list = []
    merge_list_size = []
    for file_id in range(file_count):
        generator_list = [tf.data.Dataset.from_generator(HDF5CompositeGenerator(hdf5_dir + fn + '_' + str(i) + '_' + str(file_id) + '.h5'), output_types = output_types, output_shapes = output_shapes).repeat() for i in range(class_n)]
        merge_list.append(tf.data.experimental.sample_from_datasets(generator_list, weights=class_n * [1 / class_n]))
        merge_list_size.append(get_hdf5_dataset_size(hdf5_dir + fn + '_0_' + str(file_id) + '.h5'))

    total_size = sum(merge_list_size)
    full_dataset = tf.data.experimental.sample_from_datasets(merge_list, weights=[x / total_size for x in merge_list_size])
    full_dataset = full_dataset.batch(batch_size)
    # full_dataset = full_dataset.repeat()
    return full_dataset
