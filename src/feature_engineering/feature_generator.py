import random
import itertools

import numpy as np
import tensorflow as tf


class SequenceSeparator():
    def __init__(self, rasters, window_size = 28):
        self.window_size = window_size
        input_rasters = []
        for key in rasters:
            if rasters[key]['type'] == 'input':
                raster_shape = rasters[key]['data'].shape
                input_rasters.append(rasters[key]['data'])
            else:
                self.output_raster = rasters[key]['data']
        self.input_data = np.stack(input_rasters, axis = -1)

    def separate_sequence(self, sequence):
        seq = [[], []]
        for (i, j) in sequence:
            if (np.amin(self.input_data[i:i + self.window_size, j:j + self.window_size]) > -1000) & (np.amin(self.output_raster[i:i + self.window_size, j:j + self.window_size]) > -1):
                seq[int(self.output_raster[i + int(round(self.window_size / 2, 0)), j + int(round(self.window_size / 2, 0))])].append((i, j))
        self.generator_sequences = seq

    def __call__(self, generator_sequence = None):
        self.separate_sequence(generator_sequence)
        return self.generator_sequences


def create_generator_sequence(rasters, window_size = 28, padding_size = 0):
    for key in rasters:
        if rasters[key]['type'] == 'input':
            raster_shape = rasters[key]['data'].shape
            break
    generator_sequence = random.sample(list(itertools.product(range(0 + padding_size, raster_shape[0] - window_size + 1 - padding_size), range(0 + padding_size, raster_shape[1] - window_size + 1 - padding_size))), (raster_shape[0] - window_size + 1 - 2 * padding_size) * (raster_shape[1] - window_size + 1 - 2 * padding_size))
    return generator_sequence


class InMemoryStridedArrayGenerator:
    def __init__(self, rasters, window_size = 28, generator_sequences = None):
        self.window_size = window_size
        self.generator_sequences = generator_sequences
        input_rasters = []
        for key in rasters:
            if rasters[key]['type'] == 'input':
                raster_shape = rasters[key]['data'].shape
                input_rasters.append(rasters[key]['data'])
            else:
                self.output_raster = rasters[key]['data']
        self.input_data = np.stack(input_rasters, axis = -1)
        self.restart_all_generators()
        self.state = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def restart_all_generators(self):
        self.generators = [] * len(self.generator_sequences)
        self.generators = [(n for n in random.sample(sequence, len(sequence))) for sequence in self.generator_sequences]

    def restart_specific_generator(self, index):
        self.generator_sequences[index] = random.sample(self.generator_sequences[index], len(self.generator_sequences[index]))
        self.generators[index] = (n for n in self.generator_sequences[index])

    def next(self):
        if random.random() > 2 / 4:
            self.state = 1
        else:
            self.state = 0
        i, j = next(self.generators[self.state])
        return self.input_data[i:i + self.window_size, j:j + self.window_size], [int(np.amax(self.output_raster[i + int(round(self.window_size / 2, 0)) - 1:i + int(round(self.window_size / 2, 0)) + 1, j + int(round(self.window_size / 2, 0)) - 1:j + int(round(self.window_size / 2, 0)) + 1]))]

    def __call__(self):
        try:
            yield self.next()
        except StopIteration:
            self.restart_specific_generator(self.state)
            yield self.next()


def CreateTFDatasetFromInMemoryGenerator(gen, batch_size = 64, window_size = 28, channel_n = 1):

    output_shapes = ((window_size, window_size, channel_n), (1))
    output_types = (tf.float32, tf.int8)

    full_dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
    full_dataset = full_dataset.batch(batch_size)
    # full_dataset = full_dataset.shuffle(1000)
    full_dataset = full_dataset.repeat()
    return full_dataset


window_size = 101
rasters = None
full_generator_sequence = create_generator_sequence(rasters, window_size = window_size, padding_size = 140)
sequence_separator = SequenceSeparator(rasters, window_size = window_size)
generator_sequences = sequence_separator(full_generator_sequence)
train_sizes = [int(len(generator_sequence) * 0.8) for generator_sequence in generator_sequences]
total_size = sum([len(generator_sequence)for generator_sequence in generator_sequences])

train_gen = InMemoryStridedArrayGenerator(
    rasters,
    window_size = window_size,
    generator_sequences = [generator_sequence[0:int(len(generator_sequence) * 0.8)] for generator_sequence in generator_sequences]
)
train_dataset = CreateTFDatasetFromInMemoryGenerator(train_gen, batch_size = 64, window_size = window_size, channel_n = len(rasters) - 1)
test_gen = InMemoryStridedArrayGenerator(
    rasters,
    window_size = window_size,
    generator_sequences = [generator_sequence[int(len(generator_sequence) * 0.8):] for generator_sequence in generator_sequences]
)
test_dataset = CreateTFDatasetFromInMemoryGenerator(test_gen, batch_size = 64, window_size = window_size, channel_n = len(rasters) - 1)
