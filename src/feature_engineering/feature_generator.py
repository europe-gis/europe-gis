import random

import numpy as np
import tensorflow as tf


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
        self.state = random.randrange(0, len(self.generator_sequences))
        i, j = next(self.generators[self.state])
        return self.input_data[i:i + self.window_size, j:j + self.window_size], [int(np.amax(self.output_raster[i + int(round(self.window_size / 2, 0)) - 1:i + int(round(self.window_size / 2, 0)) + 1, j + int(round(self.window_size / 2, 0)) - 1:j + int(round(self.window_size / 2, 0)) + 1]))]

    def __call__(self):
        try:
            yield self.next()
        except StopIteration:
            self.restart_specific_generator(self.state)
            yield self.next()


def create_tfds_from_imgenerator(gen, batch_size = 64, window_size = 28, channel_n = 1):

    output_shapes = ((window_size, window_size, channel_n), (1))
    output_types = (tf.float32, tf.int8)

    full_dataset = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
    full_dataset = full_dataset.batch(batch_size)
    # full_dataset = full_dataset.shuffle(1000)
    full_dataset = full_dataset.repeat()
    return full_dataset
