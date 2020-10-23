import random
import itertools

import numpy as np


def create_separated_sequences(rasters, window_size = 28, padding_size = 0):

    validation_raster = rasters['validation']['data']
    for key in rasters:
        if rasters[key]['type'] == 'output':
            output_raster = rasters[key]['data']
            raster_shape = rasters[key]['data'].shape
            break
    full_sequence = random.sample(
        list(
            itertools.product(
                range(0 + padding_size, raster_shape[0] - window_size + 1 - padding_size),
                range(0 + padding_size, raster_shape[1] - window_size + 1 - padding_size)
            )
        ),
        (raster_shape[0] - window_size + 1 - 2 * padding_size) * (raster_shape[1] - window_size + 1 - 2 * padding_size)
    )

    sequences = [[] for _ in range(len(np.unique(output_raster)))]
    output_value_dict = {}
    for x in range(0, len(np.unique(output_raster))):
        output_value_dict[np.unique(output_raster)[x]] = x

    for (i, j) in full_sequence:
        if np.amin(validation_raster[i:i + window_size, j:j + window_size]) > 0:
            sequences[output_value_dict[int(output_raster[i + int(round(window_size / 2, 0)), j + int(round(window_size / 2, 0))])]].append((i, j))
    return sequences
