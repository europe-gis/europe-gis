from . import raster_preprocessor
from . import sequence_separator
from . import feature_generator

if __name__ == "__main__":

    rasters = {
        'pop': {
            'type': 'input',
            'fn': '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/pop_rst.tif'
        },
        'dem': {
            'type': 'input',
            'fn': '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/dem_aggr_rst.tif'
        },
        'ww': {
            'type': 'input',
            'fn': '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/ww_aggr_rst.tif'
        },
        'nuts': {
            'type': 'output',
            'fn': '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/nuts_rst.tif'
        }
    }
    window_size = 101
    padding_size = 50

    data_preprocessor = raster_preprocessor.DataPreprocessor(raster_config_dict = rasters)
    rasters = data_preprocessor.create_preprocessed_data()

    generator_sequences = sequence_separator.create_separated_sequences(rasters, window_size, padding_size)

    train_gen = feature_generator.InMemoryStridedArrayGenerator(
        rasters,
        window_size = window_size,
        generator_sequences = [generator_sequence[0:int(len(generator_sequence) * 0.8)] for generator_sequence in generator_sequences]
    )
    train_dataset = feature_generator.create_tfds_from_imgenerator(train_gen, batch_size = 64, window_size = window_size, channel_n = len(rasters) - 1)
    test_gen = feature_generator.InMemoryStridedArrayGenerator(
        rasters,
        window_size = window_size,
        generator_sequences = [generator_sequence[int(len(generator_sequence) * 0.8):] for generator_sequence in generator_sequences]
    )
    test_dataset = feature_generator.create_tfds_from_imgenerator(test_gen, batch_size = 64, window_size = window_size, channel_n = len(rasters) - 1)
