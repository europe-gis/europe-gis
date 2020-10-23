import raster_preprocessor
import sequence_separator

import pickle


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
            'fn': '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/nuts_rst3.tif'
        },
        'area': {
            'type': 'aux',
            'fn': '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/nuts_area_rst3.tif'
        }
    }
    window_size = 50
    padding_size = 50

    data_preprocessor = raster_preprocessor.DataPreprocessor(raster_config_dict = rasters, window_size = window_size)
    rasters = data_preprocessor.create_preprocessed_data()

    generator_sequences = sequence_separator.create_separated_sequences(rasters, window_size, padding_size)

    pickle.dump([rasters, generator_sequences, window_size], open("../../tmp/features.p", "wb"))
