from raster_preprocessor import DataPreprocessor

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

    data_preprocessor = DataPreprocessor(raster_config_dict = rasters)
