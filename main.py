# from scripts.jobs.process_raster_layer import ProcessRasterLayer

# raster_processor = ProcessRasterLayer(
#     restart = False
# )
# raster_processor.CreateBoundedDEMRaster()
# raster_processor.PreparePopulationShapefile()
# raster_processor.RasterizePopulationShapefile()
# raster_processor.CreateNonEUCountryRaster()
# raster_processor.CreateBoundedWWRaster()


# from scripts.jobs.model_training import BorderRecognitionNetwork

# network = BorderRecognitionNetwork(
#     [
#         'ww', 'dem', 'pop', 'nuts'
#     ],
#     image_size=100,
#     nb_labels=2,
#     architecture='resnet'
# )

from scripts.jobs.dataset_creation import ReadRasterFile, CreateCompositeStridedArray, PreProcessBorderRaster, StoreCompositeDataHDF5

raster_dem_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/dem_aggr_rst.tif'
raster_nuts_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/nuts_rst.tif'

rasters = {
    'dem': {
        'type': 'input',
        'data': ReadRasterFile(raster_dem_fn),
        'bad_value_threshold': -1000
    },
    'nuts': {
        'type': 'output',
        'data': PreProcessBorderRaster(ReadRasterFile(raster_nuts_fn)),
        'bad_value_threshold': 0
    }
}

(train_x, train_y), (test_x, test_y) = CreateCompositeStridedArray(rasters)
StoreCompositeDataHDF5(train_x, train_y, 'dem_nuts_train')
StoreCompositeDataHDF5(test_x, test_y, 'dem_nuts_test')
