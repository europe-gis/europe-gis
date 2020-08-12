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


from scripts.jobs.dataset_creation import ReadRasterFile, CreateStridedArray

raster_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/pop_rst.tif'

a = ReadRasterFile(raster_fn)
input_array = CreateStridedArray(a)
