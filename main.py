from scripts.jobs.process_raster_layer import ProcessRasterLayer

raster_processor = ProcessRasterLayer(
    restart = False
)
# raster_processor.CreateBoundedDEMRaster()
# raster_processor.PreparePopulationShapefile()
# raster_processor.RasterizePopulationShapefile()
raster_processor.CreateNonEUCountryRaster()
raster_processor.CreateBoundedWWRaster()
