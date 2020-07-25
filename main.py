from scripts.jobs.process_raster_layer import ProcessRasterLayer

root_dem_location = '/mnt/share/mnt/RESEARCH/SATELLITE/DEM/'
root_nuts_location = '/mnt/share/mnt/RESEARCH/SATELLITE/NUTS/'

raster_processor = ProcessRasterLayer(root_dem_location, root_nuts_location)
# raster_processor.CreateVRT()
# raster_processor.CreateTIFFromVRT()
raster_processor.CreateBorderRaster()
bb_coordinates = raster_processor.GetTargetBoundingBox()

w = raster_processor.ReadWindowFromCoordinates(root_dem_location + 'rasterized.tif', bb_coordinates)
raster_processor.plotArray(w)

w = raster_processor.ReadWindowFromCoordinates(root_dem_location + 'test.tif', bb_coordinates)
raster_processor.plotArray(w)
