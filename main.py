from scripts.jobs.process_raster_layer import ProcessRasterLayer

root_dem_location = '/mnt/share/mnt/RESEARCH/SATELLITE/DEM/'
root_nuts_location = '/mnt/share/mnt/RESEARCH/SATELLITE/NUTS/'
root_work_path = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/'

raster_processor = ProcessRasterLayer(root_dem_location, root_nuts_location, root_work_path)
raster_processor.CreateBoundedRaster()


raster_processor.CreateVRT('dem_full', xy_resolution=1)
raster_processor.CreateVRT('dem_aggr', xy_resolution=100)
raster_processor.CreateTIFFromVRT('dem_aggr')
raster_processor.CreateBorderRaster()
bb_coordinates = raster_processor.GetTargetBoundingBox()

w = raster_processor.ReadWindowFromCoordinates(root_dem_location + 'rasterized.tif', bb_coordinates)
raster_processor.plotArray(w)

w = raster_processor.ReadWindowFromCoordinates(root_dem_location + 'test.tif', bb_coordinates)
raster_processor.plotArray(w)
