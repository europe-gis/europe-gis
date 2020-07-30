from scripts.jobs.process_raster_layer import ProcessRasterLayer

root_dem_location = '/mnt/share/mnt/RESEARCH/SATELLITE/DEM/'
root_nuts_location = '/mnt/share/mnt/RESEARCH/SATELLITE/NUTS/'
root_work_path = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/'

raster_processor = ProcessRasterLayer(root_dem_location, root_nuts_location, root_work_path)
# raster_processor.CreateBoundedRaster()
raster_processor.RasterizePopulationShapefile()


bb_coordinates = raster_processor.ReadWindowFromCoordinates(root_work_path + 'dem_aggr_rst.tif')
w = raster_processor.ReadWindowFromCoordinates(root_work_path + 'dem_aggr_rst.tif', bb_coordinates)
raster_processor.plotArray(w)

bb_coordinates = raster_processor.ReadWindowFromCoordinates(root_work_path + 'nuts_rst.tif')
w = raster_processor.ReadWindowFromCoordinates(root_work_path + 'nuts_rst.tif', bb_coordinates)
raster_processor.plotArray(w)
