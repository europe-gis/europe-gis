from rasterio.windows import Window
import os
import sys

from transformations import raster_scripts
from util.config import GetConfig

config = GetConfig()

filename = 'D:/RESEARCH/SATELLITE/DEM/eu_dem_v11_E40N20.tif'
filename = '/mnt/d/RESEARCH/SATELLITE/DEM/eu_dem_v11_E40N20.tif'

raster_scripts.printRasterFileStats(filename)

w = raster_scripts.loadWindowOfRasterFile(filename, Window(0, 0, config["WINDOW"]["DEM"], config["WINDOW"]["DEM"]))

raster_scripts.plotArray(w)
