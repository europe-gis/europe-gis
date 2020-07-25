import os
import time
import struct
import collections
import numpy as np
from osgeo import gdal, ogr
import sys

gdal.UseExceptions()

os.environ['POSTGIS_ENABLE_OUTDB_RASTERS'] = '1'
# dataset = gdal.Open('D:/RESEARCH/SATELLITE/DEM/eu_dem_v11_E30N20.tif', gdal.GA_ReadOnly)
dataset = gdal.Open("PG: dbname=postgis_25_sample host=localhost user=postgres password=forever port=5432 mode=2 schema=public column=rast table=demelevation ", gdal.GA_ReadOnly)

print(
    "Driver: {}/{}".format(
        dataset.GetDriver().ShortName,
        dataset.GetDriver().LongName
    )
)
print(
    "Size is {} x {} x {}".format(
        dataset.RasterXSize,
        dataset.RasterYSize,
        dataset.RasterCount
    )
)
# print("Projection is {}".format(dataset.GetProjection()))
geotransform = dataset.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

band = dataset.GetRasterBand(1)
print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

min = band.GetMinimum()
max = band.GetMaximum()
if not min or not max:
    (min, max) = band.ComputeRasterMinMax(True)
print("Min={:.3f}, Max={:.3f}".format(min, max))

if band.GetOverviewCount() > 0:
    print("Band has {} overviews".format(band.GetOverviewCount()))

if band.GetRasterColorTable():
    print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

# scanline = band.ReadRaster(
#     xoff=0,
#     yoff=0,
#     xsize=band.XSize,
#     ysize=1,
#     buf_xsize=band.XSize,
#     buf_ysize=1,
#     buf_type=gdal.GDT_Float32
# )

n = 10000

t = time.time()

scanline = band.ReadRaster(
    xoff=band.XSize / 2,
    yoff=band.YSize / 2,
    xsize=n,
    ysize=n,
    buf_xsize=n,
    buf_ysize=n,
    buf_type=gdal.GDT_Float32
)

print(time.time() - t)

square_tuple_of_floats = struct.unpack('f' * n * n, scanline)

print(time.time() - t)

sresults = np.array(list(square_tuple_of_floats)).reshape(n, n)
# scanline = band.ReadRaster(
#     xoff=band.XSize / 2,
#     yoff=band.YSize / 2,
#     xsize=n,
#     ysize=1,
#     buf_xsize=n,
#     buf_ysize=1,
#     buf_type=gdal.GDT_Float32
# )
# tuple_of_floats = struct.unpack('f' * 1 * n, scanline)

# tuple_of_floats = struct.unpack('f' * band.XSize, scanline)
counter = collections.Counter(list(square_tuple_of_floats))

print(counter)
