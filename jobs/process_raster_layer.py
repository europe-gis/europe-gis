import os
import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio import features
from rasterio.windows import Window, from_bounds
from rasterio.enums import MergeAlg
from matplotlib import pyplot
from osgeo import gdal

path = '/mnt/z/RESEARCH/SATELLITE/DEM/'


class ProcessRasterLayer(object):
    def __init__(self):
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if file[-4:] == '.TIF':
                    files.append(os.path.join(r, file))


def CreateTIFFromVRT(vrt_fn, tif_fn):
    # vrt_in = '/mnt/z/RESEARCH/SATELLITE/DEM/test.vrt'
    # tif_out = '/mnt/z/RESEARCH/SATELLITE/DEM/test.tif'
    gdal.Translate(tif_fn, vrt_fn)
    return


def CreateVRT(input_fns, target_fn):

    xy_resolution = 1000
    vrt_options = gdal.BuildVRTOptions(
        resampleAlg='cubic',
        xRes = xy_resolution,
        yRes = xy_resolution,
        addAlpha=True
    )
    test_vrt = gdal.BuildVRT(
        target_fn,  # r'/mnt/z/RESEARCH/SATELLITE/DEM/test.vrt',
        input_fns,
        options=vrt_options
    )
    return


def PrintRasterFileStatistics(input_fn):
    with rasterio.open(input_fn) as src:  # r'/mnt/z/RESEARCH/SATELLITE/DEM/test.vrt'
        rasterio.plot.show(src)
        print(src.crs)
        print(src.count)
        print(src.width)
        print(src.height)
        print(src.bounds)
    return


def CreateBorderRaster(shp_fn, rst_fn, out_fn):

    shp_fn = '/mnt/z/RESEARCH/SATELLITE/NUTS/ref-nuts-2016-01m.shp/NUTS_RG_01M_2016_3035_LEVL_3.shp'
    rst_fn = '/mnt/z/RESEARCH/SATELLITE/DEM/test.tif'
    out_fn = '/mnt/z/RESEARCH/SATELLITE/DEM/rasterized.tif'

    counties = gpd.read_file(shp_fn)
    rst = rasterio.open(rst_fn)
    counties["C"] = 1
    meta = rst.meta.copy()
    meta.update(compress='lzw')

    counties.to_crs(meta["crs"])

    with rasterio.open(out_fn, 'w+', **meta) as out_rst:

        out_rst_data = out_rst.read(1)
        shapes = ((geom, value) for geom, value in zip(counties.geometry, counties.C) if features.is_valid_geom(geom))

        burned = features.rasterize(
            shapes=shapes,
            fill=0,
            out=out_rst_data,
            transform=out_rst.transform,
            all_touched = True,
            merge_alg = MergeAlg.replace
        )
        out_rst.write_band(1, burned)

        shapes = ((geom, value) for geom, value in zip(counties.geometry, counties.C) if features.is_valid_geom(geom))

        burned = features.rasterize(
            shapes=shapes,
            fill=0,
            out=out_rst_data,
            transform=out_rst.transform,
            all_touched = True,
            merge_alg = MergeAlg.add
        )
        out_rst.write_band(1, burned)

    return


def GetRasterBoundingBox(input_fn):

    with rasterio.open(input_fn) as src:
        bounds = src.bounds
    # (lower_left_x, lower_left_y, upper_right_x, upper_right_y) = bounds
    return bounds


def ReadWindowFromOffset(src_fn, coordinates):
    column_offset, row_offset, width, height = coordinates
    with rasterio.open(src_fn) as src:
        w = src.read(
            1,
            window=Window(column_offset, row_offset, width, height)
        )
    return w


def ReadWindowFromCoordinates(src_fn, coordinates):
    left, bottom, right, top = coordinates
    with rasterio.open(src_fn) as src:
        window = from_bounds(
            left,
            bottom,
            right,
            top,
            transform=src_fn.transform
        )
        w = src.read(
            1,
            window=window
        )
    return w


def plotArray(array):
    pyplot.imshow(array, cmap='pink')
    pyplot.show()
    return
