import os
import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio import features
from rasterio.windows import Window, from_bounds
from rasterio.enums import MergeAlg
from matplotlib import pyplot
from osgeo import gdal


class ProcessRasterLayer(object):
    def __init__(self, root_dem_location, root_nuts_location):
        self.root_dem_location = root_dem_location
        self.root_nuts_location = root_nuts_location
        self.dem_source_files = []
        for r, d, f in os.walk(root_dem_location):
            for file in f:
                if file[-4:] == '.TIF':
                    self.dem_source_files.append(os.path.join(r, file))

    def CreateVRT(self):
        self.dem_vrt_file = self.root_dem_location + 'test.vrt'
        xy_resolution = 1000
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg='cubic',
            xRes = xy_resolution,
            yRes = xy_resolution,
            addAlpha=True
        )
        test_vrt = gdal.BuildVRT(
            self.dem_vrt_file,
            self.dem_source_files,
            options=vrt_options
        )
        test_vrt = None
        return

    def CreateTIFFromVRT(self):
        self.dem_rst_fn = self.root_dem_location + 'test.tif'
        gdal.Translate(self.dem_rst_fn, self.dem_vrt_file)
        return

    def CreateBorderRaster(self):

        shp_fn = self.root_nuts_location + 'ref-nuts-2016-01m.shp/NUTS_RG_01M_2016_3035_LEVL_3.shp'
        out_fn = self.root_dem_location + 'rasterized.tif'

        counties = gpd.read_file(shp_fn)
        rst = rasterio.open(self.dem_rst_fn)
        counties["C"] = 1
        meta = rst.meta.copy()
        meta.update(compress='lzw')

        counties.to_crs(meta["crs"])
        self.crs = meta["crs"]
        self.counties = counties
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

    def GetTargetBoundingBox(self):
        if not hasattr(self, 'counties'):
            shp_fn = self.root_nuts_location + 'ref-nuts-2016-01m.shp/NUTS_RG_01M_2016_3035_LEVL_3.shp'

            counties = gpd.read_file(shp_fn)
            rst = rasterio.open(self.dem_rst_fn)
            meta = rst.meta.copy()
            meta.update(compress='lzw')
            counties.to_crs(meta["crs"])
        bounding_boxes = self.counties[self.counties['CNTR_CODE'] == 'HU'].bounds
        bounds = (min(bounding_boxes['minx']), min(bounding_boxes['miny']), max(bounding_boxes['maxx']), max(bounding_boxes['maxy']))
        return bounds

    @staticmethod
    def ReadWindowFromCoordinates(src_fn, coordinates):
        left, bottom, right, top = coordinates
        with rasterio.open(src_fn) as src:
            window = from_bounds(
                left,
                bottom,
                right,
                top,
                transform=src.transform
            )
            w = src.read(
                1,
                window=window
            )
        return w

    @staticmethod
    def plotArray(array):
        pyplot.imshow(array, cmap='pink')
        pyplot.show()
        return

    @staticmethod
    def PrintRasterFileStatistics(input_fn):
        with rasterio.open(input_fn) as src:
            rasterio.plot.show(src)
            print(src.crs)
            print(src.count)
            print(src.width)
            print(src.height)
            print(src.bounds)
        return

    @staticmethod
    def GetRasterBoundingBox(input_fn):

        with rasterio.open(input_fn) as src:
            bounds = src.bounds
        # (lower_left_x, lower_left_y, upper_right_x, upper_right_y) = bounds
        return bounds

    @staticmethod
    def ReadWindowFromOffset(src_fn, coordinates):
        column_offset, row_offset, width, height = coordinates
        with rasterio.open(src_fn) as src:
            w = src.read(
                1,
                window=Window(column_offset, row_offset, width, height)
            )
        return w
