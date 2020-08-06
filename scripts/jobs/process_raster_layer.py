import os
import json
import shutil
import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.plot
import rasterio.env
from rasterio import features
from rasterio.windows import Window, from_bounds
from rasterio.enums import MergeAlg
from matplotlib import pyplot
from osgeo import gdal


class FilePath(object):
    def __init__(self, config):
        self.root_dem_path = config['ROOT_LOCATIONS']['DEM']
        self.root_nuts_path = config['ROOT_LOCATIONS']['NUTS']
        self.root_work_path = config['ROOT_LOCATIONS']['WORK']
        self.root_pop_path = config['ROOT_LOCATIONS']['POP']

        self.dem_source_fns = []
        for r, d, f in os.walk(self.root_dem_path):
            for file in f:
                if file[-4:] == '.TIF':
                    self.dem_source_fns.append(os.path.join(r, file))


class ProcessRasterLayer(object):
    @staticmethod
    def ReadWindowFromCoordinates(cls, src_fn, coordinates):
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
    def plotArray(cls, array, cmap='viridis'):
        pyplot.imshow(array, cmap=cmap)
        pyplot.show()
        return

    @staticmethod
    def PrintRasterFileStatistics(cls, input_fn):
        with rasterio.open(input_fn) as src:
            # rasterio.plot.show(src)
            print(src.crs)
            print(src.count)
            print(src.width)
            print(src.height)
            print(src.bounds)
        return

    @staticmethod
    def GetRasterBoundingBox(cls, input_fn):

        with rasterio.open(input_fn) as src:
            bounds = src.bounds
        # (lower_left_x, lower_left_y, upper_right_x, upper_right_y) = bounds
        return bounds

    @staticmethod
    def RoundBoundingBox(cls, fr, bb, resolution, pad = 0):

        def RoundBound(outer, inner):
            if outer > inner:
                return inner + (outer - inner) % resolution + pad
            else:
                return inner - (inner - outer) % resolution - pad

        result_list = []
        for i in range(len(fr)):
            result_list.append(RoundBound(fr[i], bb[i]))
        return tuple(result_list)

    @staticmethod
    def ReadWindowFromOffset(cls, src_fn, coordinates):
        column_offset, row_offset, width, height = coordinates
        with rasterio.open(src_fn) as src:
            w = src.read(
                1,
                window=Window(column_offset, row_offset, width, height)
            )
        return w

    @staticmethod
    def PolygonizeRasterLayer(cls, src_fn):
        mask = None
        with rasterio.Env():
            with rasterio.open(src_fn) as src:
                meta = src.meta.copy()
                meta.update(compress='lzw')
                image = src.read(1)
                results = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for i, (s, v)
                    in enumerate(
                        features.shapes(image, mask=mask, transform=src.transform)
                    )
                )

        geoms = list(results)
        gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms, meta["crs"])
        return gpd_polygonized_raster

    def __init__(self, restart = False):

        with open(str(pathlib.Path().absolute()).replace('/notebooks', '') + '/config/config.json') as json_file:
            self.config = json.load(json_file)

        self.filepath = FilePath(self.config)

        if restart:
            shutil.rmtree(self.filepath.root_work_path, ignore_errors=True)
            # if not os.path.exists(root_work_path):
            os.makedirs(self.filepath.root_work_path)

        self.raster_metadata = {}

    def LoadBorderShapefile(self):

        shp_fn = self.filepath.root_nuts_path + self.config['FILES']['NUTS']
        self.nuts_borders = gpd.read_file(shp_fn).to_crs(self.config['CRS'])
        return

    def LoadCountryBorderShapefile(self):

        shp_fn = self.config['ROOT_LOCATIONS']['CB'] + self.config['FILES']['CB']
        self.country_borders = gpd.read_file(shp_fn).to_crs(self.config['CRS'])
        self.country_borders = self.country_borders[~self.country_borders['ISO2'].isin(self.config['COUNTRY_CODE_LIST'])]

        return

    def GetTargetBoundingBox(self):
        if not hasattr(self, 'nuts_borders'):
            self.LoadBorderShapefile()
        self.bounding_box = tuple(self.nuts_borders[self.nuts_borders['CNTR_CODE'] == 'HU'].total_bounds)
        pad = [-1, -1, 1, 1]
        pad = list(map(lambda x: x * self.config['AGGREGATION']['DEM'] * (self.config['AGGREGATION']['TARGET_SQ_RESOLUTION'] * 2), pad))
        self.padded_bounding_box = tuple([self.bounding_box[i] + pad[i] for i in range(len(self.bounding_box))])

        pop_shp_fn = self.filepath.root_pop_path + self.config['FILES']['POP']
        pop_shp = gpd.read_file(pop_shp_fn)
        xmin, ymin, xmax, ymax = self.padded_bounding_box
        pop_shp = pop_shp.cx[xmin:xmax, ymin:ymax]

        self.padded_bounding_box = tuple(pop_shp.total_bounds)

        return

    def LoadRasterStatistics(self, input_fn, name):
        if not hasattr(self, 'loaded_files'):
            self.loaded_files = {}

        metadata = {}
        with rasterio.open(input_fn) as src:
            metadata['crs'] = src.crs
            metadata['count'] = src.count
            metadata['width'] = src.width
            metadata['height'] = src.height
            metadata['bounds'] = src.bounds

        self.loaded_files[name] = metadata

        return

    def CreateVRT(self, fn, bounds=None, xy_resolution = 100, src_fns = None):
        if not src_fns:
            src_fns = self.filepath.dem_source_fns
        vrt_fn = self.filepath.root_work_path + fn + '.vrt'
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg='cubic',
            xRes = xy_resolution,
            yRes = xy_resolution,
            addAlpha=True,
            outputBounds=bounds
        )
        test_vrt = gdal.BuildVRT(
            vrt_fn,
            src_fns,
            options=vrt_options
        )
        test_vrt = None
        setattr(self.filepath, fn + '_vrt_fn', vrt_fn)
        return

    def CreateFullVRT(self):
        self.CreateVRT('dem_full', xy_resolution=self.config['AGGREGATION']['DEM'])
        self.LoadRasterStatistics(self.filepath.dem_full_vrt_fn, 'dem_full')
        return

    def CreateAggregatedVRT(self, bounds=None):
        self.CreateVRT('dem_aggr', bounds = bounds, xy_resolution=self.config['AGGREGATION']['DEM'])
        self.LoadRasterStatistics(self.filepath.dem_aggr_vrt_fn, 'dem_aggr')
        return

    def CreateAggregatedWWVRT(self, bounds=None):
        self.CreateVRT(
            'ww_aggr',
            bounds = bounds,
            xy_resolution=self.config['AGGREGATION']['DEM'],
            src_fns=self.config['ROOT_LOCATIONS']['WW'] + self.config['FILES']['WW'])
        self.LoadRasterStatistics(self.filepath.ww_aggr_vrt_fn, 'ww_aggr')
        return

    def CreateTIFFromVRT(self, fn = 'dem_aggr_rst'):
        if hasattr(self.filepath, 'dem_aggr_vrt_fn'):
            dem_rst_fn = self.filepath.root_work_path + fn + '.tif'
            gdal.Translate(dem_rst_fn, self.filepath.dem_aggr_vrt_fn)
            setattr(self.filepath, fn + '_fn', dem_rst_fn)
        else:
            raise UserWarning
        return

    def BuildTIFFromVRT(self, in_vrt_fn, out_rst_fn):
        gdal.Translate(out_rst_fn, in_vrt_fn)
        return

    def LoadRasterMetadata(self, rst_fn):
        rst = rasterio.open(rst_fn)
        meta = rst.meta.copy()
        meta.update(compress='lzw')
        self.raster_metadata[rst_fn] = meta
        return

    def CreateBorderRaster(self):
        if not hasattr(self, 'nuts_borders'):
            self.LoadBorderShapefile()

        in_rst_fn = self.filepath.root_work_path + 'dem_aggr_rst.tif'
        if in_rst_fn not in self.raster_metadata:
            self.LoadRasterMetadata(in_rst_fn)
        out_rst_fn = self.filepath.root_work_path + 'nuts_rst.tif'
        with rasterio.open(out_rst_fn, 'w+', **self.raster_metadata[in_rst_fn]) as out_rst:

            out_rst_data = out_rst.read(1)
            shapes = ((geom, value) for geom, value in zip(self.nuts_borders.geometry, self.nuts_borders.shape[0] * [1]) if features.is_valid_geom(geom))

            burned = features.rasterize(
                shapes=shapes,
                fill=0,
                out=out_rst_data,
                transform=out_rst.transform,
                all_touched = True,
                merge_alg = MergeAlg.replace
            )
            out_rst.write_band(1, burned)

            shapes = ((geom, value) for geom, value in zip(self.nuts_borders.geometry, self.nuts_borders.shape[0] * [1]) if features.is_valid_geom(geom))

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

    def CreateBoundedRaster(self):

        self.CreateFullVRT()
        self.GetTargetBoundingBox()

        self.CreateAggregatedVRT(
            bounds=self.padded_bounding_box
        )
        self.CreateTIFFromVRT(fn = 'dem_aggr_rst')
        self.CreateBorderRaster()
        return

    def PreparePopulationShapefile(self):

        data = pd.concat(
            [
                pd.read_csv(
                    self.config['ROOT_LOCATIONS']['POP'] + file,
                    usecols = ['GRD_ID', 'TOT_P'],
                    dtype= {
                        'GRD_ID': pd.StringDtype(),
                        'TOT_P': np.int32
                    }
                )
                for file in self.config['FILES']['POP_CSV']
            ],
            axis=0,
            ignore_index=True
        )

        pop_shp_fn = self.config['ROOT_LOCATIONS']['POP'] + self.config['FILES']['POP']
        pop_shp = gpd.read_file(pop_shp_fn)
        pop_shp = pop_shp.merge(data, on='GRD_ID', how='left')
        data = None
        pop_shp = pop_shp.to_crs(self.config["CRS"])

        if not hasattr(self, 'padded_bounding_box'):
            self.GetTargetBoundingBox()
        xmin, ymin, xmax, ymax = self.padded_bounding_box
        pop_shp = pop_shp.cx[xmin:xmax, ymin:ymax]

        pop_shp.to_file(self.filepath.root_work_path + 'pop_shp.shp')
        return

    def RasterizePopulationShapefile(self):
        pop_shp = gpd.read_file(self.filepath.root_work_path + 'pop_shp.shp')

        in_rst_fn = self.filepath.root_work_path + 'dem_aggr_rst.tif'
        if in_rst_fn not in self.raster_metadata:
            self.LoadRasterMetadata(in_rst_fn)
        out_rst_fn = self.filepath.root_work_path + 'pop_rst.tif'
        with rasterio.open(out_rst_fn, 'w+', **self.raster_metadata[in_rst_fn]) as out_rst:

            out_rst_data = out_rst.read(1)
            shapes = ((geom, value) for geom, value in zip(pop_shp.geometry, pop_shp.TOT_P) if features.is_valid_geom(geom))

            burned = features.rasterize(
                shapes=shapes,
                fill=0,
                out=out_rst_data,
                transform=out_rst.transform,
                all_touched = False,
                merge_alg = MergeAlg.replace
            )
            out_rst.write_band(1, burned)

        return

    def AllocatePopulationToRaster(self):
        self.filepath.dem_aggr_vrt_fn = self.filepath.root_work_path + 'dem_aggr_rst' + '.tif'
        self.LoadRasterMetadata(self.filepath.dem_aggr_vrt_fn)
        pop_shp = gpd.read_file(self.filepath.root_work_path + 'pop_shp.shp')
        rst_shp = gpd.read_file(self.filepath.root_work_path + 'rst_shp.shp')

        # res_intersection = gpd.overlay(rst_shp, pop_shp, how='intersection')
        # res_intersection.to_file(self.filepath.root_work_path + 'intersection_shp.shp')
        res_intersection = gpd.read_file(self.filepath.root_work_path + 'intersection_shp.shp')

        original_size = pop_shp.area[0]
        res_intersection['TOT_P'] = res_intersection['TOT_P'] * (res_intersection.area / original_size)

        out_fn = self.filepath.root_work_path + 'pop_rst.tif'
        with rasterio.open(out_fn, 'w+', **self.raster_metadata) as out_rst:

            out_rst_data = out_rst.read(1)
            shapes = ((geom, value) for geom, value in zip(res_intersection.geometry, res_intersection.shape[0] * [0]) if features.is_valid_geom(geom))

            burned = features.rasterize(
                shapes=shapes,
                fill=0,
                out=out_rst_data,
                transform=out_rst.transform,
                all_touched = True,
                merge_alg = MergeAlg.replace
            )
            out_rst.write_band(1, burned)

            out_rst_data = out_rst.read(1)
            shapes = ((geom, value) for geom, value in zip(res_intersection.geometry, res_intersection.TOT_P) if features.is_valid_geom(geom))

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

    def CreateNonEUCountryRaster(self):
        if not hasattr(self, 'nuts_borders'):
            self.LoadCountryBorderShapefile()

        in_rst_fn = self.filepath.root_work_path + 'dem_aggr_rst.tif'
        if in_rst_fn not in self.raster_metadata:
            self.LoadRasterMetadata(in_rst_fn)
        out_rst_fn = self.filepath.root_work_path + 'cb_rst.tif'
        with rasterio.open(out_rst_fn, 'w+', **self.raster_metadata[in_rst_fn]) as out_rst:

            out_rst_data = out_rst.read(1)
            shapes = ((geom, value) for geom, value in zip(self.country_borders.geometry, self.country_borders.shape[0] * [1]) if features.is_valid_geom(geom))

            burned = features.rasterize(
                shapes=shapes,
                fill=0,
                out=out_rst_data,
                transform=out_rst.transform,
                all_touched = True,
                merge_alg = MergeAlg.replace
            )
            out_rst.write_band(1, burned)

        return

    def CreateBoundedWWRaster(self):

        self.GetTargetBoundingBox()

        self.CreateAggregatedWWVRT(
            bounds=self.padded_bounding_box
        )
        self.BuildTIFFromVRT(
            in_vrt_fn = self.config['ROOT_LOCATIONS']['WORK'] + 'ww_aggr.vrt',
            out_rst_fn = self.config['ROOT_LOCATIONS']['WORK'] + 'ww_aggr_rst.tif'
        )
        return
