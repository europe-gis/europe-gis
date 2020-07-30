import os
import json
import shutil
import pathlib
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.plot
from rasterio import features
from rasterio.windows import Window, from_bounds
from rasterio.enums import MergeAlg
from matplotlib import pyplot
from osgeo import gdal


class ProcessRasterLayer(object):
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
    def plotArray(array, cmap='viridis'):
        pyplot.imshow(array, cmap=cmap)
        pyplot.show()
        return

    @staticmethod
    def PrintRasterFileStatistics(input_fn):
        with rasterio.open(input_fn) as src:
            # rasterio.plot.show(src)
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

    def __init__(self, root_dem_path, root_nuts_path, root_work_path, restart = False):
        self.root_dem_path = root_dem_path
        self.root_nuts_path = root_nuts_path
        self.root_work_path = root_work_path

        if restart:
            shutil.rmtree(root_work_path, ignore_errors=True)
            # if not os.path.exists(root_work_path):
            os.makedirs(root_work_path)

        self.dem_source_fns = []
        for r, d, f in os.walk(root_dem_path):
            for file in f:
                if file[-4:] == '.TIF':
                    self.dem_source_fns.append(os.path.join(r, file))

        with open(str(pathlib.Path().absolute()).replace('/notebooks', '') + '/config/config.json') as json_file:
            self.config = json.load(json_file)

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

    def CreateVRT(self, fn, bounds=None, xy_resolution = 100):
        vrt_fn = self.root_work_path + fn + '.vrt'
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg='cubic',
            xRes = xy_resolution,
            yRes = xy_resolution,
            addAlpha=True,
            outputBounds=bounds
        )
        test_vrt = gdal.BuildVRT(
            vrt_fn,
            self.dem_source_fns,
            options=vrt_options
        )
        test_vrt = None
        setattr(self, fn + '_vrt_fn', vrt_fn)
        return

    def CreateFullVRT(self):
        self.CreateVRT('dem_full', xy_resolution=self.config['AGGREGATION']['DEM'])
        self.LoadRasterStatistics(self.dem_full_vrt_fn, 'dem_full')
        return

    def CreateAggregatedVRT(self, bounds=None):
        self.CreateVRT('dem_aggr', bounds = bounds, xy_resolution=self.config['AGGREGATION']['DEM'])
        self.LoadRasterStatistics(self.dem_aggr_vrt_fn, 'dem_aggr')
        return

    def CreateTIFFromVRT(self, fn = 'dem_aggr_rst'):
        if hasattr(self, 'dem_aggr_vrt_fn'):
            dem_rst_fn = self.root_work_path + fn + '.tif'
            gdal.Translate(dem_rst_fn, self.dem_aggr_vrt_fn)
            setattr(self, fn + '_fn', dem_rst_fn)
        else:
            raise UserWarning
        return

    def LoadRasterMetadata(self, rst_fn):
        rst = rasterio.open(rst_fn)
        meta = rst.meta.copy()
        meta.update(compress='lzw')
        self.raster_metadata = meta
        return

    def LoadShapefile(self):
        if not hasattr(self, 'raster_metadata'):
            self.LoadRasterMetadata(self.dem_full_vrt_fn)

        shp_fn = self.root_nuts_path + 'ref-nuts-2016-01m.shp/NUTS_RG_01M_2016_3035_LEVL_3.shp'

        counties = gpd.read_file(shp_fn)
        counties["C"] = 1
        counties.to_crs(self.raster_metadata["crs"])
        self.nuts_borders = counties
        return

    def CreateBorderRaster(self):
        if not hasattr(self, 'nuts_borders'):
            self.LoadShapefile()

        out_fn = self.root_work_path + 'nuts_rst.tif'
        with rasterio.open(out_fn, 'w+', **self.raster_metadata) as out_rst:

            out_rst_data = out_rst.read(1)
            shapes = ((geom, value) for geom, value in zip(self.nuts_borders.geometry, self.nuts_borders.C) if features.is_valid_geom(geom))

            burned = features.rasterize(
                shapes=shapes,
                fill=0,
                out=out_rst_data,
                transform=out_rst.transform,
                all_touched = True,
                merge_alg = MergeAlg.replace
            )
            out_rst.write_band(1, burned)

            shapes = ((geom, value) for geom, value in zip(self.nuts_borders.geometry, self.nuts_borders.C) if features.is_valid_geom(geom))

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
        if not hasattr(self, 'nuts_borders'):
            self.LoadShapefile()
        bounding_boxes = self.nuts_borders[self.nuts_borders['CNTR_CODE'] == 'HU'].bounds
        bounds = (min(bounding_boxes['minx']), min(bounding_boxes['miny']), max(bounding_boxes['maxx']), max(bounding_boxes['maxy']))
        return bounds

    @staticmethod
    def RoundBoundingBox(fr, bb, resolution, pad = 0):

        def RoundBound(outer, inner):
            if outer > inner:
                return inner + (outer - inner) % resolution + pad
            else:
                return inner - (inner - outer) % resolution - pad

        result_list = []
        for i in range(len(fr)):
            result_list.append(RoundBound(fr[i], bb[i]))
        return tuple(result_list)

    def CreateBoundedRaster(self):

        self.CreateFullVRT()
        rounded_bounding_box = self.RoundBoundingBox(
            self.loaded_files['dem_full']['bounds'],
            self.GetTargetBoundingBox(),
            self.config['AGGREGATION']['DEM'],
            self.config['AGGREGATION']['DEM'] * (self.config['AGGREGATION']['TARGET_SQ_RESOLUTION'] * 2)
        )
        self.CreateAggregatedVRT(
            bounds=rounded_bounding_box
        )
        self.CreateTIFFromVRT()
        self.LoadRasterMetadata(self.dem_aggr_rst_fn)
        self.CreateBorderRaster()
        return

    def RasterizePopulationShapefile(self):

        root_pop_path = '/mnt/share/mnt/RESEARCH/SATELLITE/GEOSTAT/POP_2011/Version 2_0_1/'

        data = pd.concat(
            [
                pd.read_csv(root_pop_path + 'GEOSTAT_grid_POP_1K_2011_V2_0_1.csv'),
                pd.read_csv(root_pop_path + 'JRC-GHSL_AIT-grid-POP_1K_2011.csv').astype({'TOT_P_CON_DT': 'object'})
            ],
            axis=0,
            ignore_index=True
        )

        pop_shp_fn = root_pop_path + 'GEOSTATReferenceGrid/Grid_ETRS89_LAEA_1K-ref_GEOSTAT_POP_2011_V2_0_1.shp'
        pop_shp = gpd.read_file(pop_shp_fn)
        pop_shp = pop_shp.merge(data, on='GRD_ID', how='left')

        self.dem_aggr_vrt_fn = self.root_work_path + 'dem_aggr_rst' + '.tif'
        self.LoadRasterMetadata(self.dem_aggr_vrt_fn)
        out_fn = self.root_work_path + 'pop_rst.tif'
        with rasterio.open(out_fn, 'w+', **self.raster_metadata) as out_rst:

            out_rst_data = out_rst.read(1)
            shapes = ((geom, value) for geom, value in zip(pop_shp.geometry, pop_shp.TOT_P) if features.is_valid_geom(geom))

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
