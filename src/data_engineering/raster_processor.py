import os
import json
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.plot
import rasterio.env
from rasterio import features
from rasterio.enums import MergeAlg
from osgeo import gdal


class FilePath:
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


class RasterLayerProcessor:
    def __init__(self, restart = False):

        with open('config.json') as json_file:
            self.config = json.load(json_file)

        self.filepath = FilePath(self.config)

        if restart:
            shutil.rmtree(self.filepath.root_work_path, ignore_errors=True)
            os.makedirs(self.filepath.root_work_path)

        self.raster_metadata = {}

    def load_nuts_border_shapefile(self, level = 3):

        shp_fn = self.filepath.root_nuts_path + self.config['FILES'][f'NUTS{level}']
        if not hasattr(self, 'nuts_borders'):
            self.nuts_borders = {}
        self.nuts_borders[level] = gpd.read_file(shp_fn).to_crs(self.config['CRS'])
        return

    def load_country_border_shapefile(self):

        shp_fn = self.config['ROOT_LOCATIONS']['CB'] + self.config['FILES']['CB']
        self.country_borders = gpd.read_file(shp_fn).to_crs(self.config['CRS'])
        self.country_borders = self.country_borders[~self.country_borders['ISO2'].isin(self.config['COUNTRY_CODE_LIST'])]

        return

    def get_target_bounding_box(self):
        self.load_nuts_border_shapefile()
        self.bounding_box = tuple(self.nuts_borders[3][self.nuts_borders[3]['CNTR_CODE'] == 'HU'].total_bounds)
        pad = [-1, -1, 1, 1]
        pad = list(map(lambda x: x * self.config['AGGREGATION']['DEM'] * (self.config['AGGREGATION']['TARGET_SQ_RESOLUTION'] * 2), pad))
        self.padded_bounding_box = tuple([self.bounding_box[i] + pad[i] for i in range(len(self.bounding_box))])

        pop_shp_fn = self.filepath.root_pop_path + self.config['FILES']['POP']
        pop_shp = gpd.read_file(pop_shp_fn)
        xmin, ymin, xmax, ymax = self.padded_bounding_box
        pop_shp = pop_shp.cx[xmin:xmax, ymin:ymax]

        self.padded_bounding_box = tuple(pop_shp.total_bounds)

        return

    def load_raster_statistics(self, input_fn, name):
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

    def create_vrt(self, fn, bounds=None, xy_resolution = 100, src_fns = None):
        if not src_fns:
            src_fns = self.filepath.dem_source_fns
        vrt_fn = self.filepath.root_work_path + fn + '.vrt'
        vrt_options = gdal.BuildVRTOptions(
            resampleAlg = gdal.GRA_Average,  # 'cubic',
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

    def create_dem_full_vrt(self):
        self.create_vrt(
            'dem_full',
            xy_resolution=self.config['AGGREGATION']['DEM']
        )
        self.load_raster_statistics(self.filepath.dem_full_vrt_fn, 'dem_full')
        return

    def create_dem_aggr_vrt(self, bounds=None):
        self.create_vrt(
            'dem_aggr',
            bounds = bounds,
            xy_resolution=self.config['AGGREGATION']['DEM']
        )
        self.load_raster_statistics(self.filepath.dem_aggr_vrt_fn, 'dem_aggr')
        return

    def create_ww_aggr_vrt(self, bounds=None):
        self.create_vrt(
            'ww_aggr',
            bounds = bounds,
            xy_resolution=self.config['AGGREGATION']['DEM'],
            src_fns=self.config['ROOT_LOCATIONS']['WW'] + self.config['FILES']['WW']
        )
        self.load_raster_statistics(self.filepath.ww_aggr_vrt_fn, 'ww_aggr')
        return

    def build_tiff_from_vrt(self, fn = 'dem_aggr_rst'):
        if hasattr(self.filepath, 'dem_aggr_vrt_fn'):
            dem_rst_fn = self.filepath.root_work_path + fn + '.tif'
            gdal.Translate(dem_rst_fn, self.filepath.dem_aggr_vrt_fn)
            setattr(self.filepath, fn + '_fn', dem_rst_fn)
        else:
            raise UserWarning
        return

    def load_raster_metadata(self, rst_fn):
        rst = rasterio.open(rst_fn)
        meta = rst.meta.copy()
        meta.update(compress='lzw')
        self.raster_metadata[rst_fn] = meta
        return

    def build_nuts_border_raster(self):
        for level in range(0, 4):
            self.load_nuts_border_shapefile(level = level)

            in_rst_fn = self.filepath.root_work_path + 'dem_aggr_rst.tif'
            if in_rst_fn not in self.raster_metadata:
                self.load_raster_metadata(in_rst_fn)
            out_rst_fn = self.filepath.root_work_path + f'nuts_rst{level}.tif'
            with rasterio.open(out_rst_fn, 'w+', **self.raster_metadata[in_rst_fn]) as out_rst:

                out_rst_data = out_rst.read(1)
                shapes = ((geom, value) for geom, value in zip(self.nuts_borders[level].geometry, self.nuts_borders[level].shape[0] * [1]) if features.is_valid_geom(geom))

                burned = features.rasterize(
                    shapes=shapes,
                    fill=0,
                    out=out_rst_data,
                    transform=out_rst.transform,
                    all_touched = True,
                    merge_alg = MergeAlg.replace
                )
                out_rst.write_band(1, burned)

                shapes = ((geom, value) for geom, value in zip(self.nuts_borders[level].geometry, self.nuts_borders[level].shape[0] * [1]) if features.is_valid_geom(geom))

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

    def build_area_raster(self):
        for level in range(0, 4):
            self.load_nuts_border_shapefile(level = level)

            in_rst_fn = self.filepath.root_work_path + 'dem_aggr_rst.tif'
            if in_rst_fn not in self.raster_metadata:
                self.load_raster_metadata(in_rst_fn)
            out_rst_fn = self.filepath.root_work_path + f'nuts_area_rst{level}.tif'
            with rasterio.open(out_rst_fn, 'w+', **self.raster_metadata[in_rst_fn]) as out_rst:

                out_rst_data = out_rst.read(1)
                shapes = ((geom, value) for geom, value in zip(self.nuts_borders[level].geometry, self.nuts_borders[level].index) if features.is_valid_geom(geom))

                burned = features.rasterize(
                    shapes=shapes,
                    fill=0,
                    out=out_rst_data,
                    transform=out_rst.transform,
                    all_touched = True,
                    merge_alg = MergeAlg.replace
                )
                out_rst.write_band(1, burned)

                shapes = ((geom, value) for geom, value in zip(self.nuts_borders[level].geometry, self.nuts_borders[level].index) if features.is_valid_geom(geom))

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

    def build_bounded_dem_raster(self):

        self.create_dem_full_vrt()
        self.get_target_bounding_box()

        self.create_dem_aggr_vrt(
            bounds=self.padded_bounding_box
        )
        self.build_tiff_from_vrt(fn = 'dem_aggr_rst')
        return

    def prepare_population_shapefile(self):

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

        self.get_target_bounding_box()
        xmin, ymin, xmax, ymax = self.padded_bounding_box
        pop_shp = pop_shp.cx[xmin:xmax, ymin:ymax]

        pop_shp.to_file(self.filepath.root_work_path + 'pop_shp.shp')
        return

    def rasterize_population_shapefile(self):
        pop_shp = gpd.read_file(self.filepath.root_work_path + 'pop_shp.shp')

        in_rst_fn = self.filepath.root_work_path + 'dem_aggr_rst.tif'
        if in_rst_fn not in self.raster_metadata:
            self.load_raster_metadata(in_rst_fn)
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

    def build_non_country_border_raster(self):
        self.load_country_border_shapefile()

        in_rst_fn = self.filepath.root_work_path + 'dem_aggr_rst.tif'
        if in_rst_fn not in self.raster_metadata:
            self.load_raster_metadata(in_rst_fn)
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

    def build_bounded_ww_raster(self):

        self.get_target_bounding_box()
        self.create_ww_aggr_vrt(
            bounds=self.padded_bounding_box
        )
        gdal.Translate(
            self.config['ROOT_LOCATIONS']['WORK'] + 'ww_aggr_rst.tif',
            self.config['ROOT_LOCATIONS']['WORK'] + 'ww_aggr.vrt'
        )
        return

    def build_population_raster(self):
        self.prepare_population_shapefile()
        self.rasterize_population_shapefile()
        return

    def process_all_layers(self):
        self.build_bounded_dem_raster()
        self.build_nuts_border_raster()
        self.build_population_raster()
        self.build_non_country_border_raster()
        self.build_bounded_ww_raster()
        self.build_area_raster()
        return
