import tensorflow as tf
import numpy as np
import scripts.jobs.dataset_creation as dataset_creation

from . import model_serving

if __name__ == "__main__":
    model_name = 'pop_dem_ww_nuts_linear'
    window_size = 101

    raster_ww_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/ww_aggr_rst.tif'
    raster_dem_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/dem_aggr_rst.tif'
    raster_pop_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/pop_rst.tif'
    raster_nuts_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/nuts_rst.tif'
    model = tf.keras.models.load_model('/mnt/share/mnt/RESEARCH/SATELLITE/WORK/' + model_name + '_model')

    a = dataset_creation.PreprocessForResnet(dataset_creation.PreProcessLogarithmPopulationRaster(dataset_creation.PreProcessPopulationRaster(dataset_creation.ReadRasterFile(raster_pop_fn))))
    b = dataset_creation.PreprocessForResnet(dataset_creation.PreprocessDEMRaster(dataset_creation.ReadRasterFile(raster_dem_fn)))
    c = dataset_creation.PreprocessForResnet(dataset_creation.PreProcessWWRaster(dataset_creation.ReadRasterFile(raster_ww_fn)))
    a = np.stack([a, b, c], axis = -1).astype(np.float32)

    data = model_serving.predict_classifier_raster(a, model, stride = window_size, channel_n = 3)

    out_rst_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/result_' + model_name + '.tif'
    model_serving.write_result_raster(data, raster_dem_fn, out_rst_fn, channels=0)

    data = model_serving.filter_prediction_raster(dataset_creation.PreProcessBorderRaster(dataset_creation.ReadRasterFile(raster_nuts_fn), bad_value=-1), data)
    out_rst_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/result_comparison_' + model_name + '.tif'
    model_serving.write_result_raster(data, raster_dem_fn, out_rst_fn, channels=0)
