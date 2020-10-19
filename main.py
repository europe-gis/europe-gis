
# from scripts.jobs.model_training import BorderRecognitionNetwork

# network = BorderRecognitionNetwork(
#     [
#         'ww', 'dem', 'pop', 'nuts'
#     ],
#     image_size=100,
#     nb_labels=2,
#     architecture='resnet'
# )
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# import scripts.jobs.dataset_creation as dataset_creation
# import scripts.jobs.model_prediction as model_prediction

# raster_ww_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/ww_aggr_rst.tif'
# raster_dem_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/dem_aggr_rst.tif'
# raster_pop_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/pop_rst.tif'
# raster_nuts_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/nuts_rst.tif'

# model_name = 'pop_dem_ww_nuts_logistic'
# window_size = 21
# training_size = 100000

# rasters = {
#     'pop': {
#         'type': 'input',
#         'data': dataset_creation.PreprocessForResnet(dataset_creation.PreProcessLogarithmPopulationRaster(dataset_creation.PreProcessPopulationRaster(dataset_creation.ReadRasterFile(raster_pop_fn)))),
#         'bad_value_threshold': -1000
#     },
#     'dem': {
#         'type': 'input',
#         'data': dataset_creation.PreprocessForResnet(dataset_creation.PreprocessDEMRaster(dataset_creation.ReadRasterFile(raster_dem_fn))),
#         'bad_value_threshold': -1000
#     },
#     'ww': {
#         'type': 'input',
#         'data': dataset_creation.PreprocessForResnet(dataset_creation.PreProcessWWRaster(dataset_creation.ReadRasterFile(raster_ww_fn))),
#         'bad_value_threshold': -1000
#     },
#     'nuts': {
#         'type': 'output',
#         'data': dataset_creation.PreProcessBorderRaster(dataset_creation.ReadRasterFile(raster_nuts_fn), bad_value=-1),
#         'bad_value_threshold': -1
#     }
# }

# full_generator_sequence = dataset_creation.create_generator_sequence(rasters, window_size = window_size)
# sequence_separator = dataset_creation.SequenceSeparator(rasters, window_size = window_size)
# generator_sequences = sequence_separator(full_generator_sequence)
# generator_sequence = generator_sequences[0] + generator_sequences[1] * 5

# log_regr = LogisticRegression(warm_start=True, random_state=0, verbose = 1, solver = 'saga', max_iter = 200, n_jobs = 4)

# train_dataset_generator = dataset_creation.InMemoryStridedArrayGeneratorForLogisticRegression(rasters, window_size = window_size, generator_sequence = generator_sequence, batch_size = training_size)
# for i in range(1):
#     X, y = next(train_dataset_generator)
#     log_regr.fit(X, y)
#     print(log_regr.score(X[y == 0], y[y == 0]))
#     print(log_regr.score(X[y == 1], y[y == 1]))


# raster_ww_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/ww_aggr_rst.tif'
# raster_dem_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/dem_aggr_rst.tif'
# raster_pop_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/pop_rst.tif'
# raster_nuts_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/nuts_rst.tif'
# a = dataset_creation.PreprocessForResnet(dataset_creation.PreProcessLogarithmPopulationRaster(dataset_creation.PreProcessPopulationRaster(dataset_creation.ReadRasterFile(raster_pop_fn))))
# b = dataset_creation.PreprocessForResnet(dataset_creation.PreprocessDEMRaster(dataset_creation.ReadRasterFile(raster_dem_fn)))
# c = dataset_creation.PreprocessForResnet(dataset_creation.PreProcessWWRaster(dataset_creation.ReadRasterFile(raster_ww_fn)))
# a = np.stack([a, b, c], axis = -1)

# data = model_prediction.PredictLogisticRegressionRaster(a, log_regr, stride = window_size, channel_n = 3)

# out_rst_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/result_' + model_name + '.tif'
# model_prediction.WriteResultRaster(data, raster_dem_fn, out_rst_fn, channels=0)
