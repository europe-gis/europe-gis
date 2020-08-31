# from scripts.jobs.process_raster_layer import ProcessRasterLayer

# raster_processor = ProcessRasterLayer(
#     restart = False
# )
# raster_processor.CreateBoundedDEMRaster()
# raster_processor.PreparePopulationShapefile()
# raster_processor.RasterizePopulationShapefile()
# raster_processor.CreateNonEUCountryRaster()
# raster_processor.CreateBoundedWWRaster()


# from scripts.jobs.model_training import BorderRecognitionNetwork

# network = BorderRecognitionNetwork(
#     [
#         'ww', 'dem', 'pop', 'nuts'
#     ],
#     image_size=100,
#     nb_labels=2,
#     architecture='resnet'
# )
from scripts.jobs.dataset_creation import CreateTFDatasetFromInMemoryGenerator, InMemoryStridedArrayGenerator
from scripts.jobs.networks.conv_classifier import TrainConvClassifierModel
from scripts.jobs.dataset_creation import ReadRasterFile, PreProcessBorderRaster, PreProcessPopulationRaster, PreProcessLogarithmPopulationRaster, PreprocessForResnet
import scripts.jobs.networks.siamese as networks

raster_ww_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/ww_aggr_rst.tif'
raster_dem_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/dem_aggr_rst.tif'
raster_pop_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/pop_rst.tif'
raster_nuts_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/nuts_rst.tif'

model_name = 'dem_pop_ww_nuts'

rasters = {
    'dem': {
        'type': 'input',
        'data': PreprocessForResnet(ReadRasterFile(raster_dem_fn)),
        'bad_value_threshold': -1000
    },
    'pop': {
        'type': 'input',
        'data': PreprocessForResnet(PreProcessLogarithmPopulationRaster(PreProcessPopulationRaster(ReadRasterFile(raster_pop_fn)))),
        'bad_value_threshold': -1000
    },
    'ww': {
        'type': 'input',
        'data': PreprocessForResnet(ReadRasterFile(raster_ww_fn)),
        'bad_value_threshold': -1000
    },
    'nuts': {
        'type': 'output',
        'data': PreProcessBorderRaster(ReadRasterFile(raster_nuts_fn), bad_value=-1),
        'bad_value_threshold': -1
    }
}
gen = InMemoryStridedArrayGenerator(rasters, window_size = 100)
train_dataset = CreateTFDatasetFromInMemoryGenerator(gen, batch_size = 64, window_size = 100, channel_n = 3)
# internal_model = networks.resnet(['pop'], hidden_layer_size = 100, input_size = 100)
internal_model = networks.simple_resnet(input_size = 100, channel_n = 3)

model, history = TrainConvClassifierModel(
    train_dataset,
    train_dataset,
    num_epochs = 30,
    steps_per_epoch = 10000,
    internal_model = internal_model
)
model.save('/mnt/share/mnt/RESEARCH/SATELLITE/WORK/' + model_name + '_model')
