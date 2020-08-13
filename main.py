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

import pickle
import numpy as np
from scripts.jobs.dataset_creation import ReadRasterFile, CreateStridedArray, CreateTFDataset
from scripts.jobs.model_training import TrainModel

raster_fn = '/mnt/share/mnt/RESEARCH/SATELLITE/WORK/pop_rst.tif'

a = ReadRasterFile(raster_fn)
a[a < 2] = 1
a = np.log(a)
a[a > 10] = 10
a = a / 10
input_array = CreateStridedArray(a)

with open(r"test.pickle", "wb") as output_file:
    pickle.dump(input_array, output_file)
with open(r"test.pickle", "rb") as input_file:
    input_array = pickle.load(input_file)

train_dataset, test_dataset = CreateTFDataset(input_array)
model = TrainModel(train_dataset, test_dataset)
