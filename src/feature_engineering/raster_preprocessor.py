import numpy as np
import rasterio

POPULATION_LOG_MAX = 10


def read_raster_file(input_fn):
    with rasterio.open(input_fn) as src:
        return src.read(1)


class DataPreprocessor:
    def __init__(self, raster_config_dict):
        self.raster_config_dict = raster_config_dict
        return

    def preprocess_layer(self, layer_name, is_resnet = True):

        raw_layer_data = read_raster_file(self.raster_config_dict[layer_name]['file'])
        if layer_name == 'pop':
            raw_layer_data[raw_layer_data < 0] = 0
            raw_layer_data[raw_layer_data < 2] = 1
            raw_layer_data = np.log(raw_layer_data)
            raw_layer_data[raw_layer_data > POPULATION_LOG_MAX] = POPULATION_LOG_MAX
            layer_data = raw_layer_data / POPULATION_LOG_MAX
        elif layer_name == 'dem':
            raw_layer_data[raw_layer_data < -1000] = 0
            layer_data = raw_layer_data
        elif layer_name == 'ww':
            raw_layer_data[raw_layer_data == -1000] = -1000
            raw_layer_data[raw_layer_data < 0] = 0
            layer_data = raw_layer_data
        elif layer_name == 'nuts':
            raw_layer_data[raw_layer_data < -1] = -1
            raw_layer_data[raw_layer_data > 2] = 1
            raw_layer_data[raw_layer_data == 2] = 0
            layer_data = raw_layer_data
        else:
            layer_data = raw_layer_data

        if is_resnet:
            bad_value = -1000
            min_value = np.amin(layer_data[layer_data > bad_value])
            max_value = np.amax(layer_data[layer_data > bad_value])
            layer_data[layer_data > bad_value] = 255.999 * (layer_data[layer_data > bad_value] - min_value) / (max_value - min_value)
            layer_data = layer_data.astype(np.uint8)

        return layer_data

    def prepare_inputs(self):
        for layer_name in self.raster_config_dict:
            if self.raster_config_dict[layer_name]['type'] == 'input':
                self.raster_config_dict[layer_name]['data'] = self.preprocess_layer(layer_name)
        return

    def prepare_output_layer(self):
        for layer_name in self.raster_config_dict:
            if self.raster_config_dict[layer_name]['type'] == 'ouput':
                self.raster_config_dict[layer_name]['data'] = self.preprocess_layer(layer_name, is_resnet=False)

        return

    def create_preprocessed_data(self):
        self.prepare_inputs()
        self.prepare_output_layer()
        return self.raster_config_dict
