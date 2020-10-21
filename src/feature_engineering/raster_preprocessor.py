import numpy as np
import rasterio

POPULATION_LOG_MAX = 10
LAYER_BAD_VALUE = -1000


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
            raw_layer_data[raw_layer_data < LAYER_BAD_VALUE] = 0
            layer_data = raw_layer_data
        elif layer_name == 'ww':
            raw_layer_data[raw_layer_data == LAYER_BAD_VALUE] = LAYER_BAD_VALUE
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
            min_value = np.amin(layer_data[layer_data > LAYER_BAD_VALUE])
            max_value = np.amax(layer_data[layer_data > LAYER_BAD_VALUE])
            layer_data[layer_data > LAYER_BAD_VALUE] = 255.999 * (layer_data[layer_data > LAYER_BAD_VALUE] - min_value) / (max_value - min_value)
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
            elif self.raster_config_dict[layer_name]['type'] == 'aux':
                self.raster_config_dict[layer_name]['data'] = self.preprocess_layer(layer_name, is_resnet=False)
        return

    def create_validation_layer(self):
        validation_layer = {}
        validation_layer['type'] = 'validation'

        input_rasters = []
        for layer_name in self.raster_config_dict:
            if self.raster_config_dict[layer_name]['type'] == 'input':
                raster_shape = self.raster_config_dict[layer_name]['data'].shape
                input_rasters.append(self.raster_config_dict[layer_name]['data'])
            else:
                output_raster = self.raster_config_dict[layer_name]['data']
        input_data = np.stack(input_rasters, axis = -1)
        validation_layer_data = np.zeros(raster_shape)
        for i in range(0, raster_shape[0] - self.window_size):
            for j in range(0, raster_shape[1] - self.window_size):
                if (np.amin(input_data[i:i + self.window_size, j:j + self.window_size]) > LAYER_BAD_VALUE) & (np.amin(output_raster[i:i + self.window_size, j:j + self.window_size]) > -1):
                    validation_layer_data[i][j] = 1

        validation_layer['data'] = validation_layer_data
        self.raster_config_dict['validation'] = validation_layer
        return

    def remove_aux_layers(self):
        for layer_name in self.raster_config_dict:
            if self.raster_config_dict[layer_name]['type'] == 'aux':
                self.raster_config_dict.pop(layer_name, None)
        return

    def create_preprocessed_data(self):
        self.prepare_inputs()
        self.prepare_output_layer()
        self.create_validation_layer()
        self.remove_aux_layers()
        return self.raster_config_dict
