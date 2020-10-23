import numpy as np
import rasterio
import tensorflow as tf


def predict_classifier_raster(raster_data, model, stride = 28, bad_value = -1000, channel_n = 1, total_step = 100):
    data = np.zeros((raster_data.shape[0], raster_data.shape[1]))

    for i in range(0, raster_data.shape[0] - stride, total_step):
        for j in range(0, raster_data.shape[1] - stride):
            if i + total_step > raster_data.shape[0] - stride:
                step = raster_data.shape[0] - stride - i
            else:
                step = total_step

            windows = np.stack([raster_data[i + s:i + s + stride, j:j + stride] for s in range(step)], axis=0)
            if channel_n == 1:
                windows = tf.expand_dims(
                    windows,
                    axis=-1
                )
            pred = model.predict(windows)
            data[i + int(round(stride / 2, 0)):i + int(round(stride / 2, 0)) + step, j + int(round(stride / 2, 0))] = np.squeeze(pred)

    return data


def filter_prediction_raster(base, prediction):
    filtered_prediction = np.zeros(base.shape)
    filtered_prediction[base == 1] = prediction[base == 1]
    return filtered_prediction


def write_result_raster(data, src_fn, out_rst_fn, channels = 10):
    if len(data.shape) < 3:
        data = np.expand_dims(data, axis=2)
    data = data.astype('float32')
    channels += 1
    rst = rasterio.open(src_fn)
    meta = rst.meta.copy()

    meta.update(
        dtype=np.float32,
        count=channels,
        compress='lzw')

    with rasterio.open(out_rst_fn, 'w+', **meta) as out_rst:
        for i in range(channels):
            out_rst.write(
                data[:, :, i],
                i + 1
            )

    return
