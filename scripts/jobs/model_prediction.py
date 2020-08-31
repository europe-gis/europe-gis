import numpy as np
import rasterio
import tensorflow as tf


def PredictAutoencoderRaster(raster_data, model, channels = 10, stride = 28):
    channels += 1
    data = np.zeros((raster_data.shape[0], raster_data.shape[1], channels))

    for i in range(0, raster_data.shape[0] - stride):
        for j in range(0, raster_data.shape[1] - stride):
            window = raster_data[i:i + stride, j:j + stride]
            window = tf.expand_dims(
                tf.expand_dims(
                    window,
                    axis=0
                ),
                axis=-1
            )
            pred = model.encode(window)
            loss = np.sum(
                tf.losses.mean_squared_error(
                    window,
                    model(window)
                )
            )
            data[i, j, 0:-1] = pred
            data[i, j, -1] = loss

    return data


def PredictClassifierRaster(raster_data, model, stride = 28, bad_value = -1000, channel_n = 1, total_step = 100):
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
            # window = raster_data[i:i + stride, j:j + stride]
            # if np.min(window) > -bad_value:
            #     window = tf.expand_dims(
            #         tf.expand_dims(
            #             window,
            #             axis=0
            #         ),
            #         axis=-1
            #     )
            #     pred = model.predict(window)
            #     data[i + int(round(stride / 2, 0)), j + int(round(stride / 2, 0))] = pred

    return data


def WriteResultRaster(data, src_fn, out_rst_fn, channels = 10):
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
