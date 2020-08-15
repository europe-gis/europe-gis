import numpy as np
import rasterio
import tensorflow as tf


def PredictRaster(raster_data, model, channels = 10):
    stride = 28
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


def WriteResultRaster(data, src_fn, out_rst_fn, channels = 10):
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
