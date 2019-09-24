import rasterio
from rasterio.windows import Window
from matplotlib import pyplot


def writeArrayToFile(array, dst_file, src_profile):
    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = src_profile  # src.profile

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw'
        )
        with rasterio.open(dst_file, 'w', **profile) as dst:
            dst.write(array.astype(rasterio.uint8), 1)
    return


def printRasterFileStats(file):
    with rasterio.open(file) as src:
        print(src.crs)
        print(src.count)
        print(src.width)
        print(src.height)
        print(src.bounds)
    return


def loadWindowOfRasterFile(file, window):
    with rasterio.open(file) as src:
        w = src.read(1, window=window)
    return w


def plotArray(array):
    pyplot.imshow(array, cmap='pink')
    pyplot.show()
    return
