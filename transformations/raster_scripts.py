import rasterio


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
