from raster_processor import RasterLayerProcessor

if __name__ == "__main__":

    raster_processor = RasterLayerProcessor(
        restart = False
    )
    raster_processor.process_all_layers()
