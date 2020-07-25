from pymongo import MongoClient
import datetime
client = MongoClient()
client = MongoClient('eugis_mongo', 27017)
db = client.eugis

sample = {
    'row': 1,
    'column': 1,
    'bounds': {
        'lower_left_x': 1,
        'lower_left_y': 1,
        'upper_right_x': 1,
        'upper_right_y': 1
    },
    'insertedUTC': datetime.datetime.utcnow(),
    'files': [
        {
            'raster': 'dem',
            'filenamepath': 'data/test'
        }
    ]
}


def InsertRasterPixelData(position, coordinates, raster_type, filenamepath, slice_id):
    x, y = position
    lower_left_x, lower_left_y, upper_right_x, upper_right_y = coordinates
    raster_pixel = {
        'row': x,
        'column': y,
        'bounds': {
            'lower_left_x': lower_left_x,
            'lower_left_y': lower_left_y,
            'upper_right_x': upper_right_x,
            'upper_right_y': upper_right_y
        },
        'insertedUTC': datetime.datetime.utcnow(),
        'files': [
            {
                'raster': raster_type,
                'filenamepath': filenamepath,
                'slice_id': slice_id
            }
        ]
    }
    raster_pixel_id = db.raster_pixels.insert_one(raster_pixel).inserted_id
    return raster_pixel_id


def RestartRasterCollection():
    db.raster_pixels.drop()
    return
