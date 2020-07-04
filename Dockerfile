FROM python:3.8-slim

RUN apt-get update

RUN apt-get install -y \
    python-numpy \
    gdal-bin \
    libgdal-dev \
    g++

RUN pip install \
    numpy \
    psycopg2 \
    pandas

RUN pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`

RUN pip install \
    rasterio

WORKDIR /app
 
COPY . /app