FROM ubuntu:18.04 as builder

ARG DESTDIR="/build"

RUN apt-get update -y \
    && apt-get install -y --fix-missing --no-install-recommends \
            software-properties-common build-essential ca-certificates \
            make cmake wget unzip libtool automake \
            zlib1g-dev libsqlite3-dev pkg-config sqlite3 libcurl4-gnutls-dev \
            libtiff5-dev git

WORKDIR /PROJ

RUN git clone --single-branch --branch 7.1 https://github.com/OSGeo/PROJ/ /PROJ

RUN ./autogen.sh \
    && ./configure --prefix=/usr \
    && make -j$(nproc) \
    && make install

FROM ubuntu:18.04 as runner

RUN date

RUN apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libsqlite3-0 libtiff5 libcurl4 libcurl3-gnutls \
        wget ca-certificates

RUN \
    mkdir -p /usr/share/proj; \
    wget --no-verbose --mirror https://cdn.proj.org/; \
    rm -f cdn.proj.org/*.js; \
    rm -f cdn.proj.org/*.css; \
    mv cdn.proj.org/* /usr/share/proj/; \
    rmdir cdn.proj.org

COPY --from=builder  /build/usr/share/proj/ /usr/share/proj/
COPY --from=builder  /build/usr/include/ /usr/include/
COPY --from=builder  /build/usr/bin/ /usr/bin/
COPY --from=builder  /build/usr/lib/ /usr/lib/

RUN apt-get install -y \
    python-numpy \
    g++ \
    python3-pip \
    git

RUN pip3 install cython

RUN pip3 install \
    numpy

RUN pip3 install \
    pandas \
    pymongo \
    notebook

ENV PROJ_DIR /usr

RUN apt-get install -y libgeos-dev

RUN apt-get install -y \
    gdal-bin \
    libgdal-dev

RUN pip3 install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`

RUN pip3 install git+https://github.com/pyproj4/pyproj.git

RUN pip3 install \
    geopandas

RUN pip3 install \
    rasterio

# RUN apt-get install libproj-dev proj-data proj-bin  
# RUN apt-get install libgeos-dev  
# RUN pip install cython  

WORKDIR /app

COPY . /app