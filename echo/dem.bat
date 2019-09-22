raster2pgsql -d -R -I -C -e -Y -F -s 3035 -t 1000x1000 D:\RESEARCH\SATELLITE\DEM\*.tif demelevation | psql -U postgres -d postgis_25_sample
