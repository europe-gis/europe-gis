raster2pgsql -d -R -I -C -e -Y -F -s 3035 -t 1000x1000 D:\RESEARCH\SATELLITE\WW\WAW_2015_020m_eu_03035_d06_Full\*.tif ww | psql -U postgres -d postgis_25_sample
