CREATE TABLE raster_gpop AS
WITH poprast AS (
  SELECT rid, ST_MapAlgebra(
                ST_Union(ST_AsRaster(geom, rast, '32BF', gpop, -9999)),
                ST_AddBand(ST_MakeEmptyRaster(rast), '32BF'::text, -9999, -9999), 
                '[rast1]', '32BF', 'SECOND') rast
  FROM gpop, demelevation
  WHERE ST_Intersects(geom, rast)
GROUP BY rid, rast
)
SELECT a.rid,
       CASE
         WHEN b.rid IS NULL THEN ST_AddBand(ST_MakeEmptyRaster(a.rast), '32BF'::text, -9999, -9999)
         ELSE b.rast
       END rast
FROM demelevation a LEFT OUTER JOIN poprast b 
ON a.rid = b.rid;