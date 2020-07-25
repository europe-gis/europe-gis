UPDATE gpopgrid
SET gpop=cast(gpop."TOT_P" as int)
FROM gpop
WHERE gpop."GRD_ID" = gpopgrid.grd_id;

UPDATE gpopgrid
SET gpop=cast(gpop1."TOT_P" as int)
FROM gpop1
WHERE gpop1."GRD_ID" = gpopgrid.grd_id;
