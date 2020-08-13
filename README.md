# europe-gis
Processing GIS data for Europe

# saving virtual environment
- `pip freeze > requirements.txt` - save list of packages to file

```
docker build . -t eugis/eugis:latest
docker push eugis/eugis:latest

```



# Processing steps
- define bounding box
- create virtual raster
- process real raster
- get the size of the dataset
- create empty hdf file
- loop through the raster and process images
- store meta information in mongo
- create random batches

# Training steps
- define input generator
- define network
- run hyperparameter optimization
- train model
- create prediction raster layer
- visualize results
