import arcpy
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from arcgis.geometry import Geometry


def get_selected_features(layer, esri_geom=True):
    # Get the selected feature IDs as strings
    selected_oids = set(map(str, arcpy.Describe(layer).FIDSet.split(";")))

    # Get the selected features using a search cursor
    selected_features = [
        row
        for row in arcpy.da.SearchCursor(layer, ["OID@", "SHAPE@"])
        if str(row[0]) in selected_oids
    ]

    # Ensure there is a selected feature
    if not selected_features:
        raise ValueError("No features selected")

    # Get the geometry of the first selected feature
    selected_geometry = selected_features[0][1]

    if esri_geom:
        return selected_geometry

    else:
        # Convert the ESRI JSON to a Geometry object
        esri_json_str = selected_geometry.JSON
        esri_geometry = Geometry(esri_json_str)

        # Convert the ESRI Geometry to GeoJSON
        geojson_dict = esri_geometry.__geo_interface__

        # Convert the GeoJSON dictionary to a shapely geometry
        shapely_geometry = shape(geojson_dict)

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame([{"geometry": shapely_geometry}], crs="EPSG:4326")

        # Display the GeoDataFrame
        return gdf
