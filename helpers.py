import arcpy
import geopandas as gpd
import pandas as pd
from collections import OrderedDict
from shapely.geometry import shape
from arcgis.geometry import Geometry

import logging


def arcgis_table_to_df(in_fc, input_fields=None, query="", spatial_filter=None):
    """Function will convert an arcgis table into a pandas dataframe with an object ID index, and the selected
    input fields using an arcpy.da.SearchCursor.
    :param - in_fc - input feature class or table to convert
    :param - input_fields - fields to input to a da search cursor for retrieval
    :param - query - sql query to grab appropriate values
    :returns - pandas.DataFrame"""
    OIDFieldName = arcpy.Describe(in_fc).OIDFieldName
    available_fields = [field.name for field in arcpy.ListFields(in_fc)]
    logging.debug(f"Available fields: {available_fields}")
    logging.debug(f"Input fields: {input_fields}")
    if input_fields:
        # Preserve order of the 'input_fields'
        final_fields = list(
            OrderedDict.fromkeys(
                item for item in input_fields if item in available_fields
            )
        )
    else:
        final_fields = available_fields
    logging.debug(f"intersection: {final_fields}")
    data = [
        row
        for row in arcpy.da.SearchCursor(
            in_fc,
            final_fields,
            where_clause=query,
            spatial_filter=spatial_filter,
            search_order="SPATIALFIRST",
        )
    ]
    fc_dataframe = pd.DataFrame(data, columns=final_fields)
    fc_dataframe = fc_dataframe.set_index(OIDFieldName, drop=True)
    return fc_dataframe


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
