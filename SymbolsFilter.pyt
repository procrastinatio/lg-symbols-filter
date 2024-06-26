# -*- coding: utf-8 -*-

import os
import re
import geopandas as gpd
import pandas as pd
import json
import arcpy
import logging
from pathlib import Path

import helpers
import arcpy_logger

import importlib

importlib.reload(helpers)  # force reload of the module
importlib.reload(arcpy_logger)

sys.dont_write_bytecode = True

DEBUG_MODE = False

DEFAULT_WORKSPACE = r"h:/connections/GCOVERP@osa.sde"


# Get the directory of the .pyt file
toolbox_path = os.path.abspath(__file__)
toolbox_dir = os.path.dirname(toolbox_path)

DEFAULT_SYMBOL_RULES_JSON = os.path.join(toolbox_dir, "layer_symbols_rules.json")
DEFAULT_FILTERED_SYMBOL_FILE = os.path.join(toolbox_dir, "output", "filtered_feature_count.xlsx")

"""logger = arcpy_logger.get_logger(
    log_level="INFO", logfile_pth=Path(r"H:/SymbolFilter.log"), propagate=False
)"""

log_str_lst = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "WARN", "FATAL"]
log_int_lst = [0, 10, 20, 30, 40, 50]

log_level = "WARNING"

# set logging level
if isinstance(log_level, str):
    log_level = getattr(logging, log_level)


logger = logging.getLogger("filter_symbols")
while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])
logger.propagate = False
logger.setLevel(log_level)
log_frmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ah = arcpy_logger.ArcpyHandler()
ah.setFormatter(log_frmt)
logger.addHandler(ah)


def convert_to_int(x):
    if x == "<Null>" or x is None:
        return 0
    else:
        return int(x)


def clean_headings(headings):
    if headings is None or None in headings:
        logger.error(f"No headings {headings}")
        return None
    else:
        headings = list(map(get_last_element, headings))
    logger.debug(f"    headings={headings}")

    return headings


"""def filter_from_criteria(data, gdf):
    headings = data.get("headings")

    values = data.get("values")
    labels = data.get("labels")

    headings = clean_headings(headings)

    filter_criteria = zip(labels, values)

    filters = []

    for criterion in filter_criteria:
        label, values = criterion
        # Create the filter expression dynamically

        for value in values:
            filter_expression = pd.Series([True] * len(gdf))
            for i, head in enumerate(headings):
                filter_expression = filter_expression & (
                    gdf[head] == convert_to_int(value[i])
                )
        filters.append(filter_expression)

    return filters"""


def get_last_element(s):
    if s is None:
        return s
    elements = s.split(".")
    return elements[-1]


"""def process_layer(layername, gdf, data, all_value=True):
    results = {}
    logger.info(f"-----{layername}--------")

    headings = data.get("headings")
    logger.debug(headings)

    values = data.get("values")
    labels = data.get("labels")

    if headings is None or None in headings:
        logger.warning(f"No headings found for {layername}: {headings}")
        return results
    else:
        headings = list(map(get_last_element, headings))
    logger.debug(headings)
    if headings:
        logger.debug(headings)
        logger.debug(f"Before cleanup: {gdf.columns}")
        gdf = gdf[headings]
        logger.debug(f"After cleanup: {gdf.columns}")

    # Check if conversion is possible and convert:

    for col in headings:
        if (
            gdf[col]
            .dropna()
            .apply(lambda x: isinstance(x, float) and x.is_integer())
            .all()
        ):
            gdf[col] = gdf[col].fillna(0).astype(int)

    gdf = gdf.fillna(0.0).astype(int)




    filters = filter_from_criteria(data, gdf)

    filter_criteria = zip(labels, values, filters)

    for filter_criterion in filter_criteria:
        logging.info(f"\nApplying criteria: {filter_criterion}")
        label, values, filter_expression = filter_criterion

        # Apply the filter
        filtered_df = gdf[filter_expression]

        # Store the count and the matching rows
        results[label] = len(filtered_df)

        if len(filtered_df) > 0:
            logger.info(f"    {label}: {len(filtered_df)}")

    return {"rules": results}"""


def get_dataset(data):
    dataset = None
    datasource = data.get("dataSource")

    m = re.findall(",Dataset=(.*)", datasource)
    if m and len(m) > 0:
        dataset = m[0]  # .split(".").pop()

    return dataset


def get_renderer(data):
    renderer = None
    try:
        renderer = data.get("renderer")
    except Exception as e:
        logger.warning(f"    Cannot get renderer for {layername}: {e}")
    return renderer


def get_columns(renderer, layername):
    columns = renderer.get("headings")

    if columns is None or None in columns:
        logger.warning(f"No headings found for {layername}: {columns}")
    else:
        columns = list(map(get_last_element, columns))
    return columns


def get_complex_filter_criteria(labels, values, columns):
    # Initialize the complex filter criteria list
    complex_filter_criteria = []

    # Iterate over the list of value sets and labels
    for label, value_set in zip(labels, values):
        for value_group in value_set:
            # Create a list of (column, value) pairs
            criteria = [
                (col, convert_to_int(val)) for col, val in zip(columns, value_group)
            ]  # TODO: if val is not None]
            # Add the criteria to the complex filter list along with the label
            complex_filter_criteria.append((label, criteria))

    # Print the complex filter criteria
    logger.debug("Complex Filter Criteria with Labels:")
    for label, criteria in complex_filter_criteria:
        logger.debug(f"Label: {label}, Criteria: {criteria}")

    return complex_filter_criteria


def convert_columns(df, columns_to_convert):
    # Check if conversion is possible and convert
    try:
        for col in columns_to_convert:
            if (
                df[col]
                .dropna()
                .apply(lambda x: isinstance(x, float) and x.is_integer())
                .all()
            ):
                # Fill NaN values with 0 (or another specific value) before conversion
                df[col] = df[col].fillna(0).astype(int)
    except KeyError as ke:
        logger.error(f"Key error while converting column {col}: {ke}")
    except Exception as e:
        logger.error(f"Unknown error: {e}")

    return df


def save_to_files(output_path, filtered, drop_null=True):
    try:
        data = filtered  # results["layers"]

        with open(output_path.replace(".xlsx", ".json"), "w", encoding="utf-8") as f:
            # Serialize the data and write it to the file
            json.dump(filtered, f, ensure_ascii=False, indent=4)
    except Exception as e:
        messages.addErrorMessage(e)
        logger.error(e)

    try:
        flattened_data = [
            (k1, k2, v) for k1, subdict in data.items() for k2, v in subdict.items()
        ]

        # Convert to a DataFrame
        df = pd.DataFrame(flattened_data, columns=["Layer", "Rule", "Count"])
        if drop_null:
            df = df[df.Count != 0]

        with pd.ExcelWriter(output_path) as writer:
            df.to_excel(writer, sheet_name="RULES")

    except Exception as e:
        logger.error(e)
        raise arcpy.ExecuteError


class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Geocover"
        self.alias = "Geocover"

        # List of tool classes associated with this toolbox
        self.tools = [SymbolFilter]


class SymbolFilter:
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "SymbolFilter"
        self.description = ""

    def getParameterInfo(self):
        """Define the tool parameters."""
        # First parameter
        param0 = arcpy.Parameter(
            displayName="Input Features",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )

        # Second parameter
        param1 = arcpy.Parameter(
            displayName="Input File",
            name="in_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Input",
        )

        param2 = arcpy.Parameter(
            displayName="Output File",
            name="out_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Output",
        )

        param0.values = "Mapsheet"
        param1.values = DEFAULT_SYMBOL_RULES_JSON
        param2.values = DEFAULT_FILTERED_SYMBOL_FILE

        params = [param0, param1, param2]
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # from filter_symbols import process_layers_symbols

        # from export_symbol_rules import arcgis_table_to_df  # Twice imported
        from utils import arcgis_table_to_df  # Twice imported

        # from helpers import process_layer

        inLayer = parameters[0].valueAsText
        inSymbolsFile = parameters[1].valueAsText
        output_path = parameters[2].valueAsText
        spatial_filter = None
        filtered = {}
        dataset = None
        drop = True

        arcpy.env.workspace = DEFAULT_WORKSPACE

        try:
            # Read the mask file (shapefile or GeoJSON)
            spatial_filter = helpers.get_selected_features(inLayer)
            # Assuming the mask is a single geometry, you can dissolve to create a single unified geometry
            # mask_geom = mask_gdf.unary_union

        except Exception as e:
            logger.error(e)
            messages.addErrorMessage(
                "Layer {0} has no selected features.".format(inLayer)
            )
            raise arcpy.ExecuteError

        try:
            with open(inSymbolsFile, "r") as f:
                layers = json.load(f)
        except IOError as e:
            messages.addErrorMessage(f"Cannot open {inSymbolsFile}")
            raise arcpy.ExecuteError

        for layername in layers.keys():
            messages.addMessage(f"--- {layername} ---".encode("cp1252"))
            data = layers.get(layername)

            dataset = get_dataset(data)
            renderer = get_renderer(data)

            if dataset is None or renderer is None:
                logger.warning(f"    No dataset found for {layername}")
                continue

            feature_class_path = dataset

            # headers
            columns = get_columns(renderer, layername)
            values = renderer.get("values")
            labels = renderer.get("labels")

            if columns is None:
                logger.warning(f"No headings found for {layername}: {columns}")
                continue

            # Get the selected features using a search cursor with spatial filter
            selected_features = []

            gdf = None

            # TODO: this should be dynamic
            if "Bedrock_HARMOS" in layername:
                gdf = arcgis_table_to_df(
                    feature_class_path, spatial_filter=spatial_filter
                )
                df = arcgis_table_to_df("TOPGIS_GC.GC_BED_FORM_ATT")
                logger.debug(df)
                logger.debug(gdf)
                gdf = gdf.merge(df, left_on="FORM_ATT", right_on="UUID")
                logger.debug(f"     ====== MERGING")
                logger.debug(gdf)

            if not "toto" in layername:  # "Quelle" in layername:
                if columns is None or any(col is None for col in columns):
                    logger.error(f"<null> column are not valid: {columns}")
                    continue
                if gdf is None:
                    try:
                        gdf = arcgis_table_to_df(
                            feature_class_path,
                            input_fields=["OBJECTID"] + columns,
                            spatial_filter=spatial_filter,
                            # query= "KIND IN (11901001,12501002,12501003,12501004,12501006,12101006,13601001,13601002 ,13601003,13701001,13701002,13701004,13801003,13801004,13801005,13801006,14601004) AND (PRINTED = 1 OR PRINTED IS NULL)"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error while getting dataframe fro layer {layername}: {e}"
                        )
                        continue
                feat_total = str(len(gdf))

                messages.addMessage(
                    f"{feat_total : >10} objects in selected feature".encode("cp1252")
                )

                complex_filter_criteria = get_complex_filter_criteria(
                    labels, values, columns
                )

                df = gdf

                columns_to_convert = columns

                df = convert_columns(df, columns_to_convert)

                # Dictionary to store counts and rows for each complex filter criterion
                results = {}

                for label, criteria in complex_filter_criteria:
                    logger.info(f"\nApplying criteria: {label}, {criteria}")

                    # Start with a True series to filter
                    filter_expression = pd.Series([True] * len(df), index=df.index)

                    for column, value in criteria:
                        # Update the filter expression for each (column, value) pair
                        filter_expression &= df[column] == value
                        logger.debug(f"Filter status for ({column} == {value}):")
                        logger.debug(filter_expression)
                        logger.debug(f"Matching rows count: {filter_expression.sum()}")

                    # Apply the final filter to the DataFrame
                    filtered_df = df[filter_expression]

                    """results[label] = {
                        "count": len(filtered_df),
                        "rows": filtered_df.to_json(orient='records') , # filtered_df,
                        "criteria": criteria,
                    }"""

                    count = len(filtered_df)

                    if count > 0:
                        count_str = str(count)
                        messages.addMessage(
                            f"{count_str : >10} {label}".encode("cp1252")
                        )
                        results[label] = count

                # Print the results

                """for label, result in results.items():
                    logger.info(f"\nFilter Label: {label}")
                    logger.info(f"Criteria: {result['criteria']}")
                    logger.info(f"Count: {result['count']}")
                    logger.info("Matching Rows:")
                    logger.info(result["rows"])"""
                logger.info("---")

                filtered[layername] = results

        messages.addMessage(f"---- Saving results to {output_path} ----------")

        save_to_files(output_path, filtered, drop_null=True)

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return
