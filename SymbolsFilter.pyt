# -*- coding: utf-8 -*-

import os
import re
import geopandas as gpd
import pandas as pd
import json
import arcpy
import logging
from pathlib import Path
import shutil

import helpers
import arcpy_logger
import exporter

import importlib

try:
    import openpyxl

    engine = "openpyxl"
    print("openpyxl is available, using it as the engine")
except ImportError:
    engine = None
    print("openpyxl is not available, using the default engine")

importlib.reload(helpers)  # force reload of the module
importlib.reload(arcpy_logger)
importlib.reload(exporter)

sys.dont_write_bytecode = True

DEBUG_MODE = False

DEFAULT_WORKSPACE = r"h:/connections/GCOVERP@osa.sde"


# Get the directory of the .pyt file
toolbox_path = os.path.abspath(__file__)
toolbox_dir = os.path.dirname(toolbox_path)

DEFAULT_SYMBOL_RULES_JSON = os.path.join(toolbox_dir, "layer_symbols_rules.json")
DEFAULT_FILTERED_SYMBOL_FILE = os.path.join(
    toolbox_dir, "output", "filtered_feature_count.xlsx"
)

"""logger = arcpy_logger.get_logger(
    log_level="INFO", logfile_pth=Path(r"H:/SymbolFilter.log"), propagate=False
)"""


log_level = "WARN"

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

fileHandler = logging.FileHandler("{0}/{1}.log".format(toolbox_dir, "SymbolsFilter"))
fileHandler.setFormatter(log_frmt)
logger.addHandler(fileHandler)


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


def get_last_element(s):
    if s is None:
        return s
    elements = s.split(".")
    return elements[-1]


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


def get_query_defn(data):
    sql_query = None
    # TODO: is there more than one sql statement?
    try:
        sql_query = next(item["sql"] for item in data["query_defn"] if "sql" in item)
    except Exception as e:
        logger.warning(f"    Cannot get SQL query: {e}")

    return sql_query


def get_columns(renderer, layername):
    columns = renderer.get("fields")

    if columns is None or None in columns:
        logger.warning(f"No fields found for {layername}: {columns}")
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

    for col in columns_to_convert:
        if col is None or col == "":
            logger.warning(f"Not converting column: {col}")
            continue
        try:
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


def save_to_files(output_path, filtered, drop_null=True, engine=None):
    try:
        data = filtered

        with open(
            output_path.replace(".xlsx", ".json"), "w", encoding="windows-1252"
        ) as f:
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

        # TODO: why do we have to convert from cp252?
        df["Rule"] = df["Rule"].str.encode("windows-1252").str.decode("utf-8")
        df["Layer"] = df["Layer"].str.encode("windows-1252").str.decode("utf-8")

        with pd.ExcelWriter(output_path) as writer:
            if engine:
                df.to_excel(
                    writer,
                    sheet_name="RULES",
                    engine=engine,
                    index=False,
                    encoding="utf-8",
                )
            else:
                df.to_excel(writer, sheet_name="RULES", index=False)

    except Exception as e:
        logger.error(e)
        raise arcpy.ExecuteError


def setup_connection(destination_dir):
    source_file = (
        r"\\v0t0020a.adr.admin.ch\topgisprod\01_Admin\Connections\GCOVERP@osa.sde"
    )
    destination_file = os.path.join(destination_dir, os.path.basename(source_file))

    if os.path.isfile(destination_file):
        return destination_file

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    try:
        shutil.copy2(source_file, destination_file)
    except OSError as e:
        raise arcpy.ExecuteError

    return destination_file


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
        self.description = "Filtering out symbol classes without any object"

    def getParameterInfo(self):
        """Define the tool parameters."""
        # First parameter
        param0 = arcpy.Parameter(
            displayName="Input Perimeter",
            name="in_features",
            datatype="GPFeatureLayer",
            parameterType="Required",
            direction="Input",
        )

        param1 = arcpy.Parameter(
            displayName="Output File (.xlsx)",
            name="out_file",
            datatype="DEFile",
            parameterType="Required",
            direction="Output",
        )

        param0.values = "Mapsheet"
        param1.values = DEFAULT_FILTERED_SYMBOL_FILE

        params = [param0, param1]
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

        from helpers import arcgis_table_to_df

        inLayer = parameters[0].valueAsText
        output_path = parameters[1].valueAsText
        spatial_filter = None
        filtered = {}
        dataset = None
        drop = True

        arcpy.env.workspace = setup_connection(toolbox_dir)

        output_dir = os.path.dirname(output_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        try:
            # Read the mask file (shapefile or GeoJSON)
            spatial_filter = helpers.get_selected_features(inLayer)

            messages.AddMessage(
                f"Found feature: area={spatial_filter.area/1e6} km2, bbox={spatial_filter.extent}"
            )

        except Exception as e:
            logger.error(e)
            messages.addErrorMessage(
                "Layer {0} has no selected features.".format(inLayer)
            )
            raise arcpy.ExecuteError

        # Get the current project and active map
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        active_map = aprx.activeMap
        # List all layers in the active map
        layers = active_map.listLayers()

        rules_dict = {}

        messages.addMessage(f"##### EXTRACTING RULES #####")

        for layer in layers:
            if layer.name == inLayer or not layer.isFeatureLayer:
                continue
            layername = layer.name
            try:
                messages.addMessage(f"Getting symbols rules for '{layername}''")
                attributes = exporter.rules_exporter(layer)
                rules_dict[layername] = attributes
            except Exception as e:
                logger.error(f"Cannot get symbols rules for {layername}: {e}")
                raise arcpy.ExecuteError
        messages.addMessage(f"Writting rules to {DEFAULT_SYMBOL_RULES_JSON}")
        with open(DEFAULT_SYMBOL_RULES_JSON, "w", encoding="utf-8") as f:
            json.dump(rules_dict, f, ensure_ascii=False, indent=4)
            del rules_dict

        try:
            with open(DEFAULT_SYMBOL_RULES_JSON, "r") as f:
                rules_dict = json.load(f)
        except IOError as e:
            messages.addErrorMessage(f"Cannot open {DEFAULT_SYMBOL_RULES_JSON}")
            raise arcpy.ExecuteError

        messages.addMessage(f"##### FILTERING SYMBOLS WITH RULES #####")

        for layername in rules_dict.keys():
            messages.addMessage(f"--- {layername} ---".encode("cp1252"))
            logger.info(f"--- {layername} ---")

            data = rules_dict.get(layername)

            dataset = get_dataset(data)
            renderer = get_renderer(data)

            if dataset is None or renderer is None:
                logger.warning(f"    No dataset found for {layername}")
                continue

            feature_class_path = dataset

            # headers
            values = []
            labels = []
            columns = get_columns(renderer, layername)
            for grp in renderer.get("groups", []):
                values += grp.get("values", [])
                labels += grp.get("labels", [])

            sql = get_query_defn(data)
            messages.addMessage(f"    sql={sql}")

            if columns is None:
                logger.warning(
                    f"    No headings found for {layername}: {columns}. Skipping"
                )
                continue

            # Get the selected features using a search cursor with spatial filter
            selected_features = []

            gdf = None

            # TODO: this should be dynamic
            if "Bedrock_HARMOS" in layername:
                gdf = arcgis_table_to_df(
                    feature_class_path, spatial_filter=spatial_filter, query=sql
                )
                df = arcgis_table_to_df("TOPGIS_GC.GC_BED_FORM_ATT")
                gdf = gdf.merge(df, left_on="FORM_ATT", right_on="UUID")

            features_rules_sum = 0
            if columns is None or any(col is None for col in columns):
                messages.addErrorMessage(
                    f"<null> column are not valid: {columns}. Skipping"
                )
                logger.error(f"<null> column are not valid: {columns}")
                continue
            if gdf is None:
                try:
                    gdf = arcgis_table_to_df(
                        feature_class_path,
                        input_fields=["OBJECTID"] + columns,
                        spatial_filter=spatial_filter,
                        query=sql,
                    )
                except Exception as e:
                    logger.error(
                        f"Error while getting dataframe from layer {layername}: {e}"
                    )
                    continue
            feat_total = str(len(gdf))

            complex_filter_criteria = get_complex_filter_criteria(
                labels, values, columns
            )

            df = gdf

            columns_to_convert = columns

            df = convert_columns(df, columns_to_convert)

            # Store counts and rows for each complex filter criterion
            results = {}

            for label, criteria in complex_filter_criteria:
                logger.debug(f"\nApplying criteria: {label}, {criteria}")

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

                count = len(filtered_df)
                features_rules_sum += count

                if count > 0:
                    count_str = str(count)
                    messages.addMessage(f"{count_str : >10} {label}".encode("cp1252"))
                    results[label] = count

            filtered[layername] = results
            messages.addMessage(
                f"          ----total------\n{feat_total : >10} in selected extent (with query_defn)".encode(
                    "cp1252"
                )
            )
            messages.addMessage(
                f"{features_rules_sum : >10} in classes".encode("cp1252")
            )

        messages.addMessage(f"---- Saving results to {output_path} ----------")

        # TODO: encoding issue
        save_to_files(output_path, filtered, drop_null=True, engine=None)
        messages.addMessage("Done.")

        return

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return
