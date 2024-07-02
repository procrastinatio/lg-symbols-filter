import arcpy
import json
import logging

"""logging.root.handlers = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(r"H:/code/lg-symbols-filter/debug.log"),
        logging.StreamHandler()
    ]
)"""


def rules_exporter(l):
    rules_dict = {"renderer": {}, "query_defn": {}}
    if l.isFeatureLayer:
        sym = l.symbology
        renderer = sym.renderer

        rules_dict["renderer"]["type"] = sym.renderer.type

        ## Layer Validation
        if l.supports("dataSource"):
            rules_dict["dataSource"] = l.dataSource

        if l.supports("DefinitionQuery"):
            # Lists Definition Queries
            dfn = []
            query = l.listDefinitionQueries()
            # List Dictionary
            for dic in query:
                for key, value in dic.items():
                    logging.debug(f"{key}: {value}")
                    dfn.append({key: value})
            rules_dict["query_defn"] = dfn

        fields = {}
        for fld in arcpy.Describe(l).fields:
            fields[fld.aliasName] = fld.name

        if hasattr(renderer, "groups"):
            nb_groups = len(renderer.groups)

            logging.debug(f"GROUPS: {nb_groups}")
            # https://pro.arcgis.com/en/pro-app/latest/arcpy/mapping/uniquevaluerenderer-class.htm
            renderer_dict = {}
            renderer_dict["fields"] = renderer.fields
            renderer_dict["groups"] = []

            for grp in renderer.groups:
                logging.debug(f"New group: {grp.heading}")
                grp_dict = {}

                headings = [v.strip() for v in grp.heading.split(",")]

                grp_dict["headings"] = headings
                logging.debug(f"headings={headings}")
                grp_dict["labels"] = []
                grp_dict["values"] = []
                for itm in grp.items:
                    logging.debug(f"New item")
                    logging.debug(f"   label={itm.label}")
                    logging.debug(f"   values={itm.values}")
                    grp_dict["labels"].append(itm.label)
                    cleaned_list = [
                        [None if item == "<Null>" else item for item in sublist]
                        for sublist in itm.values
                    ]

                    logging.debug(cleaned_list)
                    grp_dict["values"].append(cleaned_list)
                renderer_dict["groups"].append(grp_dict)
            try:
                rules_dict["renderer"] = renderer_dict
                logging.debug(f"--{l.name}--")
                logging.debug(json.dumps(renderer_dict, indent=4))

            except Exception as e:
                logging.error(f"Cannot add symbology: {l.name}: {e}")

        else:
            logging.error(f"Layer {l.name}: {sym.renderer.type}")
            return rules_dict

    return rules_dict
