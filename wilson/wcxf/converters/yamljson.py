import wcxf
import yaml
import json
import logging


def convert_json(stream_in, stream_out):
    try:
        return wcxf.classes._yaml_to_json(stream_in, stream_out, indent=2)
    except yaml.YAMLError:
        logging.error("Input file cannot be parsed as YAML.")
        return 1


def convert_yaml(stream_in, stream_out):
    try:
        return wcxf.classes._json_to_yaml(stream_in, stream_out,
                                          default_flow_style=False)
    except json.decoder.JSONDecodeError:
        logging.error("Input file cannot be parsed as JSON.")
        return 1
