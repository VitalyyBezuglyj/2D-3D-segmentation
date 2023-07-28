## Borrowed from https://gitee.com/mindspore/models/tree/master/official/cv/ResNeXt 

"""Parse arguments"""

import os
import ast
import argparse
from pprint import pprint, pformat
import yaml

_config_path = "./config/mipt_5_floor.yaml"

class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()

def add_args_from_cfg(parser: argparse.ArgumentParser, cfg: Config, prefix=''):
    for item in cfg: # type: ignore
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict): # type: ignore
            if isinstance(cfg[item], bool): # type: ignore
                if len(prefix) > 0:
                    parser.add_argument("--" + prefix + '.' + item, type=ast.literal_eval, default=cfg[item]) # type: ignore
                else:
                    parser.add_argument("--" + item, type=ast.literal_eval, default=cfg[item]) # type: ignore
            else:
                if len(prefix) > 0:
                    parser.add_argument("--" + prefix + '.' + item, type=type(cfg[item]), default=cfg[item]) # type: ignore
                else:
                    parser.add_argument("--" + item, type=type(cfg[item]), default=cfg[item]) # type: ignore
        elif isinstance(cfg[item], dict): # type: ignore
            if len(prefix) > 0:
                new_prefix = f"{prefix}.{item}"
            else:  
                new_prefix = item
            add_args_from_cfg(parser, cfg[item], prefix=new_prefix) # type: ignore
    

def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path="default_config.yaml"):
    """
    Parse command line arguments to the configuration according to the default yaml.

    Args:
        parser: Parent parser.
        cfg: Base configuration.
        helper: Helper description.
        cfg_path: Path to the default yaml config.
    """
    parser = argparse.ArgumentParser(description="[REPLACE THIS at config.py]",
                                     parents=[parser])
    add_args_from_cfg(parser, cfg)
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError("At most 3 docs (config description for help, choices) are supported in config yaml")
            print(cfg_helper)
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices

def parse_yaml2(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]

            result_cfg = {}

            for cfg in cfgs:
                result_cfg = {**result_cfg, **cfg}
        except:
            raise ValueError("Failed to parse yaml")
    return result_cfg


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments.

    Args:
        args: Command line arguments.
        cfg: Base configuration.
    """
    args_var = vars(args)
    for item in args_var:
        arr = item.split('.')
        if len(arr) == 3:
            # print(arr)
            # input()
            cfg[arr[0]][arr[1]][arr[2]] = args_var[item]
        elif len(arr) == 2:
            cfg[arr[0]][arr[1]] = args_var[item]
        elif len(arr) == 1:
            cfg[item] = args_var[item]
        else:
            print(arr)
            raise ValueError("Too complex config! Support only 2 level args!")
    return cfg


def get_config():
    """
    Get Config according to the yaml file and cli arguments.
    """
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config_path", type=str, default=_config_path,
                        help="Config file path")
    path_args, _ = parser.parse_known_args() #type: ignore
    config = parse_yaml2(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=config, cfg_path=path_args.config_path)
    final_config = merge(args, config)
    pprint(final_config)
    print("Please check the above information for the configurations", flush=True)
    return Config(final_config)

def get_config_notebook(config_path: str):
    """
    Get Config according to the yaml filefrom ipynb notebook.
    """
    final_config = parse_yaml2(config_path)
    pprint(final_config)
    print("Please check the above information for the configurations", flush=True)
    return Config(final_config)

if __name__ == '__main__':
    config = get_config()