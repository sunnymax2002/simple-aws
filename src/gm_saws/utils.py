from random import randint
import time
import hashlib
import pickle
import os
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
import yaml

def time_uid() -> int:
    """Generates a time dependent unique-id"""
    ts = int(time.time()) # limit to recent
    randid = randint(0, 511)
    ts = (ts * 64)   # bit-shift << 6
    return (ts * 512) + randid

# REF: https://stackoverflow.com/questions/14023350/cheap-mapping-of-string-to-small-fixed-length-string
def cheaphash(string,length=6):
    if len(string) < length:
        return string
    
    _hash = hashlib.sha256(string.encode('utf-8')).hexdigest()
    if length<len(_hash):
        return _hash[:length]
    else:
        raise Exception("Length too long. Length of {y} when hash length is {x}.".format(x=str(len(hashlib.sha256(string).hexdigest())),y=length))

def GetFullPath(rel_path):
    # Expand rel_path in case a list of relative path elements and then join
    # REF: https://stackoverflow.com/questions/4934806/how-can-i-find-scripts-directory
    base = os.path.abspath(os.path.dirname(sys.path[0]))
    # print('Base directory: ', base, os.getcwd())
    # print('rel_path: ', rel_path)
    # file_dir = os.path.dirname(__file__)
    if(type(rel_path) is list):
        return os.path.join(base, *rel_path)
    else:
        return os.path.join(base, rel_path)
    
def ReadConfig(secrets: bool = False, config_file_path: str = None) -> dict:
    """Returns config read from yaml file, else an empty dict"""
    if config_file_path is None:
        # Import Config
        if secrets:
            fpath = 'config/secrets.yaml'
        else:
            fpath = 'config/app.yaml'
        
        config = yaml.safe_load(open(GetFullPath(fpath)))
    else:
        config = yaml.safe_load(open(config_file_path))

    if config is not None and isinstance(config, dict):
        return config
    else:
        return {}
    
def IsDevEnv():
    return os.path.exists(GetFullPath('config/dev_env'))