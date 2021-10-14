import yaml
import os
import logging
import pickle

def read_yaml(path_to_yaml: str) -> dict:
    """
    It reads a path to yaml file and returns a dict
    :param path_to_yaml:str
    :return dict:dict
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
        return content
    except Exception as e:
        logging.exception("error while loading yaml file:  ",e)
        raise e

def create_directory(dirs: list):
    """
    It is used to create a directory
    :param dirs: list
    """
    try:
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"directory is created at {dir_path}")
            print(f"directory is created at {dir_path}")
    except Exception as e:
        logging.exception("error while creating directory:   ", e)
        raise e

def save_local_df(data, data_path, index_status=False):
    """
    This function saves the dataframe to CSV File
    :param data: dataframe
    :param data_path: path to save dataframe
    :param index_status: default: False
    """
    try:
        data.to_csv(data_path, index=index_status)
        logging.info(f"data is saved at {data_path}")
    except Exception as e:
        logging.exception("error while saving dataframe to csv file: ", e)
        raise e

def save_models(model, model_file_path: str):
    """
    This function is used to save model files
    :param model: model object
    :param model_file_path: model file path
    """
    try:
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model is saved at {model_file_path}")
    except Exception as e:
        logging.exception("error while saving model:  ",e)
        raise e