import os
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
from src.utils.all_utils import read_yaml, create_directory,save_local_df, save_models
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logg"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'logging.log'), level=logging.INFO, format=logging_str,filemode="a")

def read_splitted_data_path(config_path):
    """
    This function returns path to x_train & y_train CSV files
    :param config_path: path to config file
    :return: path to csv files
    """
    try:
        config = read_yaml(config_path)  # reading config file
        logging.info("Read the yaml successfully")
        data_dir = config["data"]['data_dir']
        splitted_data_dir = config["data"]['splitted_data_dir']
        x_train_data_file = config["data"]['x_train_data_file']
        y_train_data_file = config["data"]['y_train_data_file']
        x_train_data_file_path = os.path.join(data_dir, splitted_data_dir, x_train_data_file)  # path: data/splitted_data/x_train.csv
        y_train_data_file_path = os.path.join(data_dir, splitted_data_dir,y_train_data_file)  # path: data/splitted_data/y_train.csv
        return x_train_data_file_path, y_train_data_file_path
    except Exception as e:
        logging.exception(e)
        raise e

def training_model(config_path, params_path):
    """
    This function trains a model and saves the same trained model after creating directory for the same model
    :param config_path: path to config file
    :param params_path: path to params file
    """
    try:
        config = read_yaml(config_path)  # reading config file
        params = read_yaml(params_path)  # reading params file
        logging.info("Read the yaml successfully")
        x_train_data_file_path, y_train_data_file_path = read_splitted_data_path(config_path)  # path to x_train.csv & y_train.csv
        logging.info("Read the path to csv files successfully")

        model_dir = config["models"]['model_dir']
        trained_model_dir = config["models"]["trained_model_dir"]
        trained_model_dir_path = os.path.join(model_dir, trained_model_dir)  # path: models/Trained_model
        create_directory(dirs=[trained_model_dir_path])

        trained_model_file = config['models']['trained_model_file']
        trained_model_path = os.path.join(trained_model_dir_path, trained_model_file)  # path: models/Trained_model/Logistic_model.sav

        x_train = pd.read_csv(x_train_data_file_path)
        y_train = pd.read_csv(y_train_data_file_path)

        multi_class = params["Logistic_model_params"]["multi_class"]
        solver = params["Logistic_model_params"]["solver"]
        random_state = params["Logistic_model_params"]["random_state"]

        model = LogisticRegression(multi_class=multi_class, solver=solver,random_state= random_state)
        model.fit(x_train, y_train)
        logging.info("LogisticRegression Model has been trained successfully!")
        save_models(model, trained_model_path)
    except Exception as e:
        logging.exception("Something went wrong while training the model n saving the data:  ", e)
        raise e

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info(">>>>> stage_04 started")
        training_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage_04 completed>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e
