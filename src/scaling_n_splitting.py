import os
import pandas as pd
import argparse
import logging
from src.utils.all_utils import read_yaml, create_directory,save_local_df, save_models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logg"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'logging.log'), level=logging.INFO, format=logging_str,filemode="a")


def read_processed_data_path(config_path):
    """
    his function returns path to CSV file
    :param config_path: path to config file
    :return: path to config file
    """
    try:
        config = read_yaml(config_path)  # reading config file
        logging.info("Read the yaml successfully")
        data_dir = config["data"]['data_dir']
        preprocess_data_dir = config["data"]['raw_local_dir']
        preprocess_data_file = config["data"]['raw_local_file']
        preprocess_data_file_path = os.path.join(data_dir, preprocess_data_dir, preprocess_data_file)  # path: data/preprocess_data/preprocess_Data.csv
        return preprocess_data_file_path
    except Exception as e:
        logging.exception(e)
        raise e

def scaling(config_path):
    """
    This function reads CSV file and producing Standard Scalled Model after creating directory for the same model
    :param config_path: path to config file
    :return: x_scaled: Returns scaled dataset
    """

    try:
        config = read_yaml(config_path)  # reading config file
        logging.info("Read the yaml successfully")
        processed_data_file_path = read_processed_data_path(config_path)  # path to preprocess_Data.csv
        logging.info("Read the path to csv file successfully")
        model_dir = config["models"]['model_dir']
        scaled_model_dir = config["models"]["scaled_model"]
        scaled_model_dir_path = os.path.join(model_dir, scaled_model_dir)  # path: models/Standardized_model
        create_directory(dirs=[scaled_model_dir_path])

        scaled_model_file = config['models']['scaled_model_file']
        scaled_model_path = os.path.join(scaled_model_dir_path, scaled_model_file)  # path: models/Standardized_model/StandardScalar_model.sav
        df = pd.read_csv(processed_data_file_path)
        x = df.drop(columns=['label', 'time'])
        y = df['label']

        scalar = StandardScaler()  #Scaling the data
        x_scaled = scalar.fit_transform(x)
        Xscaled = pd.DataFrame(x_scaled,columns=['avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23'])
        logging.info(f'Standard Scaling has been successfully applied')

        save_models(scalar, scaled_model_path)
        return Xscaled,y
    except Exception as e:
        logging.exception(e)
        raise e

def splitting_n_saving(config_path,params_path):
    """
    This function reads CSV file and splits the data into X_train,X_test,Y_train,Y_test after creating directory for saving the splitted dataset
    :param config_path: path to config file
    """

    try:
        Xscaled,y = scaling(config_path)
        config = read_yaml(config_path)  # reading config file
        params = read_yaml(params_path)  # reading params file
        logging.info("Read the yaml successfully")

        data_dir = config["data"]['data_dir']
        splitted_data_dir = config['data']['splitted_data_dir']
        splitted_data_dir_path = os.path.join(data_dir, splitted_data_dir)  # path: data/splitted_data
        create_directory(dirs=[splitted_data_dir_path])  # creating directory for saving splitted data

        x_train_data_file = config["data"]["x_train_data_file"]
        x_test_data_file = config["data"]["x_test_data_file"]
        y_train_data_file = config["data"]["y_train_data_file"]
        y_test_data_file = config["data"]["y_test_data_file"]
        xtrain_data_file_path = os.path.join(splitted_data_dir_path, x_train_data_file)  # path: data/splitted_data/x_train.csv
        xtest_data_file_path = os.path.join(splitted_data_dir_path, x_test_data_file)    # path: data/splitted_data/x_test.csv
        ytrain_data_file_path = os.path.join(splitted_data_dir_path, y_train_data_file)  # path: data/splitted_data/y_train.csv
        ytest_data_file_path = os.path.join(splitted_data_dir_path, y_test_data_file)    # path: data/splitted_data/y_test.csv

        split_ratio = params["base"]["test_size"]
        random_state = params["base"]["random_state"]

        x_train, x_test, y_train, y_test = train_test_split(Xscaled, y, test_size = split_ratio, random_state=random_state)
        logging.info("Splitting of the dataet has been done successfully")
        save_local_df(x_train,xtrain_data_file_path)
        save_local_df(x_test,xtest_data_file_path)
        save_local_df(y_train,ytrain_data_file_path)
        save_local_df(y_test,ytest_data_file_path)
    except Exception as e:
        logging.exception("Something went wrong while splitting n saving the data:  ",e)
        raise e

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info(">>>>> stage_03 started")
        splitting_n_saving(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage_03 completed!>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e