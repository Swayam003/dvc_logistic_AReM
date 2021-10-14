import pandas as pd
import os
from src.utils.all_utils import read_yaml, create_directory, save_local_df
import argparse
import logging
from pandas_profiling import ProfileReport

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logg"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'logging.log'), level=logging.INFO, format=logging_str,filemode="a")

def read_raw_data(config_path):
    """
    his function returns path to CSV file
    :param config_path: path to config file
    :return: path to config file
    """
    try:
        config = read_yaml(config_path)  # reading config file
        logging.info("Read the yaml successfully")
        data_dir = config["data"]['data_dir']
        raw_local_dir = config["data"]['raw_local_dir']
        raw_local_file = config["data"]['raw_local_file']
        raw_local_file_path = os.path.join(data_dir, raw_local_dir, raw_local_file)  # path: data/raw/final_data.csv
        return raw_local_file_path
    except Exception as e:
        logging.exception(e)
        raise e

def EDA(config_path):
    """
    This function reading CSV file and producing Report file after creating directory for the same report file
    :param config_path: path to config file
    """
    try:
        config = read_yaml(config_path)  # reading config file
        logging.info("Read the yaml successfully")
        raw_local_file_path = read_raw_data(config_path) # path to final_data.csv

        data_dir = config["data"]['data_dir']
        EDA_report_dir = config['data']['EDA_report_dir']
        EDA_report_dir_path = os.path.join(data_dir, EDA_report_dir)   # path: data/EDA_report
        create_directory(dirs=[EDA_report_dir_path])       #creating directory for EDA report

        EDA_report_file = config['data']['EDA_file']
        report_file_path = os.path.join(EDA_report_dir_path,EDA_report_file)  # path: data/EDA_report/profiling_report.html
        df = pd.read_csv(raw_local_file_path)
        report = ProfileReport(df)          # ProfileReport
        report.to_file(output_file=report_file_path)
        logging.info(f'Pandas Profiling report has been generated successfully to {report_file_path}')
    except Exception as e:
        logging.exception(e)
        raise e

def preprocessing(config_path):
    """
    This function do all the preprocessing tasks for final_data.csv file and saving the processed data in CSV file
    :param config_path: path to config file
    """
    try:
        raw_local_file_path = read_raw_data(config_path)  # path to final_data.csv
        config = read_yaml(config_path)  # reading config file
        logging.info("Read the yaml successfully")

        data_dir = config["data"]['data_dir']
        preprocess_data_dir = config['data']['preprocess_data_dir']
        preprocess_data_dir_path = os.path.join(data_dir, preprocess_data_dir)  # path: data/preprocess_data
        create_directory(dirs=[preprocess_data_dir_path])  # creating directory for preprocess data

        preprocess_data_file = config['data']['preprocess_data_file']
        preprocess_data_file_path = os.path.join(preprocess_data_dir_path, preprocess_data_file)  # path: data/preprocess_data/preprocess_Data.csv

        df = pd.read_csv(raw_local_file_path)   # reading final_data.csv file

        # Handling Skewness in the data
        q = df['avg_rss12'].quantile(0.02)
        data_cleaned = df[df['avg_rss12'] > q] # we are removing the bottom 2% data from the avg_rss12 column
        q = df['var_rss12'].quantile(0.95)
        data_cleaned = data_cleaned[data_cleaned['var_rss12'] < q] # we are removing the top 5% data from the var_rss12 column
        q = df['avg_rss13'].quantile(0.99)
        data_cleaned = data_cleaned[data_cleaned['avg_rss13'] < q] # we are removing the top 1% data from the avg_rss13 column
        q = df['avg_rss13'].quantile(0.99)
        data_cleaned = data_cleaned[data_cleaned['avg_rss23'] < q] # we are removing the top 1% data from the avg_rss23 column
        q = df['var_rss13'].quantile(0.95)
        data_cleaned = data_cleaned[data_cleaned['var_rss13'] < q] # we are removing the top 5% data from the var_rss13 column
        q = df['var_rss23'].quantile(0.95)
        data_cleaned = data_cleaned[data_cleaned['var_rss23'] < q] #we are removing the top 5% data from the var_rss23 column

        save_local_df(data_cleaned,preprocess_data_file_path)   # saving processed data
    except Exception as e:
        logging.exception(e)
        raise e

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage_02 started")
        EDA(config_path=parsed_args.config)
        preprocessing(config_path=parsed_args.config)
        logging.info("stage_02 completed!>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e