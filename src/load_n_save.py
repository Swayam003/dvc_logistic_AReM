from src.utils.all_utils import read_yaml, create_directory,save_local_df
import argparse
import pandas as pd
import os
import logging

#logging
logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logg"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"logging.log"),level=logging.INFO, format=logging_str, filemode="a")

def get_data(config_path):
    """
    This function saves dataset after merging multiple csv files from various folders into the local directory
    :param config_path:
    """
    # read the config
    config = read_yaml(config_path)
    logging.info("Read the yaml successfully")

    data_path = config["data_source"]["s3_source"]

    # create path to directory: data/raw/final_data.csv
    data_dir = config["data"]['data_dir']
    raw_local_dir = config["data"]['raw_local_dir']
    raw_local_dir_path = os.path.join(data_dir, raw_local_dir)
    create_directory(dirs=[raw_local_dir_path])

    # create path for file
    raw_local_file = config["data"]['raw_local_file']
    raw_local_file_path = os.path.join(raw_local_dir_path, raw_local_file)

    # Merging files and saving Merged dataset
    df = pd.DataFrame()
    for dirpath, dirnames, files in os.walk(data_path):
        for file_name in files:
            if file_name.endswith('.csv'):
                current_data = pd.read_csv(dirpath + "/" + file_name, encoding="ISO-8859-1", skiprows=4,error_bad_lines=False)
                current_data['label'] = dirpath[11:]
                df = pd.concat([df, current_data])
    df.rename(columns={'# Columns: time': 'time'}, inplace=True)
    save_local_df(df,raw_local_file_path)          # saving the dataframe to CSV file
    logging.info("Successfully merged data")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    try:
        logging.info(">>>>> stage_01 started")
        get_data(config_path=parsed_args.config)
        logging.info("stage_01 completed!>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e