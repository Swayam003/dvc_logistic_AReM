import os
import pandas as pd
import pickle
import logging
import argparse
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score
from src.utils.all_utils import read_yaml, create_directory,save_metrics_report

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logg"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'logging.log'), level=logging.INFO, format=logging_str,filemode="a")


def read_splitted_data_path(config_path):
    """
    This function returns path to x_test & y_test CSV files
    :param config_path: path to config file
    :return: path to csv files
    """
    try:
        config = read_yaml(config_path)  # reading config file
        logging.info("Read the yaml successfully")
        data_dir = config["data"]['data_dir']
        splitted_data_dir = config["data"]['splitted_data_dir']
        x_test_data_file = config["data"]['x_test_data_file']
        y_test_data_file = config["data"]['y_test_data_file']
        x_test_data_file_path = os.path.join(data_dir, splitted_data_dir, x_test_data_file)  # path: data/splitted_data/x_test.csv
        y_test_data_file_path = os.path.join(data_dir, splitted_data_dir,y_test_data_file)  # path: data/splitted_data/y_test.csv
        return x_test_data_file_path, y_test_data_file_path
    except Exception as e:
        logging.exception(e)
        raise e


def evaluate_metrics(actual_values, predicted_values, predicted_values_probabilities):
    """
    This function calculates various metrics: auc_score, accuracy, recall, precision, f1 score
    :param actual_values: CSV file (y_test.csv)
    :param predicted_values: CSV file (y_predicted.csv)
    :param predicted_values_probabilities:  Probabilities for Predicted values
    :return: auc_score, accuracy, recall, precision, f1
    """

    try:
        y_binary = label_binarize(actual_values, classes=['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking'])
        auc_score = round(roc_auc_score(y_binary, predicted_values_probabilities, average='macro') * 100, 6)
        accuracy = round(accuracy_score(actual_values, predicted_values) * 100, 6)
        recall = round(recall_score(actual_values, predicted_values, average='macro') * 100, 6)
        precision = round(precision_score(actual_values, predicted_values, average='macro') * 100, 6)
        f1 = round(f1_score(actual_values, predicted_values, average='macro') * 100, 6)
        return auc_score, accuracy, recall, precision, f1
    except Exception as e:
        logging.exception("Something went wrong while calculating evaluate metrics:  ",e)
        raise e


def evaluate(config_path):
    """
    This function computes the metrics and saves the metrics file after creating directory for the same file
    :param config_path: str
    """

    try:
        config = read_yaml(config_path)  # reading config file
        logging.info("Read the yaml successfully")
        x_test_data_file_path, y_test_data_file_path = read_splitted_data_path(config_path)  # path to x_test.csv & y_test.csv
        logging.info("Read the path to csv files successfully")

        x_test = pd.read_csv(x_test_data_file_path)
        y_test = pd.read_csv(y_test_data_file_path)

        """# Reading the path and Loading the Scaled_Model file
        model_dir = config["models"]['model_dir']
        scaled_model_dir = config["models"]["scaled_model"]
        scaled_model_file = config['models']['scaled_model_file']
        scaled_model_path = os.path.join(model_dir, scaled_model_dir, scaled_model_file)  # path: models/Standardized_model/StandardScalar_model.sav
        Standardized_model = pickle.load(open(scaled_model_path,'rb'))
        X_Scaled = Standardized_model.transform(x_test) # transforming the x_test dataset"""

        # Reading the path and Loading the Trained_Model file
        model_dir = config["models"]['model_dir']
        trained_model_dir = config["models"]["trained_model_dir"]
        trained_model_file = config['models']['trained_model_file']
        trained_model_path = os.path.join(model_dir, trained_model_dir, trained_model_file)  # path: models/Trained_model/Logistic_model.sav
        trained_Model = pickle.load(open(trained_model_path, 'rb'))

        y_predicted_values = trained_Model.predict(x_test)
        y_predic_values_prob = trained_Model.predict_proba(x_test)
        auc_score, accuracy, recall, precision, f1 = evaluate_metrics(y_test, y_predicted_values, y_predic_values_prob)

        data_dir = config["data"]['data_dir']
        evaluation_metrics_dir = config['data']['evaluation_metrics_dir']
        evaluation_metrics_dir_path = os.path.join(data_dir, evaluation_metrics_dir)  # path: data/evaluation_metrics
        create_directory(dirs=[evaluation_metrics_dir_path])  # creating directory for saving evaluation metrics

        score_file = config['data']['scores']
        score_file_path = os.path.join(evaluation_metrics_dir_path, score_file)  # path: data/evaluation_metrics/scores.json

        scores = { 'auc_score' : auc_score, 'accuracy' : accuracy, 'recall' : recall, 'precision' : precision, 'f1' : f1 }
        save_metrics_report(scores, score_file_path)
        logging.info('Evaluation metrics has been successfully computed')
    except Exception as e:
        logging.exception("Something went wrong during evaluation (Stage 5)",e)
        raise e

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage_05 started")
        evaluate(config_path=parsed_args.config)
        logging.info("stage_05 completed!>>>>>")
    except Exception as e:
        logging.exception(e)
        raise e