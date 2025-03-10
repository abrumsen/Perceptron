import pandas as pd
import numpy as np
import os

def store_data(dataframe:pd.DataFrame, file_path:str):
    """
    Creates or completes a csv based on the supplied dataframe
    :param dataframe: Pandas dataframe containing the various data generated by the perceptron during training.
    :param file_path: Path to the csv file containing the data to be stored.
    :return: returns None
    """
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, index_col= 0,sep=";")
        frames = [dataframe, df]
        big_dataframe = pd.concat(frames, axis=0)
    else:
        big_dataframe = dataframe
    big_dataframe = big_dataframe.sort_values(by="Iteration")
    big_dataframe.to_csv(file_path, index=True, sep=";", mode="w")

def add_p_data_to_dataframe(iteration:int, weights:list, variables:list, obtained_value:float,expected_value:float):
    """
    Retrieves the various data generated by the perceptron and stores them in a pandas dataframe.
    :param iteration:
    :param weights:
    :param variables:
    :param obtained_value:
    :param expected_value:
    :return:
    """
    p_data = {"Iteration" : [iteration], "Weights":[weights], "Variables":[variables], "Obtained_value": obtained_value, "Expected_value": expected_value}
    dtf = pd.DataFrame(p_data)
    return dtf

def load_dataframe_from_file(file_name:str="data.csv"):
    loaded_df = pd.read_csv(file_name, index_col=0, sep=";")
    return loaded_df

def demonstration():
    store_data(add_p_data_to_dataframe(1, [5, 7], [4, 8], 12.5, 10.6), "./data.csv")
    store_data(add_p_data_to_dataframe(2, [6, 8], [9, 8], 13.5, 11.6), "./data.csv")
    store_data(add_p_data_to_dataframe(3, [6, 8], [9, 8], 13.5, 11.6), "./data.csv")
    store_data(add_p_data_to_dataframe(4, [6, 8], [9, 8], 13.5, 11.6), "./data.csv")
    load_dataframe_from_file()

