import pandas as pd
import matplotlib.pyplot as plt
import data_processing as dp

def visualize_dataf(dataframe:pd.DataFrame):
    """
    This first function is a TEST function, it compares the difference between
    the Expected value and the Obtained value of each iteration during the training
    of the perceptron and put it into a plot.
    :param dataframe: The dataframe to be analysed
    :return: returns None
    """
    accuracy = dataframe["Expected_value"] - dataframe["Obtained_value"]
    dataframe["Accuracy"] = accuracy.abs()
    dataframe.plot(x="Iteration", y="Accuracy")
    plt.show()


# dp.store_data(dp.add_p_data_to_dataframe(5,[10,20], [20,25], 20, 25), "data.csv")
# dp.generate_random_data("./data.csv", 15)
# print(dp.load_dataframe_from_file("data.csv"))
# visualize_dataf(dp.load_dataframe_from_file("data.csv"))