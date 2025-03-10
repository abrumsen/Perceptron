import pandas as pd
import matplotlib.pyplot as plt
import data_processing as dp

def visualize_dataf(dataframe:pd.DataFrame):
    accuracy = dataframe["Expected_value"] - dataframe["Obtained_value"]
    dataframe["Accuracy"] = accuracy
    dataframe.plot(x="Iteration", y="Accuracy")
    plt.show()


# dp.store_data(dp.add_p_data_to_dataframe(5,[10,20], [20,25], 20, 25), "data.csv")
# dp.demonstration()
# visualize_dataf(dp.load_dataframe_from_file("data.csv"))