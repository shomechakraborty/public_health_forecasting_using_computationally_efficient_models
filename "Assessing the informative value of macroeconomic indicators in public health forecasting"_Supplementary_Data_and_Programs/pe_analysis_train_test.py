from sklearn.model_selection import TimeSeriesSplit
from neural_network_class import NeuralNetworkModel
from sklearn.neural_network import MLPRegressor
import math, statistics as stats, copy, random, numpy as np


"""This function runs the Stochastic Gradient Descent (SGD) regression neural network along with a standard
MLP neural network Economic testing data for benchmark testing and reports each model's RMSE for 10 time series splits
on a dataset of a given target variable, along well as the total weighted feature importances of each trial SGD Model."""
def run_evaluation(target_data_file_name, train_proportion):
    input_data_file_names = ["business_applications_processed.csv", "manufacturing_and_trade_inventories_and_sales_processed.csv", "construction_spending_rate_processed.csv", "advance_retail_and_food_sales_processed.csv", "new_manufacturer_shipments_inventories_and_orders_processed.csv", "international_goods_and_services_trade_processed.csv"]
    input_data, target_data, input_data_months, target_data_months = process_data(input_data_file_names, target_data_file_name, 2005, 2025)
    feed_input_data = input_data[:int(len(input_data) * train_proportion)]
    feed_target_data = target_data[:int(len(target_data) * train_proportion)]
    testing_input_data = input_data[int(len(input_data) * train_proportion):]
    testing_target_data = target_data[int(len(target_data) * train_proportion):]
    feed_input_months = [input_data_months[0], input_data_months[int(len(input_data_months) * 0.8) - 1]]
    feed_target_months = [target_data_months[0], target_data_months[int(len(target_data_months) * 0.8) - 1]]
    testing_input_months = [input_data_months[int(len(input_data_months) * 0.8)], input_data_months[-1]]
    testing_target_months = [target_data_months[int(len(target_data_months) * 0.8)], target_data_months[-1]]
    print(f"Feed Inputs - {feed_input_data}")
    print(f"Feed Targets - {feed_target_data}")
    print(f"Testing Inputs - {testing_input_data}")
    print(f"Testing Targets - {testing_target_data}")
    print(f"Feed Input Months - {feed_input_months}")
    print(f"Feed Target Months - {feed_target_months}")
    print(f"Testing Input Months - {testing_input_months}")
    print(f"Testing Target Months - {testing_target_months}")
    feed_data_reference = {
    "input_data": feed_input_data,
    "target_data": feed_target_data
    }
    sgd_model = NeuralNetworkModel(**feed_data_reference, hidden_layers = 3, neuron_size_base = 3, random_state = 1)
    lbfgs_model_ = MLPRegressor(solver = "lbfgs", hidden_layer_sizes = (27, 9, 3,), random_state = 1)
    lbfgs_model_.fit(feed_input_data, feed_target_data)
    adam_model_ = MLPRegressor(solver = "adam", hidden_layer_sizes = (27, 9, 3,), random_state = 1)
    adam_model_.fit(feed_input_data, feed_target_data)
    sgd_model_testing_predictions = sgd_model.predict(testing_input_data)
    lbfgs_model_testing_predictions = lbfgs_model_.predict(testing_input_data)
    adam_model_testing_predictions = adam_model_.predict(testing_input_data)
    print(f"Testing Targets - {testing_target_data}")
    print(f"SGD Model Predictions - {sgd_model_testing_predictions}")
    print(f"Adam Model Predictions - {adam_model_testing_predictions}")
    print(f"LBFGS Model Predictions - {lbfgs_model_testing_predictions}")
    print(f"SGD Model N_RMSE - {np.sqrt(np.mean(np.power(np.array(testing_target_data) - np.array(sgd_model_testing_predictions), 2))) / np.mean(np.array(testing_target_data))}")
    print(f"Adam Model N_RMSE - {np.sqrt(np.mean(np.power(np.array(testing_target_data) - np.array(adam_model_testing_predictions), 2))) / np.mean(np.array(testing_target_data))}")
    print(f"LBFGS Model N_RMSE - {np.sqrt(np.mean(np.power(np.array(testing_target_data) - np.array(lbfgs_model_testing_predictions), 2))) / np.mean(np.array(testing_target_data))}")


"""This model processes Economic datasets and returns the respective input and target
datasets to be used for testing between the neural network models."""
def process_data(input_data_file_names, target_data_file_name, start_year, end_year):
    random.seed(42)
    input_data = []
    input_data_months = []
    target_data = []
    target_data_months = []
    months = ((end_year - start_year) * 12) + 1
    for i in range(months):
        input_data.append([])
    for i in range(len(input_data_file_names)):
        with open(input_data_file_names[i]) as file:
            j = 0
            for line in file:
                data_line = line.strip().split(",")
                if data_line[0] == "" or data_line[0] == "Period":
                    continue
                year = int(data_line[0][-4:])
                month = data_line[0][:-5]
                if year < start_year or year > end_year or (year == end_year and month != "Jan"):
                    continue
                if i == 0:
                    input_data_months.append((year, month))
                input_data[j].append(float(data_line[1]))
                j += 1
    with open(target_data_file_name) as file:
        for line in file:
            data_line = line.strip().split(",")
            if data_line[0] == "observation_date":
                continue
            year = int(data_line[0][:4])
            month = int(data_line[0][5:7])
            if year < start_year or year > end_year or (year == start_year and month == 1) or (year == end_year and month != 1 and month != 2):
                continue
            target_data_months.append((year, month))
            target_data.append(float(data_line[1]))
    return input_data, target_data, input_data_months, target_data_months
