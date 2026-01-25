from sklearn.model_selection import TimeSeriesSplit
from neural_network_class import NeuralNetworkModel
from sklearn.neural_network import MLPRegressor
import math, statistics as stats, copy, random, numpy as np


"""This function runs the Stochastic Gradient Descent (SGD) regression neural network along with a standard
MLP neural network Economic testing data for benchmark testing and reports each model's RMSE for 10 time series splits
on a dataset of a given target variable, along well as the total weighted feature importances of each trial SGD Model."""
def run_evaluation(target_data_file_name, train_size):
    input_data_file_names = ["business_applications_processed.csv", "manufacturing_and_trade_inventories_and_sales_processed.csv", "construction_spending_rate_processed.csv", "advance_retail_and_food_sales_processed.csv", "new_manufacturer_shipments_inventories_and_orders_processed.csv", "international_goods_and_services_trade_processed.csv"]
    input_data, target_data, input_data_months, target_data_months = process_data(input_data_file_names, target_data_file_name, 2005, 2025)
    testing_targets = []
    sgd_model_testing_predictions = []
    adam_model_testing_predictions = []
    lbfgs_model_testing_predictions = []
    tscv = TimeSeriesSplit(n_splits = len(input_data) - train_size, test_size = 1, max_train_size = train_size)
    i = 1
    for feed_indices, testing_indices in tscv.split(input_data):
        testing_index = testing_indices[0]
        print(f"Month {i}")
        print(f"Feed Indices - {feed_indices}")
        print(f"Testing Index - {testing_index}")
        feed_input_data = [input_data[i] for i in feed_indices]
        feed_input_months = [input_data_months[i] for i in feed_indices]
        feed_target_data = [target_data[i] for i in feed_indices]
        feed_target_months = [target_data_months[i] for i in feed_indices]
        testing_input_data = input_data[testing_index]
        testing_input_month = input_data_months[testing_index]
        testing_target_data = target_data[testing_index]
        testing_target_month = target_data_months[testing_index]
        print(f"Feed Inputs - {feed_input_data}")
        print(f"Feed Targets - {feed_target_data}")
        print(f"Testing Inputs - {testing_input_data}")
        print(f"Testing Target - {testing_target_data}")
        print(f"Feed Input Months - {feed_input_months}")
        print(f"Feed Target Months - {feed_target_months}")
        print(f"Testing Input Month - {testing_input_month}")
        print(f"Testing Target Month - {testing_target_month}")
        feed_data_reference = {
        "input_data": feed_input_data,
        "target_data": feed_target_data
        }
        sgd_model = NeuralNetworkModel(**feed_data_reference, hidden_layers = 3, neuron_size_base = 3, random_state = 1)
        lbfgs_model_ = MLPRegressor(solver = "lbfgs", hidden_layer_sizes = (27, 9, 3,), random_state = 1)
        lbfgs_model_.fit(feed_input_data, feed_target_data)
        adam_model_ = MLPRegressor(solver = "adam", hidden_layer_sizes = (27, 9, 3,), random_state = 1)
        adam_model_.fit(feed_input_data, feed_target_data)
        sgd_model_testing_prediction = sgd_model.predict([testing_input_data])[0]
        lbfgs_model_testing_prediction = lbfgs_model_.predict([testing_input_data])[0]
        adam_model_testing_prediction = adam_model_.predict([testing_input_data])[0]
        testing_targets.append(testing_target_data)
        sgd_model_testing_predictions.append(sgd_model_testing_prediction)
        adam_model_testing_predictions.append(adam_model_testing_prediction)
        lbfgs_model_testing_predictions.append(lbfgs_model_testing_prediction)
        print(f"Month {i} Results")
        print(f"Testing Target - {testing_target_data}")
        print(f"SGD Model Prediction - {sgd_model_testing_predictions[-1]}")
        print(f"Adam Model Prediction - {adam_model_testing_predictions[-1]}")
        print(f"LBFGS Model Prediction - {lbfgs_model_testing_predictions[-1]}")
        i += 1
    print("Summary")
    print(f"Testing Targets - {testing_targets}")
    print(f"SGD Model Predictions - {sgd_model_testing_predictions}")
    print(f"Adam Model Predictions - {adam_model_testing_predictions}")
    print(f"LBFGS Model Predictions - {lbfgs_model_testing_predictions}")
    print(f"SGD Model N_RMSE - {np.sqrt(np.mean(np.power(np.array(testing_targets) - np.array(sgd_model_testing_predictions), 2))) / np.mean(np.array(testing_targets))}")
    print(f"Adam Model N_RMSE - {np.sqrt(np.mean(np.power(np.array(testing_targets) - np.array(adam_model_testing_predictions), 2))) / np.mean(np.array(testing_targets))}")
    print(f"LBFGS Model N_RMSE - {np.sqrt(np.mean(np.power(np.array(testing_targets) - np.array(lbfgs_model_testing_predictions), 2))) / np.mean(np.array(testing_targets))}")

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
