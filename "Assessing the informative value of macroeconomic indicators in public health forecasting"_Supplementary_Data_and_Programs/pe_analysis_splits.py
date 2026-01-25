from sklearn.model_selection import TimeSeriesSplit
from neural_network_class import NeuralNetworkModel
from sklearn.neural_network import MLPRegressor
import math, statistics as stats, copy, random, numpy

"""This function runs the Stochastic Gradient Descent (SGD) regression neural network along with a standard
MLP neural network Economic testing data for benchmark testing and reports each SGD Model's RMSE for 10 time series splits
on a dataset of a given target variable, along well as the total weighted feature importances of each trial SGD sgd_model."""
def run_evaluation(target_data_file_name, train_time, test_time):
    input_data_file_names = ["business_applications_processed.csv", "manufacturing_and_trade_inventories_and_sales_processed.csv", "construction_spending_rate_processed.csv", "advance_retail_and_food_sales_processed.csv", "new_manufacturer_shipments_inventories_and_orders_processed.csv", "international_goods_and_services_trade_processed.csv"]
    input_data, target_data, input_data_months, target_data_months = process_data(input_data_file_names, target_data_file_name, 2005, 2025)
    testing_targets_across_splits = []
    sgd_model_testing_predictions_across_splits = []
    lbfgs_model_testing_predictions_across_splits = []
    adam_model_testing_predictions_across_splits = []
    sgd_model_maes_across_splits = []
    lbfgs_model_maes_across_splits = []
    adam_model_maes_across_splits = []
    sgd_model_rmses_across_splits = []
    lbfgs_model_rmses_across_splits = []
    adam_model_rmses_across_splits = []
    sgd_model_nrmses_across_splits = []
    lbfgs_model_nrmses_across_splits = []
    adam_model_nrmses_across_splits = []
    tscv = TimeSeriesSplit(n_splits = int((len(input_data) - train_time) / test_time), test_size = test_time, max_train_size = train_time)
    i = 1
    for feed_indices, testing_indices in tscv.split(input_data):
        print(f"Split {i}")
        print(f"Feed Indices - {feed_indices}")
        print(f"Testing Indices - {testing_indices}")
        feed_input_data = [input_data[i] for i in feed_indices]
        feed_target_data = [target_data[i] for i in feed_indices]
        testing_input_data = [input_data[i] for i in testing_indices]
        testing_target_data = [target_data[i] for i in testing_indices]
        feed_input_months = [input_data_months[i] for i in feed_indices]
        feed_target_months = [target_data_months[i] for i in feed_indices]
        testing_input_months = [input_data_months[i] for i in testing_indices]
        testing_target_months = [target_data_months[i] for i in testing_indices]
        print(f"Feed Input Data - {feed_input_data}")
        print(f"Feed Target Data - {feed_target_data}")
        print(f"Testing Input Data - {testing_input_data}")
        print(f"Testing Target Data - {testing_target_data}")
        print(f"Feed Input Data Months - {feed_input_months}")
        print(f"Feed Target Data Months - {feed_target_months}")
        print(f"Testing Input Data Months - {testing_input_months}")
        print(f"Testing Target Data Months - {testing_target_months}")
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
        sgd_model_aes = []
        lbfgs_model_aes = []
        adam_model_aes = []
        sgd_model_ses = []
        lbfgs_model_ses = []
        adam_model_ses = []
        for j in range(len(testing_target_data)):
            sgd_model_aes.append(abs(testing_target_data[j] - sgd_model_testing_predictions[j]))
            lbfgs_model_aes.append(abs(testing_target_data[j] - lbfgs_model_testing_predictions[j]))
            adam_model_aes.append(abs(testing_target_data[j] - adam_model_testing_predictions[j]))
            sgd_model_ses.append(pow(testing_target_data[j] - sgd_model_testing_predictions[j], 2))
            lbfgs_model_ses.append(pow(testing_target_data[j] - lbfgs_model_testing_predictions[j], 2))
            adam_model_ses.append(pow(testing_target_data[j] - adam_model_testing_predictions[j], 2))
        sgd_model_mae = stats.mean(sgd_model_aes)
        lbfgs_model_mae = stats.mean(lbfgs_model_aes)
        adam_model_mae = stats.mean(adam_model_aes)
        sgd_model_rmse = math.sqrt(stats.mean(sgd_model_ses))
        lbfgs_model_rmse = math.sqrt(stats.mean(lbfgs_model_ses))
        adam_model_rmse = math.sqrt(stats.mean(adam_model_ses))
        sgd_model_nrmse = sgd_model_rmse / stats.mean(testing_target_data)
        lbfgs_model_nrmse = lbfgs_model_rmse / stats.mean(testing_target_data)
        adam_model_nrmse = adam_model_rmse / stats.mean(testing_target_data)
        for j in range(len(testing_target_data)):
            testing_targets_across_splits.append(testing_target_data[j])
            sgd_model_testing_predictions_across_splits.append(sgd_model_testing_predictions[j])
            lbfgs_model_testing_predictions_across_splits.append(lbfgs_model_testing_predictions[j])
            adam_model_testing_predictions_across_splits.append(adam_model_testing_predictions[j])
        sgd_model_maes_across_splits.append(sgd_model_mae)
        lbfgs_model_maes_across_splits.append(lbfgs_model_mae)
        adam_model_maes_across_splits.append(adam_model_mae)
        sgd_model_rmses_across_splits.append(sgd_model_rmse)
        lbfgs_model_rmses_across_splits.append(lbfgs_model_rmse)
        adam_model_rmses_across_splits.append(adam_model_rmse)
        sgd_model_nrmses_across_splits.append(sgd_model_nrmse)
        lbfgs_model_nrmses_across_splits.append(lbfgs_model_nrmse)
        adam_model_nrmses_across_splits.append(adam_model_nrmse)
        print(f"Split {i} Results")
        print(f"SGD Model Predictions - {sgd_model_testing_predictions}")
        print(f"Adam Model Predictions - {adam_model_testing_predictions}")
        print(f"LBFGS Model Predictions - {lbfgs_model_testing_predictions}")
        print(f"SGD Model MAE - {sgd_model_mae}")
        print(f"Adam Model MAE - {adam_model_mae}")
        print(f"LBFGS Model MAE - {lbfgs_model_mae}")
        print(f"SGD Model RMSE - {sgd_model_rmse}")
        print(f"Adam Model RMSE - {adam_model_rmse}")
        print(f"LBFGS Model RMSE - {lbfgs_model_rmse}")
        print(f"SGD Model N-RMSE - {sgd_model_nrmse}")
        print(f"Adam Model N-RMSE - {adam_model_nrmse}")
        print(f"LBFGS Model N-RMSE - {lbfgs_model_nrmse}")
        i += 1
    sgd_model_mean_mae = stats.mean(sgd_model_maes_across_splits)
    lbfgs_model_mean_mae = stats.mean(lbfgs_model_maes_across_splits)
    adam_model_mean_mae = stats.mean(adam_model_maes_across_splits)
    sgd_model_mean_rmse = stats.mean(sgd_model_rmses_across_splits)
    lbfgs_model_mean_rmse = stats.mean(lbfgs_model_rmses_across_splits)
    adam_model_mean_rmse = stats.mean(adam_model_rmses_across_splits)
    sgd_model_mean_nrmse = stats.mean(sgd_model_nrmses_across_splits)
    lbfgs_model_mean_nrmse = stats.mean(lbfgs_model_nrmses_across_splits)
    adam_model_mean_nrmse = stats.mean(adam_model_nrmses_across_splits)
    print("Summary")
    print(f"Testing Targets - {testing_targets_across_splits}")
    print(f"SGD Model Testing Predictions - {sgd_model_testing_predictions_across_splits}")
    print(f"Adam Model Testing Predictions - {adam_model_testing_predictions_across_splits}")
    print(f"LBFGS Model Testing Predictions - {lbfgs_model_testing_predictions_across_splits}")
    print(f"SGD Model MAEs Across Splits - {sgd_model_maes_across_splits}")
    print(f"Adam Model MAEs Across Splits - {adam_model_maes_across_splits}")
    print(f"LBFGS Model MAEs Across Splits - {lbfgs_model_maes_across_splits}")
    print(f"SGD Model RMSEs Across Splits - {sgd_model_rmses_across_splits}")
    print(f"Adam Model RMSEs Across Splits - {adam_model_rmses_across_splits}")
    print(f"LBFGS Model RMSEs Across Splits - {lbfgs_model_rmses_across_splits}")
    print(f"SGD Model N-RMSEs Across Splits - {sgd_model_nrmses_across_splits}")
    print(f"Adam Model N-RMSEs Across Splits - {adam_model_nrmses_across_splits}")
    print(f"LBFGS Model N-RMSEs Across Splits - {lbfgs_model_nrmses_across_splits}")
    print(f"SGD Model Mean MAE - {sgd_model_mean_mae}")
    print(f"Adam Model Mean MAE - {adam_model_mean_mae}")
    print(f"LBFGS Model Mean MAE - {lbfgs_model_mean_mae}")
    print(f"SGD Model Mean RMSE - {sgd_model_mean_rmse}")
    print(f"Adam Model Mean RMSE - {adam_model_mean_rmse}")
    print(f"LBFGS Model Mean RMSE - {lbfgs_model_mean_rmse}")
    print(f"SGD Model Mean N-RMSE - {sgd_model_mean_nrmse}")
    print(f"Adam Model Mean N-RMSE - {adam_model_mean_nrmse}")
    print(f"LBFGS Model Mean N-RMSE - {lbfgs_model_mean_nrmse}")

"""This SGD Model processes Economic datasets and returns the respective input and target
datasets to be used for testing between the neural network SGD Models."""
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

"""This function determines the total weighted feature weights of a given Bayesian Optimized SGD SGD Model using a
backward weighting propagation method."""
def calculate_total_weighted_feature_importances(sgd_model):
    parameters = sgd_model.get_parameters()
    total_weighted_input_importances_across_layers = []
    for i in range(sgd_model.get_sgd_model_depth()):
        total_weighted_input_importances_across_layers.append([])
    for i in reversed(range(sgd_model.get_sgd_model_depth())):
        layer_total_weighted_input_importances_across_neurons = []
        for j in range(int(math.pow(sgd_model.get_neuron_size_base(), sgd_model.get_sgd_model_depth() - 1 - i))):
            if i != len(parameters) - 1:
                neuron_total_weighted_input_importances = [parameters[i][j][0][k] * total_weighted_input_importances_across_layers[i + 1][j] for k in range(len(parameters[i][j][0]))]
            else:
                neuron_total_weighted_input_importances = [parameters[i][j][0][k] for k in range(len(parameters[i][j][0]))]
            layer_total_weighted_input_importances_across_neurons.append(neuron_total_weighted_input_importances)
        layer_total_weighted_input_importances = [sum([layer_total_weighted_input_importances_across_neurons[k][j] for k in range(len(layer_total_weighted_input_importances_across_neurons))]) for j in range(len(layer_total_weighted_input_importances_across_neurons[0]))]
        total_weighted_input_importances_across_layers[i] = layer_total_weighted_input_importances
    return total_weighted_input_importances_across_layers[0]
