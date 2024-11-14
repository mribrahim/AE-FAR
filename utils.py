
import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def read_data(dataset):
    

    if "SMD" == dataset:

        data_path = "../Anomaly/SMD/"

        scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")[:,:]
        scaler.fit(data)
        data = scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")[:,:]
        test_data = scaler.transform(test_data)
        train_data = data
        data_len = len(train_data)
        val_data = test_data#train_data[(int)(data_len * 0.8):]
        test_labels = np.load(data_path + "/SMD_test_label.npy")[:]

    elif "SWAT" == dataset:

        data_path = "../Anomaly/SWAT/"

        train_data = pd.read_csv( data_path + 'swat_train2.csv')
        test_data = pd.read_csv(data_path + 'swat2.csv')

        test_labels = test_data.values[:, -1]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        scaler = StandardScaler()
        scaler.fit(train_data)

        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        data_len = len(train_data)
        val_data = test_data

    elif "MSL" == dataset:

        data_path = "../Anomaly/MSL/"

        scaler = StandardScaler()
        train_data = np.load(data_path + "/MSL_train.npy")
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = np.load(data_path + "/MSL_test.npy")
        test_data = scaler.transform(test_data)
        test_labels = np.load(data_path + "/MSL_test_label.npy")
        data_len = len(train_data)
        val_data = test_data
    
    elif  "PSM" == dataset:
        scaler = StandardScaler()
        data_path = "../Anomaly/PSM/"
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        scaler.fit(data)
        data = scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        test_data = scaler.transform(test_data)
        train_data = data
        val_data = test_data
        test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

    elif  "SMAP" == dataset:
        scaler = StandardScaler()
        data_path = "../Anomaly/SMAP/"
        data = np.load(data_path + "/SMAP_train.npy")
        scaler.fit(data)
        data = scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        test_data = scaler.transform(test_data)
        train_data = data
        val_data = test_data
        test_labels = np.load(data_path + "/SMAP_test_label.npy")

    return train_data, test_data, val_data, test_labels



# to predict a single target value, not the entire window
def iterate_batches(data, window_size, batch_size, start_idx = 0):
    for start in range(start_idx, len(data) - window_size, batch_size):
        end = min(start + batch_size, len(data) - window_size)
        batch_data = [data[i:i + window_size] for i in range(start, end)]
        batch_targets = [data[i + window_size] for i in range(start, end)]
        yield torch.stack(batch_data), torch.stack(batch_targets)


def apply_adjustment(gt_, pred_):
    gt = gt_.copy()
    pred = pred_.copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred
# end function

def sliding_window_anomaly_detection(mse_list, window_size, threshold_factor=3):
    mse_series = pd.Series(mse_list)
    
    # Calculate moving average and moving standard deviation
    moving_avg = mse_series.rolling(window=window_size, min_periods=1).mean()
    moving_std = mse_series.rolling(window=window_size, min_periods=1).std()
    
    # Calculate dynamic threshold
    dynamic_threshold = moving_avg + (threshold_factor * moving_std)
    
    # Identify anomalies
    anomalies = (mse_series > dynamic_threshold).astype(int)

    # Convert to list for output
    anomalies_list = anomalies.tolist()
    
    return anomalies_list, dynamic_threshold.tolist()

def get_precision_recall_f1(true_labels, pred_y):
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, pred_y, average='binary')
    return round(precision, 4), round(recall, 4), round(f1_score, 4)