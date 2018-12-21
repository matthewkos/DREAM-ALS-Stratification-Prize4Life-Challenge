import pandas
import numpy as np


def read_data():
    train_filename = 'Data/Train_data_processed.pkl'
    val_filename = 'Data/Val_data_processed.pkl'

    train_data = pandas.read_pickle(train_filename)
    val_data = pandas.read_pickle(val_filename)
    return train_data, val_data


def times_series(data):
    # features = ['ALSFRS_Total', 'bp_diastolic', 'bp_systolic', 'Chloride', 'hands', 'Hematocrit', 'Hemoglobin',
    #             'leg', 'mouth', 'Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing',
    #             'Q4_Handwriting', 'Q5_Cutting', 'Q6_Dressing_and_Hygiene', 'Q7_Turning_in_Bed', 'Q8_Walking',
    #             'Q9_Climbing_Stairs', 'respiratory', 'weight']
    # sub-1
    # features =  ['ALSFRS_Total', 'bp_diastolic','Chloride', 'hands', 'Q1_Speech', 'Q6_Dressing_and_Hygiene']
    # sub-2
    features = ['bp_systolic', 'hands', 'leg', 'mouth', 'Q7_Turning_in_Bed', 'weight']
    data = data[features].values
    t_data = np.empty((data.shape[0], data.shape[1], data[0, 0].shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            t_data[i, j, :] = data[i, j]
    return t_data


def saclar(data):
    # features = ['Age', 'Gender']
    # sub-1
    #features = ['Gender']
    # sub-2
    features = ['Age']
    data = data[features].values
    return data


def truth(data):
    features = ['ALSFRS_slope']
    data = data[features].values
    return data


if __name__ == '__main__':
    train_data, val_data = read_data()
    t_train = times_series(train_data)
    s_train = saclar(train_data)
    y_train = truth(train_data)
    np.save("Data/t_train.npy", t_train)
    np.save("Data/s_train.npy", s_train)
    np.save("Data/y_train.npy", y_train)

    t_val = times_series(val_data)
    s_val = saclar(val_data)
    y_val = truth(val_data)
    np.save("Data/t_val.npy", t_val)
    np.save("Data/s_val.npy", s_val)
    np.save("Data/y_val.npy", y_val)
