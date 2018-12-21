import pandas
from TicToc import TicToc
import numpy as np
from process import print_pd
import sklearn.feature_selection

    
def read_data():
    train_filename = 'Data/Train_data_processed.pkl'
    val_filename = 'Data/Val_data_processed.pkl'

    train_data = pandas.read_pickle(train_filename)
    val_data = pandas.read_pickle(val_filename)
    return train_data, val_data

def process_into_np_sclar(x, name):
    m = dict()
    array_name = []
    for i in x.keys():
        if i in ['SubjectID', 'Age', 'Gender', 'ALSFRS_slope', 'time_event', 'status']:
            m[i] = x[i].values.reshape(-1, 1)
            array_name.append(i)
            continue
        m[i] = np.zeros((x.shape[0], 1))
        array_name.append(i)
        for j in range(m[i].shape[0]):
            m[i][j] = np.sum(x[i][j])
    mx = np.concatenate(tuple((m[i] for i in x.keys())), axis=1)
    mx = mx.astype(float)
    fx = mx[:, :-3]
    fy = mx[:, -3]
    fz = mx[:, -3:]
    return fx, fy, fz, np.array(array_name)


def process_into_np_slope(x, name):
    m = dict()
    array_name = []
    for i in x.keys():
        if i in ['SubjectID', 'Age', 'Gender', 'ALSFRS_slope', 'time_event', 'status']:
            m[i] = x[i].values.reshape(-1, 1)
            array_name.append(i)
            continue
        m[i] = np.zeros((x.shape[0], 1))
        array_name.append(i)
        for j in range(m[i].shape[0]):
            m[i][j] = (x[i][j][-1] - x[i][j][-0]) / 90.0

    mx = np.concatenate(tuple((m[i] for i in x.keys())), axis=1)
    mx = mx.astype(float)
    fx = mx[:, :-3]
    fy = mx[:, -3]
    fz = mx[:, -3:]
    return fx, fy, fz, np.array(array_name)


def process_into_np_avg(x, name):
    m = dict()
    array_name = []
    for i in x.keys():
        if i in ['SubjectID', 'Age', 'Gender', 'ALSFRS_slope', 'time_event', 'status']:
            m[i] = x[i].values.reshape(-1, 1)
            array_name.append(i)
            continue
        m[i] = np.zeros((x.shape[0], 1))
        array_name.append(i)
        for j in range(m[i].shape[0]):
            m[i][j] = np.average(x[i][j])

    mx = np.concatenate(tuple((m[i] for i in x.keys())), axis=1)
    mx = mx.astype(float)
    fx = mx[:, :-3]
    fy = mx[:, -3]
    fz = mx[:, -3:]
    return fx, fy, fz, np.array(array_name)


def process_into_np_idv(x, name):
    m = dict()
    array_name = []
    for i in x.keys():
        if i in ['SubjectID', 'Age', 'Gender', 'ALSFRS_slope', 'time_event', 'status']:
            m[i] = x[i].values.reshape(-1, 1)
            array_name.append(i)
            continue
        m[i] = np.zeros((x.shape[0], x[i][0].shape[0]))
        for k in range(m[i].shape[1]):
            array_name.append(i + '_' + str(k * 15))
            for j in range(m[i].shape[0]):
                m[i][j] = x[i][j]

    mx = np.concatenate(tuple((m[i] for i in x.keys())), axis=1)
    mx = mx.astype(float)
    fx = mx[:, :-3]
    fy = mx[:, -3]
    fz = mx[:, -3:]
    # np.savetxt('Data/{}_fx.txt'.format(name), fx, header=str(array_name[:-3]))
    # np.savetxt('Data/{}_fy.txt'.format(name), fy, header=str(array_name[-3]))
    # np.savetxt('Data/{}_fz.txt'.format(name), fz, header=str(array_name[-3:]))
    return fx, fy, fz, np.array(array_name)


def process_into_np_idv(x, name):
    m = dict()
    array_name = []
    for i in x.keys():
        if i in ['SubjectID', 'Age', 'Gender', 'ALSFRS_slope', 'time_event', 'status']:
            m[i] = x[i].values.reshape(-1, 1)
            array_name.append(i)
            continue
        m[i] = np.zeros((x.shape[0], x[i][0].shape[0]))
        for k in range(m[i].shape[1]):
            array_name.append(i + '_' + str(k * 15))
            for j in range(m[i].shape[0]):
                m[i][j] = x[i][j]

    mx = np.concatenate(tuple((m[i] for i in x.keys())), axis=1)
    mx = mx.astype(float)
    fx = mx[:, :-3]
    fy = mx[:, -3]
    fz = mx[:, -3:]
    # np.savetxt('Data/{}_fx.txt'.format(name), fx, header=str(array_name[:-3]))
    # np.savetxt('Data/{}_fy.txt'.format(name), fy, header=str(array_name[-3]))
    # np.savetxt('Data/{}_fz.txt'.format(name), fz, header=str(array_name[-3:]))
    return fx, fy, fz, np.array(array_name)


def analysis_mi_slope(x, y, array_name):
    print('*' * 20, "Slope", '*' * 20)
    mi = sklearn.feature_selection.mutual_info_regression(x, y, n_neighbors=3)
    mi_sort_index = np.argsort(mi)[::-1]
    for i in range(mi.shape[0]):
        print('{:>20} : {:.4f}'.format(array_name[mi_sort_index[i]], mi[mi_sort_index[i]]))
    print('*' * 50)

def analysis_mi_dead(x, y, array_name):
    print('*' * 20, "Dead", '*'*20)
    mi = sklearn.feature_selection.mutual_info_classif(x, y, n_neighbors=3)
    mi_sort_index = np.argsort(mi)[::-1]
    for i in range(mi.shape[0]):
        print('{:>20} : {:.4f}'.format(array_name[mi_sort_index[i]], mi[mi_sort_index[i]]))
    print('*' * 50)

def MI_slope(x):
    # fx, fy, fz, array_name = process_into_np_idv(x, 'train')
    # analysis_mi_slope(fx, fy, array_name)
    # fx, fy, fz, array_name = process_into_np_avg(x, 'train')
    # analysis_mi_slope(fx, fy, array_name)
    # fx, fy, fz, array_name = process_into_np_slope(x, 'train')
    # analysis_mi_slope(fx, fy, array_name)
    fx, fy, fz, array_name = process_into_np_sclar(x, 'train')
    analysis_mi_slope(fx, fy, array_name)


def MI_dead(x):
    # fx, fy, fz, array_name = process_into_np_idv(x, 'train')
    # analysis_mi_dead(fx, fz[:,-1], array_name)
    # fx, fy, fz, array_name = process_into_np_avg(x, 'train')
    # analysis_mi_dead(fx, fz[:, -1], array_name)
    # fx, fy, fz, array_name = process_into_np_slope(x, 'train')
    # analysis_mi_dead(fx, fz[:, -1], array_name)
    fx, fy, fz, array_name = process_into_np_sclar(x, 'train')
    analysis_mi_dead(fx, fz[:, -1], array_name)


if __name__ == '__main__':
    tic = TicToc()
    train_data, val_data = read_data()
    print("training")
    MI_slope(train_data)
    MI_dead(train_data)
    print("validation")
    MI_slope(val_data)
    MI_dead(val_data)
    tic.toc()
