import pandas
from TicToc import TicToc
import numpy as np
from scipy.interpolate import interp1d


# SubjectID|form_name|feature_name|feature_value|feature_unit|feature_delta

def print_pd(pd):
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        print(pd)
    return


def read():
    train_filenamex = 'Data/all_forms_PROACT.txt'
    train_filenamey = 'Data/ALSFRS_slope_PROACT.txt'
    train_filenamez = 'Data/surv_response_PROACT.txt'
    val_filenamex = 'Data/all_forms_validate_spike.txt'
    val_filenamey = 'Data/ALSFRS_slope_validate_spike.txt'
    val_filenamez = 'Data/surv_response_validate_spike.txt'
    x_train = pandas.read_csv(train_filenamex, delimiter='|', header=0)
    y_train = pandas.read_csv(train_filenamey, delimiter='|', header=0)
    z_train = pandas.read_csv(train_filenamez, delimiter='|', header=0)

    x_val = pandas.read_csv(val_filenamex, delimiter='|', header=0)
    y_val = pandas.read_csv(val_filenamey, delimiter='|', header=0)
    z_val = pandas.read_csv(val_filenamez, delimiter='|', header=0)
    return x_train, y_train, z_train, x_val, y_val, z_val


def countingfeature():
    count_train = x_train.groupby('feature_name')['SubjectID'].nunique().sort_values(ascending=False)
    count_val = x_val.groupby('feature_name')['SubjectID'].nunique().sort_values(ascending=False)

    a = count_train.index.tolist()[:50]
    b = count_val.index.tolist()[:50]
    c = set(a).intersection(set(b))
    print(c)
    return c


def process(x):
    feature_subset = ['ALSFRS_Total', 'Age', 'Chloride', 'Gender', 'Hematocrit', 'Hemoglobin',
                      'Q1_Speech', 'Q2_Salivation', 'Q3_Swallowing', 'Q4_Handwriting', 'Q5_Cutting',
                      'Q6_Dressing_and_Hygiene', 'Q7_Turning_in_Bed', 'Q8_Walking', 'Q9_Climbing_Stairs',
                      'bp_diastolic', 'bp_systolic', 'hands', 'leg', 'mouth',
                      'respiratory', 'weight']

    not_time_series_feature = ['Gender', 'Age']
    time_series_feature = set(feature_subset) - set(not_time_series_feature)
    x = x.loc[x['feature_name'].isin(feature_subset)]
    x = x.loc[x['feature_value'].notna()]
    x = x.loc[x['feature_delta'].notna()]

    x_list = {}
    tic = TicToc()
    for feature in feature_subset:
        if feature == 'Gender':
            x_feature = x.loc[x['feature_name'] == feature]
            x_feature = x_feature[['SubjectID', 'feature_value']]
            x_feature['feature_value'] = x_feature['feature_value'].apply(lambda sex: 1 if sex == 'M' else 0)
            x_feature.columns = ['SubjectID', 'Gender']
            x_list[feature] = x_feature
            continue
        if feature == 'Age':
            x_feature = x.loc[x['feature_name'] == feature]
            x_feature = x_feature[['SubjectID', 'feature_value']]
            x_feature.columns = ['SubjectID', 'Age']
            x_list[feature] = x_feature
            continue
        x_feature = x.loc[x['feature_name'] == feature]
        # extract feature and feature delta AND casting
        x_feature = x_feature[['SubjectID', 'feature_value', 'feature_delta']].astype(float)
        # group feature_value to a list
        x_feature_value = x_feature.groupby('SubjectID')['feature_value'].apply(np.array)
        # drop the subject with feature too few data points
        x_feature_value = x_feature_value[x_feature_value.str.len() >= 3]
        # group delta to a list
        x_feature_delta = x_feature.groupby('SubjectID')['feature_delta'].apply(np.array)
        # drop the subject with feature too few data points
        x_feature_delta = x_feature_delta[x_feature_delta.str.len() >= 3]
        # sync
        x_feature_value, _ = delta_sync(x_feature_value, x_feature_delta)
        x_feature_value.columns = [feature]
        x_list[feature] = x_feature_value
    tic.toc()
    # merge all the datagram with inner join
    a = pandas.DataFrame(data={'SubjectID': np.arange(999999)})
    for feature in feature_subset:
        a = a.merge(x_list[feature], how='inner', left_on='SubjectID', right_on='SubjectID')
    a = a.loc[a.all(axis=1)]
    return a


def delta_sync(x_feature_value, x_feature_delta, delta_base=np.arange(0, 90 + 1, 15)):
    feature = x_feature_value.copy()
    delta = x_feature_delta.copy()
    for (i, y), x in zip(x_feature_value.items(), x_feature_delta):
        nan = np.isnan(x)
        x = x[nan == False]
        y = y[nan == False]
        nan = np.isnan(y)
        x = x[nan == False]
        y = y[nan == False]
        a = x.argsort()
        x = x[a]
        y = y[a]
        _, a = np.unique(x, return_index=True)
        x = x[a]
        y = y[a]
        if x.size < 2:
            feature[i] = False
            delta[i] = False
            continue
        delta_base_index_smaller = np.where(delta_base < x[0])[0].reshape(-1)
        delta_base_index_bigger = np.where(delta_base > x[-1])[0].reshape(-1)
        delta_base_index_normal = np.setdiff1d(np.where(delta_base <= x[-1])[0].reshape(-1), delta_base_index_smaller)
        # calcluate extrapolation
        value = np.empty_like(delta_base)
        slope = (y[1] - y[0]) / (x[1] - x[0])
        for j in delta_base_index_smaller:
            value[j] = y[1] - slope * (x[1] - delta_base[j])
        # calcluate extrapolation
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        for j in delta_base_index_bigger:
            value[j] = y[-2] - slope * (x[-2] - delta_base[j])
        # calculate interpolation
        f = interp1d(x, y)
        value[delta_base_index_normal] = f(delta_base[delta_base_index_normal])
        feature[i] = value
        delta[i] = delta_base
    return feature.to_frame(), delta.to_frame()


def merge(x, y, z):
    merge = x.merge(y, how='inner', left_on='SubjectID', right_on='SubjectID')
    merge = merge.merge(z, how='inner', left_on='SubjectID', right_on='SubjectID')
    final = merge.loc[merge['ALSFRS_slope'].notna() & merge['time_event'].notna() & merge['status'].notna()]
    final.index = np.arange(final.shape[0])
    return final


if __name__ == '__main__':
    # read data
    x_train, y_train, z_train, x_val, y_val, z_val = read()
    # process data
    train_pro = process(x_train)
    val_pro = process(x_val)
    # merge datafile x,y,z
    train_final = merge(train_pro, y_train, z_train)
    val_final = merge(val_pro, y_val, z_val)
    # save
    tic = TicToc()
    train_final.to_pickle('Data/Train_data_processed.pkl')
    val_final.to_pickle('Data/Val_data_processed.pkl')
    tic.toc()
