# <YOUR_IMPORTS>
import pandas as pd
import dill
import json
import glob
import os


def max_name_pkl():
    path = os.path.expanduser('~/airflow_hw')
    pkl_list = path + '/data/models/*.pkl'
    pkl_date = []
    for elem in glob.glob(pkl_list):
        pkl_date.append(elem[-16:-4])
    return max(*pkl_date)


def predict():
    # <YOUR_CODE>
    path = os.path.expanduser('~/airflow_hw')
    name_pkl = max_name_pkl()
    with open(f'{path}/data/models/cars_pipe_{name_pkl}.pkl', 'rb') as file:
        model = dill.load(file)

    filename_list = path + '/data/test/*.json'
    count = 0
    for elem in glob.glob(filename_list):
        with open(elem) as file:
            form = json.load(file)
            df_tmp = pd.DataFrame.from_dict([form])
            df_tmp = df_tmp[['id', 'url', 'region', 'region_url', 'price', 'year', 'manufacturer', 'model', 'fuel',
                             'odometer', 'title_status', 'transmission', 'image_url', 'description', 'state', 'lat',
                             'long', 'posting_date']]
        if count == 0:
            data = df_tmp.copy()
        else:
            data = pd.concat([data, df_tmp], ignore_index=True)
        count += 1
    data['Predict'] = model.predict(data)
    data[['id', 'Predict']].to_csv(f'{path}/data/predictions/preds_{name_pkl}.csv', index=False)


if __name__ == '__main__':
    predict()
