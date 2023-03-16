from Helpers import *
import pandas as pd
import random
import string
import numpy as np
import math

import datetime


def get_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
    # print("Random string of length", length, "is:", result_str)


def read_data(file_name):
    df_csv = pd.read_excel(file_name)

    # Correction of Discrepencies in the CallDuration Column
    if file_name=="/home/nehaj/infinity/altair/clean_data/data1.xlsx":
        for column in df_csv[['CallDuration']]:
            for idx, value in enumerate(df_csv[column].values):
                if (type(value) == datetime.datetime):
                    df_csv[column][idx] = datetime.datetime.strftime(
                        value, '%H:%M:%S')

        # print(columnData.dtypes, type(columnData[0]), type(columnData), columnName)
        for (columnName, columnData) in df_csv.iteritems():
            if isinstance(columnData[0], datetime.time):
                if columnName == 'CallTime':
                    df_csv[columnName] = pd.to_datetime(pd.to_datetime(df_csv['CallDate']).dt.strftime(
                        '%Y-%m-%d') + df_csv[columnName].astype(str), format='%Y-%m-%d%H:%M:%S').to_frame()
                    print('Time Done')
                if columnName == 'CallDuration':
                    df_csv[columnName] = pd.to_datetime(pd.to_datetime(df_csv['CallDate']).dt.strftime(
                    '%Y-%m-%d') + df_csv[columnName].astype(str), format='%Y-%m-%d%H:%M:%S').to_frame() 

    pd.set_option('display.max_columns', 500)

    fields = []
    # print(df_csv)
    # tuple ('container, y', {'uid':'', 'order': 7, 'data': []})
    for (columnName, columnData) in df_csv.iteritems():
        print((columnData[0]), columnData.dtypes, type(
            columnData[0]), type(columnData), columnName)

        if (isinstance(columnData[0], pd._libs.tslibs.timestamps.Timestamp)):
            # df_csv[columnName]=df_csv[columnName].to_frame().applymap(str)
            # data_dict = {'uid': get_random_string(6), 'order': random.randint(0, 9),
            #              'data': df_csv[columnName].values.astype(str).tolist()}
            data_dict = {'uid': get_random_string(6), 'order': random.randint(0, 9),
                         'data': columnData.to_frame().applymap(str).values.tolist()}
            print('xaaaaaaaaaa1', type(columnData[0]), columnData[0])

        # elif isinstance(columnData[0], datetime.time) or isinstance(columnData[0], datetime.datetime):
        #     df_csv[columnName] = df_csv[columnName].to_frame().applymap(str)
        #     data_dict = {'uid': get_random_string(6), 'order': random.randint(0, 9),
        #                  'data': df_csv[columnName].values.astype(str).tolist()}
        #     # data_dict = {'uid': get_random_string(6), 'order': random.randint(0, 9),
        #     #              'data': columnData.to_frame().applymap(str).values.tolist()}
        #     print('xaaaaaaaaaa2',type(columnData[0]),columnData[0])            fields.append(('container, y', data_dict))
            fields.append((columnName, data_dict))

        else:
            data_dict = {'uid': get_random_string(6), 'order': random.randint(0, 9),
                         'data': columnData.to_list()}
            fields.append((columnName, data_dict))
        # data_dict = {'uid': get_random_string(6), 'order': random.randint(0, 9),
        #              'data': columnData.to_list()}
        print("---------------------------------------------")
        # fields.append(('container, y', data_dict))

        # Get all the column names

    column_names = df_csv.columns.tolist()
    df_csv
    # Extract all possible pairs of columns
    pairs = [(col1, col2) for i, col1 in enumerate(column_names)
             for col2 in column_names[i+1:]]
    # Calculate the number of pairs using the formula C(n, 2)
    num_pairs = math.comb(len(column_names), 2)

    # Store the index combinations
    index_combinations = []

    #print(f"Number of columns: {len(column_names)}")
    #print(f"Number of pairs: {num_pairs}")
    for col1, col2 in pairs:
        col1_index = column_names.index(col1)
        col2_index = column_names.index(col2)
        index_combinations.append((col1_index, col2_index))
        #print(
        #    f"Data for columns {col1} (index: {col1_index}) and {col2} (index: {col2_index}):")
        #print(index_combinations)

    return fields, index_combinations


def extract_features(file_name):
    fid = get_random_string(7) + ":" + str(random.randint(0, 9))
    field = []
    fields, index_combinations = read_data(file_name)

    single_field_features, parsed_fields = extract_single_field_features(
        fields, fid, MAX_FIELDS=len(fields))
    
    aggregate_single_field_features = extract_aggregate_single_field_features(
        single_field_features)

    pairwise_field_features = extract_pairwise_field_features(
        parsed_fields, single_field_features, fid, MAX_FIELDS=len(fields))
        

    aggregate_pairwise_field_features = extract_aggregate_pairwise_field_features(
        pairwise_field_features)


    return aggregate_single_field_features, aggregate_pairwise_field_features


if __name__ == '__main__':
    agg_single, agg_pairwise = extract_features(
        "/home/nehaj/infinity/altair/data_to_search/Fwd_data_infinity/datasets 2/new_rape_data_file0.xlsx")

