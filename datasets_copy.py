import ground_truth
import pandas as pd
import torch
from torch.utils.data import Dataset
from Helpers import *
import pandas as pd
import random
import string
import torch
import numpy as np
import math
from torch.utils.data import DataLoader
import pickle
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.preprocessing import OneHotEncoder

class SingleTableDataset(Dataset):

    def __init__(self, file_path, groundT_dict):
        self.data_file_path = file_path
        
        #all possible combinations of the columns in one dataset
        self.field, self.combinations = self.read_data(self.data_file_path)

        #extract single column features
        self.all_features = extract_single_field_features(self.field, self.get_random_string(
            7) + ":" + str(random.randint(0, 9)), MAX_FIELDS=len(self.field))[0]
        
        #Read respective ground truth(adjacency matrix for the file)
        for key in groundT_dict:
            if key in file_path:
                self.gt = groundT_dict.get(key)
        

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        x1, x2 = self.combinations[idx]
        x1_single_comulmn_features = (self.all_features[x1])
        x2_single_comulmn_features = (self.all_features[x2])
        aggregate_single_features_x1_x2 = extract_aggregate_single_field_features(
            [x1_single_comulmn_features, x2_single_comulmn_features])

        #print(type(x1_single_comulmn_features.values()))
        #print(type(x2_single_comulmn_features.values()))
        #print(type(aggregate_single_features_x1_x2))

        x1_final_features = torch.tensor(
            [0 if v is None else v for v in x1_single_comulmn_features.values()][2:], dtype=torch.float64)
        x2_final_features = torch.tensor(
            [0 if v is None else v for v in x2_single_comulmn_features.values()][2:], dtype=torch.float64)
        agg_features = torch.tensor(
            [0 if v is None else v for v in aggregate_single_features_x1_x2.values()], dtype=torch.float64)

        print(x1_final_features)
        print(x2_final_features)
        print(agg_features)
        gr_t = self.gt[x1][x2]
        #print("GT",x1,x2,gr_t)
        return x1_final_features, x2_final_features, agg_features, gr_t

    def get_random_string(self, length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
        # print("Random string of length", length, "is:", result_str)

    def read_data(self, file_name):
        df_csv = pd.read_excel(file_name)

        # Correction of Discrepencies in the CallDuration Column
        """ for column in df_csv[['CallDuration']]:
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
                        '%Y-%m-%d') + df_csv[columnName].astype(str), format='%Y-%m-%d%H:%M:%S').to_frame() """

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
                data_dict = {'uid': self.get_random_string(6), 'order': random.randint(0, 9),
                             'data': columnData.to_frame().applymap(str).values.tolist()}
                print('xaaaaaaaaaa1', type(columnData[0]), columnData[0])

                fields.append((columnName, data_dict))

            else:
                data_dict = {'uid': self.get_random_string(6), 'order': random.randint(0, 9),
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

       # print(f"Number of columns: {len(column_names)}")
        # print(f"Number of pairs: {num_pairs}")
        for col1, col2 in pairs:
            col1_index = column_names.index(col1)
            col2_index = column_names.index(col2)
            index_combinations.append((col1_index, col2_index))
           # print(
            #    f"Data for columns {col1} (index: {col1_index}) and {col2} (index: {col2_index}):")
            # print(index_combinations)

        return fields, index_combinations



class TableDataset(Dataset):

    def __init__(self, file_path):
      
        self.features, self.grtruth, self.allcolumns , self.index, self.agg_features= self.read_data(file_path)
  

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        x1, x2 = self.index[idx]
    
        x1_single_comulmn_features = (self.features[x1])
        x2_single_comulmn_features = (self.features[x2])
        aggregate_single_features_x1_x2 = extract_aggregate_single_field_features(
            [x1_single_comulmn_features, x2_single_comulmn_features])


        #agg features of the dataset
        agg_features_dataset =self.agg_features[idx]
        #for idx, (keys, values) in enumerate(aggregate_single_features_x1_x2.items()):
        #    print(idx, keys, values)
        
        #for key, value in x1_single_comulmn_features.items():
        #    if isinstance(value, numbers.Number) and (math.isnan(value) or math.isinf(value)):
        #        print(key, value)

        #standardize the agg features for pair
        """scaler1 = StandardScaler()

        for k, v in aggregate_single_features_x1_x2.items():
            if isinstance(v, (int, float)):
                aggregate_single_features_x1_x2[k] = scaler1.fit_transform([[v]])[0][0]"""


        """ #standardize the agg features for dataset
        for k, v in agg_features_dataset:
            if isinstance(v, (int, float)):
                agg_features_dataset[k] = scaler1.fit_transform([[v]])[0][0]"""

        #print(x1_single_comulmn_features.values())
        #print(x2_single_comulmn_features.values())
        #print(aggregate_single_features_x1_x2)

        x1_final_features = torch.tensor(
            [0 if v is None else v for v in x1_single_comulmn_features.values()][2:], dtype=torch.float64)
        x2_final_features = torch.tensor(
            [0 if v is None else v for v in x2_single_comulmn_features.values()][2:], dtype=torch.float64)
        agg_features = torch.tensor(
            [0 if v is None else v for v in aggregate_single_features_x1_x2.values()], dtype=torch.float64)
        agg_features_dataset = torch.tensor(
            [0 if v is None else v for v in agg_features_dataset.values()], dtype=torch.float64)


       # print("afg",agg_features_dataset)
       # print("x2",x2,x2_final_features)
        #print("agg",x1,x2,agg_features)
        
        gr_t = self.grtruth[idx]
       # print("GT",x1,x2,gr_t)
        return x1_final_features, x2_final_features, agg_features, agg_features_dataset, gr_t

    def get_random_string(self, length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
        # print("Random string of length", length, "is:", result_str)

    def read_data(self, file_path):
        
        data_file_path = file_path
        columns =[]
        gt = []
        agg = []
        index = []
        single_field_features =[]
        all_features = []
        agg_features =[]
        for filename in os.listdir(data_file_path):
            single_file_path = os.path.join(data_file_path, filename)
            if os.path.isfile(single_file_path):    

                #read file
                df_csv = pd.read_excel(single_file_path)

                # Correction of Discrepencies in the CallDuration Column
                if single_file_path=="/home/nehaj/infinity/altair/clean_data/data1.xlsx":
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


                #get fields 
                pd.set_option('display.max_columns', 500)
                fields = []
                # tuple ('container, y', {'uid':'', 'order': 7, 'data': []})
                for (columnName, columnData) in df_csv.iteritems():
                    #print((columnData[0]), columnData.dtypes, type(
                     #   columnData[0]), type(columnData), columnName)

                    if (isinstance(columnData[0], pd._libs.tslibs.timestamps.Timestamp)):
                        # df_csv[columnName]=df_csv[columnName].to_frame().applymap(str)
                        # data_dict = {'uid': get_random_string(6), 'order': random.randint(0, 9),
                        #              'data': df_csv[columnName].values.astype(str).tolist()}
                        data_dict = {'uid': self.get_random_string(6), 'order': random.randint(0, 9),
                                    'data': columnData.to_frame().applymap(str).values.tolist()}
                        #print('xaaaaaaaaaa1', type(columnData[0]), columnData[0])

                        fields.append((columnName, data_dict))

                    else:
                        data_dict = {'uid': self.get_random_string(6), 'order': random.randint(0, 9),
                                    'data': columnData.to_list()}
                        fields.append((columnName, data_dict))

                single_field_features, parsed_fields = extract_single_field_features(fields, self.get_random_string(7) + ":" + str(random.randint(0, 9)), MAX_FIELDS=len(fields))
                all_features.extend(single_field_features)
                #pairwise_field_features = extract_pairwise_field_features(parsed_fields, single_field_features, self.get_random_string(7) + ":" + str(random.randint(0, 9)), MAX_FIELDS=len(fields))
                #agg features
                #agg_features =extract_aggregate_single_field_features(single_field_features)
                #print(agg_features)
        
                #normalize the features 
                scaler = StandardScaler()
                for feature in all_features:
                    for k, v in feature.items():
                        if isinstance(v, (int, float)):
                            feature[k] = scaler.fit_transform([[v]])[0][0]

                #print(all_features)
               # for (keys, values) in enumerate(all_features.values()):
               #     print(keys, values)

                #print(type(all_features))
                #get combinations
                
                columns.extend(df_csv.columns.tolist())
                column_names = df_csv.columns.tolist()

                # Extract all possible pairs of columns
                pairs = [(col1, col2) for i, col1 in enumerate(column_names)
                        for col2 in column_names[i+1:]]
                # Calculate the number of pairs using the formula C(n, 2)
                num_pairs = math.comb(len(column_names), 2)

                #assign agg features of the dataset to all the pairs indices
                agg =extract_aggregate_single_field_features(single_field_features)
                while num_pairs >  0:
                    agg_features.append(agg)
                    num_pairs -= 1

                # Store the index combinations
                index_combinations = []
                index_combina =[]

                for col1, col2 in pairs:
                    col1_index = columns.index(col1)
                    col2_index = columns.index(col2)
                    col1_in = column_names.index(col1)
                    col2_in = column_names.index(col2)
                    index_combina.append((col1_in, col2_in))
                    index_combinations.append((col1_index, col2_index))
                    
                index.extend(index_combinations)
                
                #load ground truths
                for key in ground_truth.adj_matrices:
                    if key in single_file_path:
                        gtruth =[]
                        df = ground_truth.adj_matrices.get(key)
                        for (col1_in, col2col2_in) in index_combina:
                            element = df.iloc[col1_in, col2_in]
                            gtruth.append(element)
       
                gt.extend(gtruth)   
     
        return all_features, gt, columns, index, agg_features




class SemanticTableDataset(Dataset):

    def __init__(self, file_path):
        self.features, self.grtruth, self.allcolumns , self.index, self.agg_features, self.semantic_column_data = self.read_data(file_path)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        x1, x2 = self.index[idx]
    
        x1_single_comulmn_features = (self.features[x1])
        x2_single_comulmn_features = (self.features[x2])
        aggregate_single_features_x1_x2 = extract_aggregate_single_field_features(
            [x1_single_comulmn_features, x2_single_comulmn_features])

        #agg features of the dataset
        agg_features_dataset =self.agg_features[idx]

        #semantic features
        # Convert dictionary values to a list
        semantic_values_list = list( self.semantic_column_data.values())

        # Access value at position 1 (remember, list indexing starts at 0)
        x1_semantic = semantic_values_list[x1] 
        x2_semantic = semantic_values_list[x2] 

        x1_final_features = torch.tensor(
            [0 if v is None else v for v in x1_single_comulmn_features.values()][2:], dtype=torch.float64)
        x2_final_features = torch.tensor(
            [0 if v is None else v for v in x2_single_comulmn_features.values()][2:], dtype=torch.float64)
        agg_features = torch.tensor(
            [0 if v is None else v for v in aggregate_single_features_x1_x2.values()], dtype=torch.float64)
        agg_features_dataset = torch.tensor(
            [0 if v is None else v for v in agg_features_dataset.values()], dtype=torch.float64)

        x1_semantic_tensor = torch.tensor(x1_semantic, dtype=torch.float64)
        x2_semantic_tensor = torch.tensor(x2_semantic, dtype=torch.float64)
       
        gr_t = self.grtruth[idx]
     
        return x1_final_features, x2_final_features, agg_features, agg_features_dataset, x1_semantic_tensor, x2_semantic_tensor, gr_t

    def get_random_string(self, length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
        # print("Random string of length", length, "is:", result_str)

    def get_semantic_datatypes(self):
        with open("/home/nehaj/infinity/altair/final_data/output_all.pkl", 'rb') as handle:
            semantic_types = pickle.load(handle)
            print("Semantic",semantic_types)
            semantic_types = self.semantic_onehot_encoding(semantic_types)

        return semantic_types

    def semantic_onehot_encoding(self,semantic_types):
        enc = OneHotEncoder(handle_unknown='ignore')
        X = [["only_date"],["crime_type"],["BirthDate"],["date_time"],["only_time"],["date"], ["type"], ["name"], ["gender"], ["crime"], ["status"],["string"], ["city"], ["address"], ["duration"], ["post_code"], ["year"], ["weekday"], ["monthday"], ["month"], ["number"], ["age"],["coordinate"],["relationship"], ["population"], ["state"], ["school_name"], ["bank_name"], ["company_name"], ["ratio"], ["country"], ["description"], ["id"], ["region"], ["continent"], ["district"], ["float"], ["integer"]]
        list_x = ["only_date","crime_type","BirthDate","date_time","only_time","date", "type", "name", "gender", "crime", "status","string", "city", "address", "duration", "post_code", "year", "weekday", "monthday", "month", "number", "age", "coordinate", "relationship", "population", "state", "school_name", "bank_name", "company_name", "ratio", "country", "description", "id", "region", "continent", "district", "float", "integer"]
        enc.fit(X)
        encoded_semantic = enc.transform([["only_date"],["crime_type"],["BirthDate"],["date_time"],["only_time"],["date"], ["type"], ["name"], ["gender"], ["crime"], ["status"],["string"], ["city"], ["address"], ["duration"], ["post_code"], ["year"], ["weekday"], ["monthday"], ["month"], ["number"], ["age"],["coordinate"],["relationship"], ["population"], ["state"], ["school_name"], ["bank_name"], ["company_name"], ["ratio"], ["country"], ["description"], ["id"], ["region"], ["continent"], ["district"], ["float"], ["integer"]]).toarray()
        for key, value in semantic_types.items():
                for k, v in value.items():
                    for count, value_list in enumerate(list_x):
                        if v == value_list:

                            semantic_types[key][k]= encoded_semantic[count]
        #print(semantic_types)
        return semantic_types


   
    def read_data(self, file_path):
        data_file_path = file_path
        columns =[] #contains all the columns from all the datatsets
        gt = [] #contains rount truths of all datasets
        agg = [] 
        index = [] #contains index of the all column combinations from all datasets
        single_field_features =[]
        all_features = [] #contains all the single filed features for all the columns
        agg_features =[]
        semantic_column_data = {}
        #get semantic datatypes of columns first
        semantic = self.get_semantic_datatypes()
        
        #iterate over the folder and access each file to process
        for filename in os.listdir(data_file_path):
            single_file_path = os.path.join(data_file_path, filename)
            if os.path.isfile(single_file_path):    

                #read file
                df_csv = pd.read_excel(single_file_path)

                #get fields 
                pd.set_option('display.max_columns', 500)
                fields = []
                # tuple ('container, y', {'uid':'', 'order': 7, 'data': []})
                for (columnName, columnData) in df_csv.iteritems():
                    #print((columnData[0]), columnData.dtypes, type(
                     #   columnData[0]), type(columnData), columnName)

                    if (isinstance(columnData[0], pd._libs.tslibs.timestamps.Timestamp)):
                        # df_csv[columnName]=df_csv[columnName].to_frame().applymap(str)
                        # data_dict = {'uid': get_random_string(6), 'order': random.randint(0, 9),
                        #              'data': df_csv[columnName].values.astype(str).tolist()}
                        data_dict = {'uid': self.get_random_string(6), 'order': random.randint(0, 9),
                                    'data': columnData.to_frame().applymap(str).values.tolist()}
                        #print('xaaaaaaaaaa1', type(columnData[0]), columnData[0])

                        fields.append((columnName, data_dict))

                    else:
                        data_dict = {'uid': self.get_random_string(6), 'order': random.randint(0, 9),
                                    'data': columnData.to_list()}
                        fields.append((columnName, data_dict))


                single_field_features = extract_single_field_features(fields, self.get_random_string(7) + ":" + str(random.randint(0, 9)), MAX_FIELDS=len(fields))[0]
                all_features.extend(single_field_features)
      
                #normalize the features 
                scaler = StandardScaler()
                for feature in all_features:
                    for k, v in feature.items():
                        if isinstance(v, (int, float)):
                            feature[k] = scaler.fit_transform([[v]])[0][0]

                
                columns.extend(df_csv.columns.tolist())
                column_names = df_csv.columns.tolist()

                # Extract all possible pairs of columns
                pairs = [(col1, col2) for i, col1 in enumerate(column_names)
                        for col2 in column_names[i+1:]]
                # Calculate the number of pairs using the formula C(n, 2)
                num_pairs = math.comb(len(column_names), 2)

                #assign agg features of the dataset to all the pairs indices
                agg =extract_aggregate_single_field_features(single_field_features)
                while num_pairs >  0:
                    agg_features.append(agg)
                    num_pairs -= 1

                # Store the index combinations
                index_combinations = []
                index_combina =[]

                for col1, col2 in pairs:
                    col1_index = columns.index(col1)
                    col2_index = columns.index(col2)
                    col1_in = column_names.index(col1)
                    col2_in = column_names.index(col2)
                    index_combina.append((col1_in, col2_in))
                    index_combinations.append((col1_index, col2_index))
                    
                index.extend(index_combinations)

                #semantic column features
                for file_path, file_columns in semantic.items():
                    semantic_filename, extension1 = os.path.splitext(os.path.basename(file_path))
                    original_filename, extension2 = os.path.splitext(os.path.basename(single_file_path))

                    if semantic_filename == original_filename:
                        for column_name, column_values in file_columns.items():
                            if column_name not in semantic_column_data:
                                semantic_column_data[column_name] = column_values
                            else:
                                repeated_column_name = f"{column_name}_{file_path}"
                                semantic_column_data[repeated_column_name] = semantic_column_data[column_name] + column_values
                                #del column_data[column_name]
          
                
                #get ground truths
                for key in ground_truth.adj_matrices:
                    if key in single_file_path:
                        gtruth =[]
                        df = ground_truth.adj_matrices.get(key)
                        for (col1_in, col2col2_in) in index_combina:
                            element = df.iloc[col1_in, col2_in]
                            gtruth.append(element)
       
                gt.extend(gtruth)   

        #input()
        
        #get semantic features #get semantic features 
        #semantic = self.get_semantic_datatypes()  
                
        """for file_path, file_columns in semantic.items():
            for column_name, column_values in file_columns.items():
                if column_name not in column_data:
                    # If the column name is not in the dictionary, add it with the
                    # column values from the current file
                    column_data[column_name] = column_values
                else:
                # If the column name is already in the dictionary, repeat the
                # column name and add it with the column values from the current file
                    repeated_column_name = f"{column_name}_{file_path}"
                    column_data[repeated_column_name] = column_data[column_name] + column_values
                    del column_data[column_name]

        # Now `column_data` contains all columns and their respective arrays,
        # with repeated column names if necessary """
        print(len(semantic_column_data))                 
        print(len(columns))        
        return all_features, gt, columns, index, agg_features, semantic_column_data




class SemanticTableDataset1(Dataset):

    def __init__(self, file_path):
        self.features, self.grtruth, self.allcolumns , self.index, self.agg_features= self.read_data(file_path)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        x1, x2 = self.index[idx]
    
        x1_single_comulmn_features = (self.features[x1])
        x2_single_comulmn_features = (self.features[x2])
        aggregate_single_features_x1_x2 = extract_aggregate_single_field_features(
            [x1_single_comulmn_features, x2_single_comulmn_features])


        #agg features of the dataset
        agg_features_dataset =self.agg_features[idx]

        #get semantic features
        semantic_features_x1_x2 = self.allcolumns[x1]

        x1_final_features = torch.tensor(
            [0 if v is None else v for v in x1_single_comulmn_features.values()][2:], dtype=torch.float64)
        x2_final_features = torch.tensor(
            [0 if v is None else v for v in x2_single_comulmn_features.values()][2:], dtype=torch.float64)
        agg_features = torch.tensor(
            [0 if v is None else v for v in aggregate_single_features_x1_x2.values()], dtype=torch.float64)
        agg_features_dataset = torch.tensor(
            [0 if v is None else v for v in agg_features_dataset.values()], dtype=torch.float64)

       
        gr_t = self.grtruth[idx]
     
        return x1_final_features, x2_final_features, agg_features, gr_t, agg_features_dataset

    def get_random_string(self, length):
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))
        # print("Random string of length", length, "is:", result_str)

    def get_semantic_datatypes(self, file_path):
        with open("/home/nehaj/infinity/altair/data_test/output.pkl", 'rb') as handle:
            semantic_types = pickle.load(handle)
            self.semantic_onehot_encoding(semantic_types)
        return semantic_types

    def semantic_onehot_encoding(self,semantic_types):
        enc = OneHotEncoder(handle_unknown='ignore')
        X = [['Male', 1], ['Female', 3], ['Female', 2]]


   
    def read_data(self, file_path):
        data_file_path = file_path
        columns =[]
        gt = []
        agg = []
        index = []
        single_field_features =[]
        all_features = []
        agg_features =[]
        semantic_types =[]
        for filename in os.listdir(data_file_path):
            single_file_path = os.path.join(data_file_path, filename)
            if os.path.isfile(single_file_path):    

                #read file
                df_csv = pd.read_excel(single_file_path)

                #get fields 
                pd.set_option('display.max_columns', 500)
                fields = []
                # tuple ('container, y', {'uid':'', 'order': 7, 'data': []})
                for (columnName, columnData) in df_csv.iteritems():
                    #print((columnData[0]), columnData.dtypes, type(
                     #   columnData[0]), type(columnData), columnName)

                    if (isinstance(columnData[0], pd._libs.tslibs.timestamps.Timestamp)):
                        # df_csv[columnName]=df_csv[columnName].to_frame().applymap(str)
                        # data_dict = {'uid': get_random_string(6), 'order': random.randint(0, 9),
                        #              'data': df_csv[columnName].values.astype(str).tolist()}
                        data_dict = {'uid': self.get_random_string(6), 'order': random.randint(0, 9),
                                    'data': columnData.to_frame().applymap(str).values.tolist()}
                        #print('xaaaaaaaaaa1', type(columnData[0]), columnData[0])

                        fields.append((columnName, data_dict))

                    else:
                        data_dict = {'uid': self.get_random_string(6), 'order': random.randint(0, 9),
                                    'data': columnData.to_list()}
                        fields.append((columnName, data_dict))


                single_field_features = extract_single_field_features(fields, self.get_random_string(7) + ":" + str(random.randint(0, 9)), MAX_FIELDS=len(fields))[0]
                all_features.extend(single_field_features)
      
                #normalize the features 
                scaler = StandardScaler()
                for feature in all_features:
                    for k, v in feature.items():
                        if isinstance(v, (int, float)):
                            feature[k] = scaler.fit_transform([[v]])[0][0]

                
                columns.extend(df_csv.columns.tolist())
                column_names = df_csv.columns.tolist()

                # Extract all possible pairs of columns
                pairs = [(col1, col2) for i, col1 in enumerate(column_names)
                        for col2 in column_names[i+1:]]
                # Calculate the number of pairs using the formula C(n, 2)
                num_pairs = math.comb(len(column_names), 2)

                #assign agg features of the dataset to all the pairs indices
                agg =extract_aggregate_single_field_features(single_field_features)
                while num_pairs >  0:
                    agg_features.append(agg)
                    num_pairs -= 1

                # Store the index combinations
                index_combinations = []
                index_combina =[]

                for col1, col2 in pairs:
                    col1_index = columns.index(col1)
                    col2_index = columns.index(col2)
                    col1_in = column_names.index(col1)
                    col2_in = column_names.index(col2)
                    index_combina.append((col1_in, col2_in))
                    index_combinations.append((col1_index, col2_index))
                    
                index.extend(index_combinations)
                
                #get ground truths
                for key in ground_truth.adj_matrices:
                    if key in single_file_path:
                        gtruth =[]
                        df = ground_truth.adj_matrices.get(key)
                        for (col1_in, col2col2_in) in index_combina:
                            element = df.iloc[col1_in, col2_in]
                            gtruth.append(element)
       
                gt.extend(gtruth)   

                #get semantic features 
                #semantic = self.get_semantic_datatypes(single_file_path)
                #semantic_types.append(semantic)
            
           
        return all_features, gt, columns, index, agg_features
   

if __name__ == '__main__':

    #training_data_single = SingleTableDataset(
    #    "/home/nehaj/infinity/altair/clean_data/", ground_truth.adj_matrices)

    #training_data_all = TableDataset(
    #    "/home/nehaj/infinity/altair/clean_data/")

    training_data_semantic = SemanticTableDataset(
        "/home/nehaj/infinity/altair/clean_data/")

    train_dataloader = DataLoader(training_data_semantic, batch_size=1, shuffle=True)
    x1_features, x2_features, agg_features, gr_t = next(
        iter(train_dataloader))
    
    #print("THIS IS THE TENSORS")
    #print(x1_features.shape, '\n')
    #print(x2_features.shape, '\n')
    #print(agg_features.shape, '\n')
    #nt = torch.cat((x1_features, x2_features, agg_features), dim=1)
    #print(nt.shape)

