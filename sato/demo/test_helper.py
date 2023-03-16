from predict import evaluate
import pandas as pd
from dataprep.clean import clean_df
import wordninja
from dataprep.clean.clean_date import validate_date


def initialize(file_path):
    df = pd.read_csv(file_path)
    result = False
    res = evaluate(df)
    inferred_type, _ = clean_df(df)
    return df,result,res,inferred_type

def check_list(list_1,list_2):
    for elem in list_1:
        if elem in list_2:
            return True
    return False

def is_city_col(df,col_name):
    cities = pd.read_csv("cities.csv").values
    city_string = [s[0] for s in cities]
    threshold = 0.7
    no_of_samples = len(df[col_name])
    samples = df[col_name].sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in city_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True

def check_res_prediction(res,idx,word_str):
    if res[idx] == word_str:
        return True
    return False

def word_split(col_name):
    word = wordninja.split(col_name.lower())
    return word

def check_semintic_type(col_type,desired_type):
    print(col_type['semantic_data_type'])
    if col_type['semantic_data_type'] == desired_type:
        return True
    return False

def my_validate_date(col):
    threshold = 0.9
    counts = validate_date(col).value_counts()
    if len(counts) == 1:
        return counts.index[0]  # Either all are True or False, return the first element
    else:
        return counts[1] / len(col) > threshold
    
def is_weekday_column(col):
    max = col.max()
    min = col.min()
    if max <= 7:
        if min >= 1:
            return True
    return False

def is_monthday_column(col):
    max = col.max()
    min = col.min()
    if max <= 31:
        if min >= 1:
            return True
    return False

def is_month_column(col):
    max = col.max()
    min = col.min()
    if max <= 12:
        if min >= 1:
            return True
    return False