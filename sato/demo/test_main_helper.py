
from test_helper import *

        
def test_city(file_path,column_name):
    city_str = ["cities","city"]
    word_str = "city"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"string"):
                if is_city_col(df,col_name) or check_list(city_str,word) or check_res_prediction(res,idx,word_str):
                    result = True
            return(result)
        idx += 1

def test_address(file_path,column_name):
    address_str = ["addr","address","street"]
    word_str = "location"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"string"):
                if check_list(address_str,word) or check_res_prediction(res,idx,word_str):
                    result = True
            elif check_semintic_type(col_type,"address"):
                result = True
            return(result)
        idx += 1


def test_date_and_time(file_path,column_name):
    date_str = ["date","birthdate","year","dob","doe"]
    word_str1 = "year"
    word_str2 = "birthPlace"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"string"):
                if my_validate_date(df[col_name]) or check_list(date_str,word) or check_res_prediction(res,idx,word_str1) or check_res_prediction(res,idx,word_str2):
                    result = True
                elif check_res_prediction(res,idx,"duration"):
                    result = False
            return(result)
        idx += 1

def test_name(file_path,column_name):
    name_str = ["name","surname","firstname"]
    word_str = "name"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"string"):
                if check_list(name_str,word) or check_res_prediction(res,idx,word_str):
                    result = True
            return(result)
        idx += 1


def test_type(file_path,column_name):
    type_str = ["type"]
    word_str = "type"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"string"):
                if check_list(type_str,word) or check_res_prediction(res,idx,word_str):
                    result = True
            return(result)
        idx += 1

def test_location(file_path,column_name):
    zipcode = ["postalcode","postcode","zip","code"]
    word_str = "location"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"integer"):
                if check_list(zipcode,word) or check_res_prediction(res,idx,word_str):
                    result = True
            return(result)
        idx += 1

def test_number(file_path,column_name):
    number_str = ["number","num","no"]
    word_str = "code"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"integer") or check_semintic_type(col_type,"floating"):
                if check_list(number_str,word) or check_res_prediction(res,idx,word_str) or ((check_res_prediction(res,idx,"rank")and(check_list(number_str,word)))):
                    result = True
            return(result)
        idx += 1
        
def test_year(file_path,column_name):
    year_str = ["year","yr"]
    word_str = "year"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"integer"):
                if check_list(year_str,word) or check_res_prediction(res,idx,word_str):
                    result = True
            return(result)
        idx += 1
        
def test_day(file_path,column_name):
    day_str = ["day"]
    word_str = "day"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"integer"):
                if (check_list(day_str,word) or check_res_prediction(res,idx,word_str)) and is_weekday_column(df[column_name]):
                    result = True
                elif (check_list(day_str,word) or check_res_prediction(res,idx,word_str)) and is_monthday_column(df[column_name]):
                    result = True
            return(result)
        idx += 1

def test_month(file_path,column_name):
    month_str = ["month"]
    word_str = "month"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"integer"):
                if check_list(month_str,word) and is_month_column(df[column_name]):
                    result = True
            return(result)
        idx += 1
        
def test_cordinates(file_path,column_name):
    cordinates_str = ["lat","lon","latitude","longitude"]
    word_str = "position"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"floating") or check_semintic_type(col_type,"coordinate"):
                if check_list(cordinates_str,word) or check_res_prediction(res,idx,word_str):
                    result = True
            return(result)
        idx += 1

def test_duration(file_path,column_name):
    duration_str = ["duration", "timeframe"]
    word_str = "duration"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"string") or check_semintic_type(col_type,"duration"):
                if (check_list(duration_str,word) and check_res_prediction(res,idx,word_str)) or check_list(duration_str,word):
                    result = True
            return(result)
        idx += 1  
        
def test_gender(file_path,column_name):
    gender_str = ["gender","sex"]
    word_str = "gender"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"string"):
                if check_list(gender_str,word) or check_res_prediction(res,idx,word_str):
                    result = True
            return(result)
        idx += 1      

def test_status(file_path,column_name):
    status_str = ["status"]
    word_str = "status"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"string"):
                if check_list(status_str,word) or check_res_prediction(res,idx,word_str):
                    result = True
            return(result)
        idx += 1    

def test_crime(file_path,column_name):
    crime_str = ["crime", "fraud"]
    word_str = "crime"
    idx = 0 
    df,result,res,inferred_type = initialize(file_path)
    for col_name, col_type in inferred_type.iterrows():
        if col_name == column_name:
            word = word_split(col_name)
            if check_semintic_type(col_type,"string"):
                if check_list(crime_str,word) or check_res_prediction(res,idx,word_str):
                    result = True
            return(result)
        idx += 1      