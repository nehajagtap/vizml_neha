import os
import pandas as pd
from predict import evaluate
import numpy as np
from dataprep.clean.clean_date import validate_date
from dataprep.clean import clean_df
import wordninja
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# nt =["company","companies","business","school","schools","population","state", "st","relationship","relations","date","birthdate","dob","doe","type","name","gender","sex","crime", "fraud","status","cities","city","addr","address","street","duration","postalcode","postcode","zip","code","year","yr","day","month","number","num","no","age","lat","lon","latitude","longitude"]

def string_semantic(df,col_types,col_name,index_num,res):
    date_str = ["date","birthdate","year","dob","doe","yr"]
    nt_date = get_nt_str(date_str)
    type_str = ["type"]
    nt_type = get_nt_str(type_str)
    name_str = ["name"]
    nt_name = get_nt_str(name_str)
    gender_str = ["gender","sex"]
    nt_gender = get_nt_str(gender_str)
    crime_str = ["crime", "fraud"]
    status_str = ["status"]
    nt_status = get_nt_str(status_str)
    city_str = ["cities","city"]
    nt_city = get_nt_str(city_str)    
    address_str = ["addr","address","street"]
    nt_address = get_nt_str(address_str)  
    duration_str = ["duration"]
    nt_duration = get_nt_str(duration_str)
    relationship_str = ["relationship","relations"]
    nt_relationship = get_nt_str(relationship_str)
    state_str = ["state", "st"]
    nt_state = get_nt_str(state_str)    
    school_str = ["school","schools"]
    nt_school = get_nt_str(school_str)    
    company_str = ["company","companies","business"]
    nt_company = get_nt_str(company_str)
    bank_str = ["bank","banks"] 
    nt_bank = get_nt_str(bank_str)
    country_str = ["country","countires"]
    nt_country = get_nt_str(country_str)
    description_str = ["description"]
    nt_description = get_nt_str(description_str)
    region_str = ["regions","region"]
    nt_region = get_nt_str(region_str)
    continent_str = ["continent","continents"]
    nt_continent = get_nt_str(continent_str)
    district_str = ["district"]
    nt_district = get_nt_str(district_str)

    if is_city(nt_city,df,col_name,index_num,res,city_str):
        col_types[col_name] = "city"
    elif is_address(nt_address,df,col_name,index_num,res,address_str):
        col_types[col_name] = "address"
    elif is_date(nt_date,df,col_name,index_num,res,date_str):
        if res[index_num] == "birthPlace": 
            col_types[col_name] = "BirthDate"
        else :
            col_types[col_name] = get_date_type(df[col_name])
    elif is_name(nt_name,df,col_name,index_num,res,name_str):
        col_types[col_name] = "name"
    elif is_type(nt_type,df,col_name,index_num,res,type_str):
        col_types[col_name] = "type"
    elif is_duration(nt_duration,df,col_name,index_num,res,duration_str):
        col_types[col_name] = "duration"
    elif is_gender(nt_gender,df,col_name,index_num,res,gender_str):
        col_types[col_name] = "gender"
    elif is_status(nt_status,df,col_name,index_num,res,status_str):
        col_types[col_name] = "status"
    elif is_crime(df,col_name,index_num,res,crime_str):
        col_types[col_name] = "crime_type"
    elif is_relationship(nt_relationship,df,col_name,index_num,res,relationship_str):
        col_types[col_name] = "relationship"
    elif is_state(nt_state,df,col_name,index_num,res,state_str):
        col_types[col_name] = "state"
    elif is_school(nt_school,df,col_name,index_num,res,school_str):
        col_types[col_name] = "school_name"
    elif is_company(nt_company,df,col_name,index_num,res,company_str):
        col_types[col_name] = "company_name"
    elif is_bank(nt_bank,df,col_name,index_num,res,bank_str):
        col_types[col_name] = "bank_name"
    elif is_country(nt_country,df,col_name,index_num,res,country_str):
        col_types[col_name] = "country"
    elif is_description(nt_description,df,col_name,index_num,res,description_str):
        col_types[col_name] = "description"  
    elif is_region(nt_region,df,col_name,index_num,res,region_str):
        col_types[col_name] = "region"
    elif is_continent(nt_continent,df,col_name,index_num,res,continent_str):
        col_types[col_name] = "continent" 
    elif is_district(nt_district,df,col_name,index_num,res,district_str):
        col_types[col_name] = "district"        
    else:
        col_types[col_name] = "string"
    return col_types

def integer_semantic(df,col_types,col_name,index_num,res):
    zipcode = ["postalcode","postcode","zip"]
    nt_location = get_nt_str(zipcode) 
    year_str = ["year","yr"]
    nt_year = get_nt_str(year_str)
    day_str = ["day"]
    nt_weekday = get_nt_str(day_str)
    month_str = ["month"]
    number_str = ["number","num","no"]
    nt_number = get_nt_str(number_str)    
    age_str = ["age"]
    nt_age  = get_nt_str(age_str)    
    population_str = ["population"]
    nt_population = get_nt_str(population_str)
    id_str = ["id","rank","ids"]
    nt_id = get_nt_str(id_str)
    
    if is_location(nt_location,df,col_name,index_num,res,zipcode):
        col_types[col_name] = "post_code"
    elif is_number(nt_number,df,col_name,index_num,res,number_str):
        col_types[col_name] = "number"
    elif is_year(nt_year,df,col_name,index_num,res,year_str):
        col_types[col_name] = "year"
    elif is_weekday(nt_weekday,df,col_name,index_num,res,day_str):
        col_types[col_name] = "weekday"
    elif is_monthday(nt_weekday,df,col_name,index_num,res,day_str):
        col_types[col_name] = "monthday"
    elif is_month(df,col_name,index_num,res,month_str):
        col_types[col_name] = "month"
    elif is_age(nt_age,df,col_name,index_num,res,age_str):
        col_types[col_name] = "age"
    elif is_population(nt_population,df,col_name,index_num,res,population_str):
        col_types[col_name] = "population"   
    elif is_id(nt_id,df,col_name,index_num,res,id_str):
        col_types[col_name] = "id"                 
    else:
        col_types[col_name] = "integer"
    return col_types   

def float_cordinates_semantic(df,col_types,col_name,index_num,res):
    number_str = ["number","num","no"]
    nt_number = get_nt_str(number_str)
    cordinates_str = ["lat","lon","latitude","longitude"]
    nt_cordinates = get_nt_str(cordinates_str)
    ratio_str = ["ratio"]
    nt_ratio = get_nt_str(ratio_str)
    population_str = ["population"]
    nt_population = get_nt_str(population_str)

    if is_cordinates(nt_cordinates,df,col_name,index_num,res,cordinates_str):
        col_types[col_name] = "coordinate"
    elif is_number(nt_number,df,col_name,index_num,res,number_str):
        col_types[col_name] = "number"
    elif is_ratio(nt_ratio,df,col_name,index_num,res,ratio_str):
        col_types[col_name] = "ratio"
    elif is_population(nt_population,df,col_name,index_num,res,population_str):
        col_types[col_name] = "population" 
    else:
        col_types[col_name] = "float"
    return col_types

def is_district(nt_district,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (is_district_column(df[col]) or check_list(words,word)) and check_list(nt_district,word) == False:
        return True
    return False

def is_continent(nt_continent,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (is_continent_column(df[col]) or check_list(words,word)) and check_list(nt_continent,word) == False:
        return True
    return False

def is_region(nt_region,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (is_region_column(df[col]) or check_list(words,word)) and check_list(nt_region,word) == False:
        return True
    return False

def is_id(nt_id,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    # if check_list(words,word) and res[indx] == "rank" and check_list(nt_id,word) == False:
    if check_list(words,word) and res[indx] == "rank":
        return True
    return False

def is_description(nt_description,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (check_list(words,word) or res[indx] == "description") and check_list(nt_description,word) == False:
        return True
    return False

def is_country(nt_country,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (is_country_column(df[col]) or check_list(words,word) or res[indx] == "country") and check_list(nt_country,word) == False:
        return True
    return False

def is_ratio(nt_ratio,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if check_list(words,word) and check_list(nt_ratio,word) == False:
        return True
    return False

def is_bank(nt_bank,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (is_bank_column(df[col]) or check_list(words,word)) and check_list(nt_bank,word) == False:
        return True
    return False

def is_company(nt_company,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (is_company_column(df[col]) or check_list(words,word)) and check_list(nt_company,word) == False:
        return True
    return False

def is_school(nt_school,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (is_school_column(df[col]) or check_list(words,word)) and check_list(nt_school,word) == False:
        return True
    return False

def is_state(nt_state,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if ((res[indx] == "state" and check_list(words,word)) or (res[indx] == "region" and check_list(words,word)) or is_state_column(df[col])) and check_list(nt_state,word) == False:
        return True
    return False

def is_population(nt_population,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (is_population_column(df[col]) and check_list(words,word)) and check_list(nt_population,word) == False:
        return True
    return False

def is_relationship(nt_relationsip,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if check_list(words,word):
        return True
    return False

def is_age(nt_age,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if ((res[indx] == "age" and check_list(words,word)) or (is_age_column(df[col]) and check_list(words,word))) and check_list(nt_age,word) == False:
        return True
    return False


def is_crime(df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if ((res[indx] == "type" and check_list(words,word)) or check_list(words,word)):
        return True
    return False


def is_status(nt_status,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (res[indx] == "status" or check_list(words,word)) and check_list(nt_status,word) == False:
        return True
    return False


def is_gender(nt_gender,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (res[indx] == "gender" or check_list(words,word) )and check_list(nt_gender,word) == False:
        return True
    return False
    
def is_duration(nt_duration,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (res[indx] == "duration" or check_list(words,word))and(check_list(nt_duration,word) == False):
        return True
    return False
    

def is_cordinates(nt_cordinates,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if check_list(words,word) and check_list(nt_cordinates,word) == False:
        return True
    return False

def is_month(df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if check_list(words,word) and is_month_column(df[col]):
        return True
    return False

def is_weekday(nt_weekday,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if ((check_list(words,word) or res[indx] == "day") and is_weekday_column(df[col])) and check_list(nt_weekday,word) == False:
        return True
    return False

def is_monthday(nt_monthdays,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if ((check_list(words,word) or res[indx] == "day") and is_monthday_column(df[col])) and check_list(nt_monthdays,word) == False:
        return True
    return False

def is_year(nt_year,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if ((check_list(words,word) or res[indx] == "year") and is_year_column(df[col])) and check_list(nt_year,word) == False:
        return True
    return False

def is_number(nt_number,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if ((check_list(words,word) and res[indx] == "code") or ((res[indx] == "rank")and(check_list(words,word)))) and check_list(nt_number,word) == False:
        return True
    return False
        
def is_location(nt_location,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (check_list(words,word) or res[indx] == "location") and check_list(nt_location,word) == False:
        return True
    return False

def is_type(nt_type,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (res[indx] == "type" or check_list(words,word)) and check_list(nt_type,word) == False:
        return True
    return False

def is_date(nt_date,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if ((my_validate_date(df[col]) or check_list(words,word) or res[indx] == "year") and res[indx] != "duration") and check_list(nt_date,word) == False:
        return True
    return False

def is_address(nt_address,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (res[indx] == "location" or check_list(words,word)) and check_list(nt_address,word) == False:
        return True
    return False


def is_city(nt_city,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (is_city_column(df[col]) or res[indx] == "city" or check_list(words,word)) and check_list(nt_city,word) == False:
        return True
    return False

def is_name(nt_name,df,col,indx,res,words):
    word = wordninja.split(col.lower())
    if (res[indx] == "name" or check_list(words,word)) and check_list(nt_name,word) == False:
        return True
    return False

def check_list(list_1,list_2):
    for elem in list_1:
        if elem in list_2:
            return True
    return False

def is_month_column(col):
    max = col.max()
    min = col.min()
    if max <= 12:
        if min >= 1:
            return True
    return False

def is_age_column(col):
    max = col.max()
    min = col.min()
    if max <= 150:
        if min >= 1:
            return True
    return False

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

def is_year_column(col):
    max = col.max()
    if max <= 2023:
        return True
    return False

def is_population_column(col):
    min = col.min()
    if min >= 10000:
        return True
    return False  

def is_city_column(col):
    cities = pd.read_csv("cities.csv").values

    city_string = [s[0] for s in cities]
    threshold = 0.7
    no_of_samples = len(col)
    samples = col.sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in city_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True
    return False

def is_bank_column(col):
    bank = pd.read_csv("banks.csv").values
    bank_string = [s[0] for s in bank]
    threshold = 0.7
    no_of_samples = len(col)
    samples = col.sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in bank_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True
    return False


def is_school_column(col):
    schools = pd.read_csv("schools.csv").values
    school_string = [s[0] for s in schools]
    threshold = 0.7
    no_of_samples = len(col)
    samples = col.sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in school_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True
    return False

def is_continent_column(col):
    region = pd.read_csv("continents.csv").values
    region_string = [s[0] for s in region]
    threshold = 0.7
    no_of_samples = len(col)
    samples = col.sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in region_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True
    return False

def is_region_column(col):
    region = pd.read_csv("regions.csv").values
    region_string = [s[0] for s in region]
    threshold = 0.7
    no_of_samples = len(col)
    samples = col.sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in region_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True
    return False

def is_district_column(col):
    region = pd.read_csv("district.csv").values
    region_string = [s[0] for s in region]
    threshold = 0.7
    no_of_samples = len(col)
    samples = col.sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in region_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True
    return False

def is_company_column(col):
    company = pd.read_csv("companies.csv").values
    company_string = [s[0] for s in company]
    threshold = 0.7
    no_of_samples = len(col)
    samples = col.sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in company_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True
    return False

def is_country_column(col):
    country = pd.read_csv("country.csv").values
    country_string = [s[0] for s in country]
    threshold = 0.7
    no_of_samples = len(col)
    samples = col.sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in country_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True
    return False

def is_state_column(col):
    state = pd.read_csv("states.csv").values
    state_string = [s[0] for s in state]
    threshold = 0.7
    no_of_samples = len(col)
    samples = col.sample(no_of_samples).values.tolist()
    count = 0
    for sample in samples:
        if sample in state_string:
            count = count + 1
    if count / no_of_samples >= threshold:
        return True
    return False

def my_validate_date(col):
    threshold = 0.9
    counts = validate_date(col).value_counts()
    if len(counts) == 1:
        return counts.index[0]  # Either all are True or False, return the first element
    else:
        return counts[1] / len(col) > threshold

def get_date_type(col):
    try:
        dt = pd.to_datetime(col)
        if (dt.dt.floor('d') == dt).all():
            return "only_date"
        else:
            dates = dt.dt.date
            if np.all(dates == dates[0]):
                return "only_time"
            else:
                return "date_time"
    except:
        return "date"

def get_nt_str(nt_list):
    total_list =["district","continent","continents","regions","region","id","rank","ids","description","country","countires","ratio","bank","banks","company","companies","business","school","schools","population","state", "st","relationship","relations","date","birthdate","dob","doe","type","name","gender","sex","crime", "fraud","status","cities","city","addr","address","street","duration","postalcode","postcode","zip","year","yr","day","month","number","num","no","age","lat","lon","latitude","longitude"]
    for i in nt_list:
        total_list.remove(i)
    return total_list

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms



