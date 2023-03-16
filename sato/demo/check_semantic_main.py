from checking_semantic_methods import *


def get_column_types(df):
    res = evaluate(df)
    inferred_type, _ = clean_df(df)
    col_types = {}
    index_num = 0

    for col_name, col_type in inferred_type.iterrows():
        
        if col_type['semantic_data_type'] == "string":
            col_types = string_semantic(df,col_types,col_name,index_num,res)
            index_num +=1

        elif col_type['semantic_data_type'] == "integer":
            col_types = integer_semantic(df,col_types,col_name,index_num,res)
            index_num +=1
        
        elif col_type['semantic_data_type'] == "floating" or col_type['semantic_data_type'] == "coordinate":
            col_types = float_cordinates_semantic(df,col_types,col_name,index_num,res)
            index_num +=1

        else:
            col_types = string_semantic(df,col_types,col_name,index_num,res)
            # print(col_type['semantic_data_type'])
            index_num +=1


    return col_types



df = pd.read_csv("/media/saadnajib/56ce7057-5d73-4456-a6b0-7a325a03ba0f/saad/RIP_VIZML/vizml_ahmed/sato/demo/1962_2006_walmart_store_openings.csv")

r1 = get_column_types(df)
print(r1)

