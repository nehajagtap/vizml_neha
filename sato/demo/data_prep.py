import altair as alt
import numpy as np
import pandas as pd
from dataprep.clean.clean_date import validate_date
from dataprep.clean import clean_df
import numbers
import json

class ColumnTypes:
    threshold = 0.9

    @staticmethod
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
            return "duration"

    @staticmethod
    def my_validate_date(col):
        counts = validate_date(col).value_counts()
        if len(counts) == 1:
            return counts.index[0]  # Either all are True or False, return the first element
        else:
            return counts[0] / len(col) > ColumnTypes.threshold

    @staticmethod
    def is_city_column(col):
        cities = pd.read_csv("cities.csv").values

        city_string = [s[0] for s in cities]
        threshold = 0.7
        no_of_samples = 500
        samples = col.sample(no_of_samples).values.tolist()
        count = 0
        for sample in samples:
            if sample in city_string:
                count = count + 1
        if count / no_of_samples >= threshold:
            return True
        return False

    @staticmethod
    def is_postcode_column(col):
        no_of_samples = 500
        samples = col.sample(no_of_samples).values.tolist()

    @staticmethod
    def get_column_types(df):
        inferred_type, _ = clean_df(df)
        col_types = {}
        for index, row in inferred_type.iterrows():
            if row['semantic_data_type'] == "string":
                if "postcode" in index.lower():
                    col_types[index] = "post_code"
                elif ColumnTypes.is_city_column(df[index]):
                    col_types[index] = "city"
                elif ColumnTypes.my_validate_date(df[index]):
                    col_types[index] = ColumnTypes.get_date_type(df[index])
                else:
                    col_types[index] = "string"
            else:
                col_types[index] = row['semantic_data_type']

        return col_types

    @staticmethod
    def merge_similar_columns(df, col_types, prefixes=['Caller', 'Called']):
        type_dict = {}
        for column_name, col_type in col_types.items():
            if col_type in type_dict:
                type_dict[col_type].append(column_name)
            else:
                type_dict[col_type] = [column_name]

        similar_columns = []
        for column_type, columns in type_dict.items():
            if len(columns) < 2:
                continue
            for prefix in prefixes:
                common_prefix = []
                for column_name in columns:
                    if column_name.startswith(prefix):
                        common_prefix.append(column_name)
                similar_columns.append(common_prefix)
        return similar_columns


def interpretable_columns(data_frame):
    interpretable_columns_indices = []
    for idx, i in enumerate(data_frame.nunique().values):
        if (i <= 20 and i > 1):
            interpretable_columns_indices.append(idx)

    return interpretable_columns_indices


def preprocessing_df(data_frame):
    similiar_cols = ColumnTypes.merge_similar_columns(data_frame, ColumnTypes.get_column_types(data_frame))
    for i, similiar_col in enumerate(similiar_cols):
        if len(similiar_col) > 1:
            for col in similiar_col:
                data_frame[col] = data_frame[col].astype(str)
            new_column = data_frame[similiar_col].agg(' '.join, axis=1).to_frame()
            new_column.columns = ['temp1{}'.format(i)]
            data_frame = pd.concat([data_frame, new_column], axis=1, join="inner")
            data_frame.drop(
                similiar_col, axis=1, inplace=True)
    return data_frame


def horizontal_bar(name, data_frame, sort_x=None, sort_y=None):
    return alt.Chart(data_frame).mark_bar().encode(
        alt.Y(field=name, type="nominal", sort=sort_y),
        alt.X(field=name, aggregate="count", type="quantitative", sort=sort_x, title="count")
    ).configure_scale(
        bandPaddingInner=0.5
    )


def vertical_bar(name, data_frame, sort_x=None, sort_y=None):
    return alt.Chart(data_frame).mark_bar().encode(
        alt.X(field=name, type="nominal", sort=sort_y),
        alt.Y(field=name, aggregate="count", type="quantitative", sort=sort_x, title="count")
    ).configure_scale(
        bandPaddingInner=0.5
    )


def horizontal_bar_sorted_days(name, data_frame):
    days = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    sort_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    data_frame["Day"] = data_frame["Day"].map(days)
    return horizontal_bar(name, day, sort_y=sort_days)


def pie_chart(name, data_frame):
    # alt.data_transformers.enable('data_server')
    return alt.Chart(data_frame).mark_arc().encode(
        theta=alt.Theta(field=name, aggregate="count", type="quantitative"),
        color=alt.Color(field=name, type="nominal"),
    )


def temporal_bar_graph(name, data_frame):
    return alt.Chart(data_frame).mark_bar().encode(
        alt.X(field=name, timeUnit="hours", type="ordinal", title='hour'),
        alt.Y(field=name, aggregate="count", type="quantitative", title="count")).configure_scale(
        bandPaddingInner=0.5
    ).configure_scale(
        bandPaddingInner=0.5
    )


def line_graph(name, data_frame):
    return alt.Chart(data_frame).mark_line().encode(
        alt.X(field=name, timeUnit="month", type="ordinal", title='month'),
        alt.Y(field=name, aggregate="count", type="quantitative", title="count")
    )

def json_to_df(json_file):
    f = open(json_file)
    json_file = json.load(f)
    data = json_file['content']
    d = json.loads(data)
    data_columns = d['data']
    return pd.DataFrame(data_columns)

def generate_table(input_file, output_path):
    col_types = {}
    # df = pd.read_csv(input_file)
    df = json_to_df(input_file)
    for column in df.columns:
        value = str(df[column][0]).replace(" ", "")
        # print(column, value)
        try:
            int(value)
            col_types[column] = "integer"
        except:
            try:
                pd.to_datetime(value)
                col_types[column] = "datetime"
            except:
                try:
                    float(value)
                    col_types[column] = "float"
                except:
                    col_types[column] = "string"

    # print(col_types)
    for idx, column in enumerate(df.columns):

        if (col_types[column] == 'datetime'):
            if ('date' in column.lower()):
                df[column] = df[column].astype("datetime64")

            if ('time' in column.lower()):
                df[column] = pd.to_datetime(df[column], format='%I:%M:%S %p').dt.strftime('%H:%M:%S')
                # df[column] = pd.to_timedelta(df[column])

    datetime_columnname = 'Datetime'
    # df = pd.read_excel("abc.xlsx")
    for column in df.columns:
        # print(type(df[column][0]))

        if 'time' in column.lower():
            timecolumn = column

    for datecolumn in df.columns:
        if type(df[datecolumn][0]) == pd._libs.tslibs.timestamps.Timestamp:
            if df[datecolumn].dt.strftime('%H:%M').nunique() == 1:
                datetime = pd.to_datetime(
                    pd.to_datetime(df[datecolumn]).dt.strftime('%Y-%m-%d') + df[timecolumn],
                    format='%Y-%m-%d%H:%M:%S').to_frame()
                datetime.columns = [datetime_columnname]
                df.drop([datecolumn, timecolumn], axis=1, inplace=True)
                df = pd.concat([df, datetime], axis=1, join="inner")
                break
    df = preprocessing_df(df)
    
    interpretable_df = pd.concat([df[[datetime_columnname]], df.iloc[:, interpretable_columns(df)]], axis=1,
                                 join="inner")
    
    for column in interpretable_df.columns:

        # Time Data Charts ###
        if isinstance(interpretable_df[column][0], pd._libs.tslibs.timestamps.Timestamp):
            # Monthly Line Chart ###
            monthly_line_chart = line_graph(datetime_columnname, interpretable_df[column].to_frame())
            monthly_line_chart.save(output_path + '/' + str(column) + "_monthly_line_chart.html")
            monthly_line_chart.save(output_path + '/' + str(column) + "_monthly_line_test.json")

            # Hourly Bar Graph ###
            hourly_bar_chart = temporal_bar_graph(datetime_columnname, interpretable_df[column].to_frame())
            hourly_bar_chart.save(output_path + '/' + str(column) + "_hourly_bar_chart.html")
            hourly_bar_chart.save(output_path + '/' + str(column) + "_hourly_bar_chart.json")

            # String Data Charts ###
        elif isinstance(interpretable_df[column][0], str):
            # Pie chart for <10 unique value columns ###

            if interpretable_df[column].nunique() <= 10:
                string_pie_chart = pie_chart(column, interpretable_df[[column]])
                string_pie_chart.save(output_path + '/' + str(column) + "_string_pie_chart.html")
                string_pie_chart.save(output_path + '/' + str(column) + "_string_pie_chart.json")

                # Bar chart for <10 unique value columns ###

            else:
                string_vertical_chart = vertical_bar(column, interpretable_df[[column]])
                string_vertical_chart.save(output_path + '/' + str(column) + "_string_horizontal_chart.html")
                string_vertical_chart.save(output_path + '/' + str(column) + "_string_horizontal_chart.json")

                
                # if int(interpretable_df[column].astype(bytes).str.len().max()) > 15:
                #     string_horizontal_chart = horizontal_bar(column, interpretable_df[[column]])
                #     # string_horizontal_chart.save(output_path + '/' + str(column) + "_string_horizontal_chart.html")
                #     string_horizontal_chart.save(output_path + '/' + str(column) + "_string_horizontal_chart.json")

                #     # Vertical bar chart for shorter strings values ###
                # else:
                #     string_vertical_chart = vertical_bar(column, interpretable_df[[column]])
                #     # string_vertical_chart.save(output_path + '/' + str(column) + "_string_horizontal_chart.html")
                #     string_vertical_chart.save(output_path + '/' + str(column) + "_string_horizontal_chart.json")

            # Numeric Data Charts ###
        elif isinstance(interpretable_df[column][0], numbers.Number):
            # Pie chart for <10 unique value columns ###
            if interpretable_df[column].nunique() <= 10:
                numeric_pie_chart = pie_chart(column, interpretable_df[[column]])
                numeric_pie_chart.save(output_path + '/' + str(column) + "_numeric_pie_chart.html")
                numeric_pie_chart.save(output_path + '/' + str(column) + "_numeric_pie_chart.json")

                # Bar chart for <10 unique value columns ###
            else:
                numeric_vertical_chart = vertical_bar(column, interpretable_df[[column]])
                numeric_vertical_chart.save(output_path + '/' + str(column) + "_numeric_horizontal_chart.html")
                numeric_vertical_chart.save(output_path + '/' + str(column) + "_numeric_horizontal_chart.json")

                # # Horizontal bar chart for longer numeric values ###
                # if int(interpretable_df[column].astype(bytes).str.len().max()) > 15:
                #     numeric_horizontal_chart = horizontal_bar(column, interpretable_df[[column]])
                #     # numeric_horizontal_chart.save(output_path + '/' + str(column) + "_numeric_horizontal_chart.html")
                #     numeric_horizontal_chart.save(output_path + '/' + str(column) + "_numeric_horizontal_chart.json")

                #     # Vertical bar chart for shorter numeric values ###

                # else:
                #     numeric_vertical_chart = vertical_bar(column, interpretable_df[[column]])
                #     # numeric_vertical_chart.save(output_path + '/' + str(column) + "_numeric_horizontal_chart.html")
                #     numeric_vertical_chart.save(output_path + '/' + str(column) + "_numeric_horizontal_chart.json")


#generate_table("response.json", output_path=".")