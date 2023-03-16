from feature_exct import *
from single_field_features import *
from Helpers import *
import numpy as np
import pandas as pd
from type_detection import *
from collections import OrderedDict
from helpers import get_unique, list_entropy, gini, parse
import datetime
import statistics


def check_integrity(file_name):

    df = pd.read_excel(file_name)

    # Correction of Discrepencies in the CallDuration Column
    for column in df[['CallDuration']]:
        for idx, value in enumerate(df[column].values):
            if (type(value) == datetime.datetime):
                df[column][idx] = datetime.datetime.strftime(value, '%H:%M:%S')

     # print(columnData.dtypes, type(columnData[0]), type(columnData), columnName)
    for (columnName, columnData) in df.iteritems():
        if isinstance(columnData[0], datetime.time):
            if columnName == 'CallTime':
                df[columnName] = pd.to_datetime(pd.to_datetime(df['CallDate']).dt.strftime(
                    '%Y-%m-%d') + df[columnName].astype(str), format='%Y-%m-%d%H:%M:%S').to_frame()
                print('Time Done')
            if columnName == 'CallDuration':
                df[columnName] = pd.to_datetime(pd.to_datetime(df['CallDate']).dt.strftime(
                    '%Y-%m-%d') + df[columnName].astype(str), format='%Y-%m-%d%H:%M:%S').to_frame()

    # stat = OrderedDict([(f['name'], None)
    #                   for f in field_c_statistical_features_list + field_q_statistical_features_list])
    check_statistical_features = OrderedDict()

    for (colname, colval) in df.iteritems():

        list = colval.values.tolist()
        stat = OrderedDict([(f['name'], None)
                            for f in field_c_statistical_features_list + field_q_statistical_features_list])
        field_type, field_scores = detect_field_type(list)
        field_general_type = data_type_to_general_type[field_type]

        v = parse(list, field_type, field_general_type)
        v = np.ma.array(v).compressed()
        #print(colname, field_type)
        if not len(list):
            return stat
        if field_general_type == 'c':
            stat['list_entropy'] = list_entropy(list)
            value_lengths = [len(x) for x in list]
            # print(value_lengths)
            stat['mean_value_length'] = statistics.mean(value_lengths)
            #print(colname, ': mean_value_length', stat['mean_value_length'])
            stat['median_value_length'] = statistics.median(value_lengths)
            #print(': median_value_length', stat['median_value_length'])
            stat['min_value_length'] = min(value_lengths)
            stat['max_value_length'] = max(value_lengths)
            stat['std_value_length'] = statistics.stdev(value_lengths)
            stat['percentage_of_mode'] = (
                pd.Series(list).value_counts().max() / len(list))

        if field_general_type == 'q' and (colname != 'CalledPostCode' and colname != 'CallerLatitude'):
            #print(colname, "statistical_q")
            sample_mean = statistics.mean(v)
            sample_median = statistics.median(v)
            sample_var = np.var(v)
            sample_min = min(list)
            sample_max = max(list)
            sample_std = np.std(list)
            q1, q25, q75, q99 = np.percentile(list, [0.01, 0.25, 0.75, 0.99])
            iqr = q75 - q25

            stat['mean'] = sample_mean
            stat['normalized_mean'] = sample_mean / sample_max
            stat['median'] = sample_median
            stat['normalized_median'] = sample_median / sample_max

            stat['var'] = sample_var
            stat['std'] = sample_std
            stat['coeff_var'] = (
                sample_mean / sample_var) if sample_var else None
            stat['min'] = sample_min
            stat['max'] = sample_max
            stat['range'] = stat['max'] - stat['min']
            stat['normalized_range'] = (stat['max'] - stat['min']) / \
                sample_mean if sample_mean else None

            stat['entropy'] = entropy(list)
            #stat['gini'] = gini(list)
            stat['q25'] = q25
            stat['q75'] = q75
            # stat['med_abs_dev'] = statistics.median(abs( (float)list - sample_median))

            # stat['avg_abs_dev'] = statistics.mean(abs((float)list - sample_mean))
            stat['quant_coeff_disp'] = (q75 - q25) / (q75 + q25)
            stat['coeff_var'] = sample_var / sample_mean
            stat['skewness'] = skew(list)
            #print("skew1", stat['skewness'])
            #x = 3 * (sample_mean - sample_median)
            #stat['skew'] = x / stat['std']
            #print("skew2", stat['skew'])
            #stat['skewness'] = df.skew(axis=0)
            #print("skew2", stat['skewness'])
            stat['kurtosis'] = kurtosis(list)
            stat['moment_5'] = moment(list, moment=5)
            stat['moment_6'] = moment(list, moment=6)
            stat['moment_7'] = moment(list, moment=7)
            stat['moment_8'] = moment(list, moment=8)
            stat['moment_9'] = moment(list, moment=9)
            stat['moment_10'] = moment(list, moment=10)

        # Outliers
            outliers_15iqr = np.logical_or(
                v < (q25 - 1.5 * iqr), v > (q75 + 1.5 * iqr))
            outliers_3iqr = np.logical_or(
                v < (q25 - 3 * iqr), v > (q75 + 3 * iqr))
            outliers_1_99 = np.logical_or(v < q1, v > q99)
            outliers_3std = np.logical_or(
                v < (
                    sample_mean -
                    3 *
                    sample_std),
                v > (
                    sample_mean +
                    3 *
                    sample_std))
            stat['percent_outliers_15iqr'] = sum(outliers_15iqr) / len(v)
            stat['percent_outliers_3iqr'] = sum(outliers_3iqr) / len(v)
            stat['percent_outliers_1_99'] = sum(outliers_1_99) / len(v)
            stat['percent_outliers_3std'] = sum(outliers_3std) / len(v)

            stat['has_outliers_15iqr'] = any(outliers_15iqr)
            stat['has_outliers_3iqr'] = any(outliers_3iqr)
            stat['has_outliers_1_99'] = any(outliers_1_99)
            stat['has_outliers_3std'] = any(outliers_3std)

            # Statistical Distribution
            if len(v) >= 8:
                normality_k2, normality_p = normaltest(v)
            stat['normality_statistic'] = normality_k2
            stat['normality_p'] = normality_p
            stat['is_normal_5'] = (normality_p < 0.05)
            stat['is_normal_1'] = (normality_p < 0.01)
            #print(field_name, stat)
        # check_statistical_features=stat
        print(colname, stat)

    return check_statistical_features


if __name__ == '__main__':

    check_integrity("/home/nehaj/infinity/altair/data1_correct.xlsx")
