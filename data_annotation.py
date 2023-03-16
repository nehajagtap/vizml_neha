import pandas as pd

# Read Excel file
file_path = "/home/nehaj/infinity/altair/data/rapeandchildsexualabusedataPJ.xlsx"
dfs = pd.read_excel(file_path, sheet_name=None)

# Get the first sheet (original data)
original_df = dfs[list(dfs.keys())[0]]

# Create a 2D and 3D matrix with number of columns in the original sheet as the dimensions
num_columns = len(original_df.columns)
two_d_matrix = [[0 for j in range(num_columns)] for i in range(num_columns)]
three_d_matrix = [[[0 for k in range(num_columns)] for j in range(
    num_columns)] for i in range(num_columns)]

# Iterate over all sheets except the first one
for sheet_name, df in dfs.items():
    if sheet_name == list(dfs.keys())[0]:
        continue

    # Check if the sheet contains a pivot table
    if ('Row Labels' in df.columns) or ('Column Labels' in df.columns):
        # Extract the column names used in the pivot table
        row_labels_column = df.columns[df.columns.str.contains(
            'Row Labels')][0] if 'Row Labels' in df.columns else None
        column_labels_column = df.columns[df.columns.str.contains(
            'Column Labels')][0] if 'Column Labels' in df.columns else None
        values_column = [c for c in df.columns if c not in [
            row_labels_column, column_labels_column]][0]

        # Get the index of the columns in the original data
        row_labels_index = original_df.columns.get_loc(
            row_labels_column.split(" ")[-1]) if row_labels_column is not None else None
        column_labels_index = original_df.columns.get_loc(column_labels_column.split(
            " ")[-1]) if column_labels_column is not None else None
        values_index = original_df.columns.get_loc(values_column.split(
            " ")[-1]) if " " in values_column else original_df.columns.get_loc(values_column)

        # Update the 2D matrix
        if row_labels_index is not None and column_labels_index is not None:
            two_d_matrix[row_labels_index][column_labels_index] = 1
            two_d_matrix[column_labels_index][row_labels_index] = 1

        # Update the 3D matrix
        if row_labels_index is not None and column_labels_index is not None and values_index is not None:
            three_d_matrix[row_labels_index][column_labels_index][values_index] = 1
            three_d_matrix[column_labels_index][row_labels_index][values_index] = 1
            three_d_matrix[values_index][row_labels_index][column_labels_index] = 1


print(two_d_matrix)
print(three_d_matrix)
