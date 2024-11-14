#Find the maximum values of vGRF and TBF for normalizing RMSE
import os
import numpy as np
import pandas as pd

# Function to extract the min and max values of column 'vGRF' or 'TBF' from an Excel file


# Main function to process all Excel files in a folder
def process_folder(folder_path, column_name='Y'):
    all_min, all_max = float('inf'), float('-inf')
    file_min_max = {}

    # Loop through each file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path).iloc[:, 16]

            file_min, file_max = df.min(), df.max()

            if file_min is not None and file_max is not None:
                file_min_max[file_name] = (file_min, file_max)
                all_min = min(all_min, file_min)
                all_max = max(all_max, file_max)

    return file_min_max, all_min, all_max


# Example usage
folder_path = 'Traindata'  # 请将此处替换为你存放Excel文件的实际文件夹路径
file_min_max, overall_min, overall_max = process_folder(folder_path)

average_Max = []
print("The minimum and maximum values in each file：")
for file, (min_val, max_val) in file_min_max.items():
    print(f"file: {file}, minimum value: {min_val}, maximum value: {max_val}")
    average_Max.append(max_val)

average_Max_value = np.array(average_Max).mean()
print(f"The average of the maximum values in each file: {average_Max_value}")

print(f"The minimum value among all files: {overall_min}, maximum value: {overall_max}")

