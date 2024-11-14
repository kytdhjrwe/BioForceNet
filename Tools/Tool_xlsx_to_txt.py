import os
import pandas as pd
#In the COMSOL finite element analysis software, the input of tibial model parameters is a txt file.
# Now, the stored muscle force and ground reaction force in Excel are converted into a specified txt format

def excel_to_txt(file_path, output_file_path):
    """
    Convert Excel files to TXT files
    """
    data = pd.read_excel(file_path).iloc[:,8:16]

    with open(output_file_path, 'w') as file:
        for column in data.columns:

            file.write(column + ' ')
            line = ' '.join(map(str, data[column].apply(lambda x: round(x, 1)).tolist()))
            file.write(line + ' [N]' + '\n')

    print(f"数据已成功写入: {output_file_path}")

def process_all_excel_files_in_directory(input_directory, output_directory):
    """
    Process all Excel files in the folder sequentially
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_name in os.listdir(input_directory):
        if file_name.endswith('.xlsx'):
            input_file_path = os.path.join(input_directory, file_name)
            output_file_name = file_name.replace('.xlsx', '.txt')
            output_file_path = os.path.join(output_directory, output_file_name)

            excel_to_txt(input_file_path, output_file_path)


if __name__ == '__main__':

    input_directory = 'Input folder path'
    output_directory = 'Output folder path'

    process_all_excel_files_in_directory(input_directory, output_directory)
