import os
from urllib import request

COUNTER_VALUES = list(range(1, 5))  # update this to account for other run IDs
VARIABLE_NAMES = ['tcc', 't2m', 'u10', 'v10', 'tp', 'tisr']  # update this if variable names should differ (probably not necessary)

OUTPUT_FOLDER = 'losses'
OUTPUT_FILE_PATTERN = 'losses_{variable}_{counter:04d}.csv'
URL_PATTERN = 'http://localhost:6006/data/plugin/scalars/scalars?tag=Loss%2Ftrain&run=val%2F{variable}%2F{counter}&format=csv'
#                                                                    ^^^^^^^^^^^^
#                                                                    ||||||||||||
# It might happen that you need to adapt this part to download data from other plots in the tensorboard surface
# In this case, just look up the correct path in the download link.


def prepare_output_directory():
    cwd = os.getcwd()
    output_directory = os.path.join(cwd, OUTPUT_FOLDER)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    print(f'[INFO] Downloading files to {output_directory}')
    return output_directory


def run_download(output_directory: str):
    for variable in VARIABLE_NAMES:
        for counter in COUNTER_VALUES:
            print(f'[INFO] Processing variable {variable} , run {counter}.')
            url = URL_PATTERN.format(variable=variable, counter=counter)
            output_file_name = OUTPUT_FILE_PATTERN.format(variable=variable, counter=counter)
            request.urlretrieve(url, os.path.join(output_directory, output_file_name))
    print('[INFO] Finished download')


def main():
    output_directory = prepare_output_directory()
    run_download(output_directory)


if __name__ == '__main__':
    main()
