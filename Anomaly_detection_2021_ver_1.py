import warnings
from datetime import datetime
from plotly.offline import iplot
import cufflinks as cf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pylab as pl
import pyodbc
from numpy import fft as npfft
from scipy import signal
from scipy.fftpack import fft
import datetime
import shutil

warnings.filterwarnings("ignore")
import time
import plotly.express as px
import io
import matplotlib as mpl
from matplotlib.pylab import rcParams


def indexing_df(x):
    print("indexing_df(df_sensors_data)")
    x_lables = x.columns
    x.index = pd.to_datetime(x[x_lables[0]], unit='ms')
    delite_column = x_lables[0]
    x = x.drop(delite_column, axis=1)
    x.sort_index(inplace=True)
    return (x)


def clean_up(save_text):
    save_text = str(save_text.replace("/", "\\"))
    save_text = str(save_text.replace("\\\\", "\\"))
    save_text = str(save_text.replace(":", "_"))
    save_text = str(save_text.replace(".mdb", ""))
    save_text = str(save_text.replace(",", "_"))
    save_text = str(save_text.replace(".", "_"))
    save_text = str(save_text.replace("-", "_"))
    save_text = str(save_text.replace(" ", "_"))
    return save_text


def directory_name(name):
    directory2 = '\\Reports\\' + newPath1 + '\\' + name
    directory2 = clean_up(directory2)

    newpath2 = directory2
    if not os.path.exists("." + newpath2):
        os.makedirs("." + newpath2)
    return newpath2


def save_path_pictures(name3):
    save_path = name3
    save_path = clean_up(save_path)
    return save_path + ".jpg"


def taking_of_nan_values_DF(df):
    # print("taking_of_nan_values_DF(df)")
    # interpolation
    df = df.interpolate(method='linear', limit_direction='forward', axis=0)
    # taking of nullmi
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df


def iplot_charts(npx, name2, directory2):
    print("iplot_charts(npx, time_of_iteration, name2)")
    name3 = directory2 + '\\' + name2 + "_iplot_charts"
    df = pd.DataFrame(npx)
    fig = (df).iplot(asFigure=True, xTitle="Time",
                     yTitle="Value", title=name3)
    fig.update_layout(autosize=True)
    if save_chart == 'y':
        save_file = "." + save_path_pictures(name3)
        fig.write_image(save_file, width=1000, height=600, scale=3)

    if show_chart == 'y':
        fig.show()
        time.sleep(close_seconds)
        if close_browser == 'y':
            os.system("taskkill /im chrome.exe /f")


def iplot_charts_3d(npx, name2, directory2):
    print("iplot_charts_3d(npx, time_of_iteration, name2)")
    name3 = directory2 + '\\' + name2 + "_iplot_charts_3d"
    # npx = np.delete(npx, 0, axis=1)
    z_data = npx
    fig = go.Figure(data=[go.Surface(z=z_data.values, x=z_data.columns)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))

    fig.update_layout(title=name3, autosize=True)
    fig.update_layout(scene=dict(
        xaxis_title='X-Sensors',
        yaxis_title='Y-time',
        zaxis_title='Z-Value'))
    if save_chart == 'y':
        save_file = "." + save_path_pictures(name3)
        fig.write_image(save_file, width=1000, height=600, scale=3)
    if show_chart == 'y':
        fig.show()
        time.sleep(close_seconds)
        if close_browser == 'y':
            os.system("taskkill /im chrome.exe /f")

def anomaly_plot(df_plot, name3, column):
    df_plot = taking_of_nan_values_DF(df_plot)

    fig = (df_plot).iplot(asFigure=True, xTitle="Time", yTitle="Value", title=name3 + column)
    fig.update_layout(autosize=True)

    if save_chart == 'y':
        save_file = "." + save_path_pictures(name3 + "_column_" + str(column))
        fig.write_image(save_file, width=1000, height=600, scale=3)

    if show_chart == 'y':
        fig.show()
        time.sleep(close_seconds)
        if close_browser == 'y':
            os.system("taskkill /im chrome.exe /f")


def leaknavigator(part_of_sensors_data_zero, part_of_sensors_data, name2, name, file_main_name, main_directory):

    if print_all == 'y':
        print_df = part_of_sensors_data_zero[::speed]
        iplot_charts_3d(indexing_df(print_df), name, f"{main_directory}\\3d")
        iplot_charts(indexing_df(print_df), file_main_name, "")
    df = part_of_sensors_data_zero[::speed].copy()
    df = indexing_df(df)
    alarm_coef = 0.8
    rolling1 = 100
    print(df.head())
    columns = df.columns
    rows = len(df)
    training_rows = int(rows * 0.5)
    anomaly_count = 0
    for column in columns:
        df[column + '_average'] = df[column].rolling(window=rolling1).mean().shift(periods=-rolling1)
        noise = (df[column].iloc[:training_rows].max() - df[column].iloc[:training_rows].min()) / 2
        df[column + '_average_min_rolling'] = df[column + '_average'] - noise * alarm_coef
        df[column + '_average_max_rolling'] = df[column + '_average'] + noise * alarm_coef
        df[column + '_average_min'] = df[column + '_average_min_rolling'] .iloc[:training_rows].min()
        df[column + '_average_max'] = df[column + '_average_max_rolling'].max()

        taking_of_nan_values_DF(df)

    for row in range(training_rows, rows):
        for column in columns:
            if df[column].iloc[row] > df[column + '_average_max_rolling'].iloc[row]:
                anomaly_count = anomaly_count + 1
                print("anomaly_count", anomaly_count, column, " value > average_max_rolling  ")

            if df[column].iloc[row] < df[column + '_average_min_rolling'].iloc[row]:
                anomaly_count = anomaly_count + 1
                print("anomaly_count", anomaly_count, column, " value < average_min_rolling ")

            if df[column].iloc[row] > df[column + '_average_max'].iloc[row]:
                anomaly_count = anomaly_count + 1
                print("anomaly_count", anomaly_count, column, " value > average_max  ")

            if df[column].iloc[row] < df[column + '_average_min'].iloc[row]:
                anomaly_count = anomaly_count + 1
                print("anomaly_count", anomaly_count, column, " value < average_min ")


            if anomaly_count > 20:
                print("===========================================================================================")
                print(f'!!!!!!!!!!!!!!!!!Anomaly detected in data column !!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                print("indexing for printing")
                print(part_of_sensors_data_zero)
                directory2 = directory_name(name)
                pd.DataFrame(part_of_sensors_data_zero).to_csv(
                    '.' + directory2 + '\\data_zero.csv',
                    index=False)

                dfx = indexing_df(part_of_sensors_data_zero[::speed])
                # anomaly_chart(dfx, name2, directory2)
                for column in columns:
                    anomaly_plot(df[[column,
                                     column + '_average',
                                     column + '_average_min',
                                     column + '_average_max',
                                     column + '_average_min_rolling',
                                     column + '_average_max_rolling']],
                                 directory2 + '\\' + name2 + "_anomaly_chart",
                                 column)

                print('Saving CSV files for deeper report ')
                print("-----------------------------------------------------------------------")
                files = glob("Deep_reports\\*")
                print(files)
                # for f in files:
                #     print("Copy : ", f, " to ", f'.{directory2}')
                #     shutil.copy(f, f'.{directory2}')
                return print('Anomaly detected')
    return print('Anomaly is not detected')

def starting_all_from_zero(part_of_sensors_data, name, time_of_iteration, main_directory, file_main_name):
    print("starting_all_from_zero(df_sensors_data, time_of_iteration)")
    name2 = name + "_starting_all_from_zero_" + str(time_of_iteration)
    columns = part_of_sensors_data.columns
    part_of_sensors_data_zero = part_of_sensors_data
    for column in columns:
        if column != columns[0]:
            part_of_sensors_data_zero[column] = part_of_sensors_data_zero[column] - \
                                                part_of_sensors_data_zero[column].iloc[0]
    leaknavigator(part_of_sensors_data_zero, part_of_sensors_data, name2, name, file_main_name, main_directory)

def identify_timeperiod_creat_directory(part_of_sensors_data, time_of_iteration):
    print("identify_timeperiod_creat_directory(part_of_sensors_data, time_of_iteration")
    first_column = part_of_sensors_data.columns[0]
    Start_stamp = int(part_of_sensors_data[first_column].iloc[0])
    End_stamp = int(part_of_sensors_data[first_column].iloc[-1])
    starting = str(pd.to_datetime(Start_stamp, unit='ms'))
    stoping = str(pd.to_datetime(End_stamp, unit='ms'))
    time_period = f"From_{starting[:16]}_to__{stoping[:16]}"
    name = f"{time_period}"
    main_directory = f"\\Reports\\{newPath1[:-4]}"
    file_main_name = f"{main_directory}\\{name}"
    print("main_directory: ", main_directory)
    print("file_main_name: ", file_main_name)
    if not os.path.exists(f".{main_directory}"):
        os.makedirs(f".{main_directory}")
    if not os.path.exists(f".{main_directory}\\3d"):
        os.makedirs(f".{main_directory}\\3d")
    starting_all_from_zero(part_of_sensors_data, name, time_of_iteration, main_directory, file_main_name)

def step_from_DF(df_sensors_data, lenghth_df_step, time_of_iteration):
    print("numpyArray_from_DF(df_sensors_data, lenghth_of_Array, time_of_iteration)")
    print("Time_of_iteration", time_of_iteration)
    if time_of_iteration == 0:
        start_interval = 0
        end_interval = lenghth_df_step + lenghth_df_step * 5
    else:
        start_interval = (time_of_iteration + 1) * lenghth_df_step
        end_interval = (time_of_iteration + 1) * lenghth_df_step + lenghth_df_step * 5
    part_of_sensors_data = df_sensors_data[start_interval:end_interval].copy()
    print(part_of_sensors_data.head())
    if convert_to_csv == 'y':
        # Converting to CSV file if needed
        # ===========================================================================
        if not os.path.exists(f".\\Produce_CSV\\"):
            os.makedirs(f".\\Produce_CSV\\")
        pd.DataFrame(part_of_sensors_data).to_csv(f'Produce_CSV\\data_zero{time_of_iteration}.csv', index=False)
        if time_of_iteration > time_of_iteration_limit_to_CSV:
            quit()
        # ===========================================================================
    identify_timeperiod_creat_directory(part_of_sensors_data, time_of_iteration)

def run_by_steps(df_sensors_data, lenghth_df_step):
    print("run_by_steps(df_sensors_data, lenghth_of_Array ")
    df_sensors_data = taking_of_nan_values_DF(df_sensors_data)
    dflength = len(df_sensors_data)
    times_for = int(dflength / lenghth_df_step) - 4
    print("Iteration times for this file will be needed: ", times_for)
    for time_of_iteration in range(times_for):
        print(time_of_iteration)
        print("===========================================================================")
        step_from_DF(df_sensors_data, lenghth_df_step, time_of_iteration)

def MS_Access_DB_reader(newPath1, lenghth_df_step):
    print("MS_Access_DB_reader(path) ")
    conn = pyodbc.connect(
        r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=' + newPath1 + ';')
    df_sensors_data = pd.read_sql_query('select * from duomenys', conn)
    print(df_sensors_data.head())
    run_by_steps(df_sensors_data, lenghth_df_step)

def csv_reader(newPath1, lenghth_df_step):
    print("CSV_reader(newPath1, lenghth_df_step)")
    df_sensors_data = pd.read_csv(f'.\\{newPath1}')
    print(df_sensors_data.head())
    run_by_steps(df_sensors_data, lenghth_df_step)

# Conditions for starting
global min, sek, times_per_sek,  newPath1, show_chart, default, close_browser, close_seconds, speed_default, save_chart
global min_default, print_all, df_columns_lables, convert_to_csv, time_of_iteration_limit_to_CSV, speed

# Default_values
speed = 10
# convert file to CSV ('y'/'n')
convert_to_csv = 'n'
time_of_iteration_limit_to_CSV = 100000
# interval to cut by steps. 1- 60 min
min = 5

# Print charts all in one an 3d? ('y'/'n')
print_all = 'y'
show_chart = 'n'
save_chart = 'y'
close_browser = 'n'
# time and freqency parameters
times_per_sek= 100
close_seconds = 30

print(
    '------------------------------------------------------------------------------------------------------------------------------------------')

print("Convert_to_csv = :", convert_to_csv)
print("print_all :", print_all)
print("Show charts of ano,aly  = ", show_chart)
print("Close_browser = ", close_browser)
print("Speed = ", speed)
print("times_per_sek = ", times_per_sek)
print("min = ", min)
print(
    '------------------------------------------------------------------------------------------------------------------------------------------')

# Reading file and starting calculation
print("Staring reading and calculate data. Pleas vate some minutes to get the result")
sek = 60
lenghth_df_step = times_per_sek * sek * min
from glob import glob
import os

start = datetime.datetime.now()
direktory = glob("data\\*")
for name_of_file in direktory:
    mame = name_of_file
    print(name_of_file)

for path in direktory:
    newPath1 = str(path.replace("/", "\\"))
    print(path)
    print(
        '-------------------------------------------------------------------------------------------------------------')
    print(f"Starting from the first file: {newPath1} ")

    if newPath1[-3:] == "mdb":
        start = datetime.datetime.now()
        print(start)
        print(
            '-------------------------------------------------------------------------------------------------------------')
        print("Starting reading MS ACCESS ")
        # ---------------------------------------------------
        MS_Access_DB_reader(newPath1, lenghth_df_step)
        print(f"{datetime.datetime.now()}, Finished MS ACCESS {datetime.datetime.now() - start}")

    if newPath1[-3:] == "csv":
        start = datetime.datetime.now()
        print(start)
        print(
            '-------------------------------------------------------------------------------------------------------------')
        print("Starting reading CSV ")
        # ---------------------------------------------------
        csv_reader(newPath1, lenghth_df_step)
        print(f"{datetime.datetime.now()}, Finished CSV {datetime.datetime.now() - start}")
