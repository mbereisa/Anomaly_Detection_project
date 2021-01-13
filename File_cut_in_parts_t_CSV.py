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

def step_from_DF(df_sensors_data, lenghth_df_step, time_of_iteration):
    print("numpyArray_from_DF(df_sensors_data, lenghth_of_Array, time_of_iteration)")
    print("Time_of_iteration", time_of_iteration)
    if time_of_iteration == 0:
        start_interval = 0
        end_interval = lenghth_df_step + lenghth_df_step * 3
    else:
        start_interval = (time_of_iteration + 1) * lenghth_df_step
        end_interval = (time_of_iteration + 1) * lenghth_df_step + lenghth_df_step * 3
    part_of_sensors_data = df_sensors_data[start_interval:end_interval].copy()
    print(part_of_sensors_data.head())
    if convert_to_csv == 'y':
        # Converting to CSV file if needed
        if Convert_todates == "y":
            df_pint = indexing_df(part_of_sensors_data)
        else:
            df_pint = part_of_sensors_data
        if not os.path.exists(f".\\Produce_CSV\\"):
            os.makedirs(f".\\Produce_CSV\\")
        # ===========================================================================
        pd.DataFrame(part_of_sensors_data).to_csv(f'Produce_CSV\\data_zero{time_of_iteration}.csv', index=False)
        if time_of_iteration > time_of_iteration_limit_to_CSV:
            quit()
        # ===========================================================================


def run_by_steps(df_sensors_data, lenghth_df_step):
    print("run_by_steps(df_sensors_data, lenghth_of_Array ")
    df_sensors_data = taking_of_nan_values_DF(df_sensors_data)
    dflength = len(df_sensors_data)
    times_for = int(dflength / lenghth_df_step) - 2
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
global min, sek, times_per_sek, average, newPath1, show_chart, default, close_browser, close_seconds, speed_default, save_chart
global min_default, print_all, alarm_coef, df_columns_lables, convert_to_csv, time_of_iteration_limit_to_CSV, speed


# conver to to real dates or keep it as TimStamp as it is?
Convert_todates = "n"

# Default_values
speed = 1

# convert file to CSV ('y'/'n')
convert_to_csv = 'y'

# The limit of number of cuts
time_of_iteration_limit_to_CSV = 1

# interval to cut by steps. 1- 60 min
min = 3

# time and freqency parameters
times_per_sek= 100


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

for path in direktory:
    if newPath1[-3:] == "csv":
        start = datetime.datetime.now()
        print(start)
        print(
            '-------------------------------------------------------------------------------------------------------------')
        print("Starting reading CSV ")
        # ---------------------------------------------------
        csv_reader(newPath1, lenghth_df_step)
        print(f"{datetime.datetime.now()}, Finished CSV {datetime.datetime.now() - start}")
