# AnomalyDetection
This repository is used for anomaly detection software created by Mindaugas Bereiša on Python. For farther use You have to get my permission.
It can be used in two ways:

# Please use Python 3.7 version just to be sure that it is the same that is this project done.

# On Pycharm: 
to run and look for anomalies in large MS Access or CSV files of DATA SERIES with DateStamp.
The software automaticly detects anomalies and saves reports in the folder Reports and detailes by date and types of charts.
You can choose by activating "y"/"n" function whether You want to save as CSV file cute by smaller parts.
You can chose many parameters.

# Default values

The number of speeds mees what is the step of data reading
speed = 1

convert file to CSV ('y'/'n')
convert_to_csv = 'n'

The limit of number of cuts of the data it is:
time_of_iteration_limit_to_CSV = 10000

interval to cut by steps. 1- 60 min
min = 20

Coeficient of 1.5 - 2.0 sensitivity
alarm_coef = 1.5

Print charts all in one an 3d? ('y'/'n')
print all = 'n'

Show charts to desktop in real time (it gets much mor slow but very nice charts)? ('y'/'n')
show_chart = "n"

Do you want to save the charts?
save_chart = 'n'

Do You want to close automatically the browser after showing the chart?
close_browser = 'n'

Time and frequency parameters of data in file 100 it means 100 records per second
times_per_sek= 100

Roling average parameter
average = 100

How many seconds You would like to sleep system in time You are looking to the life charts before automatically it will close it.
close_seconds = 20

If Your file is more than 100 MB I recommend You to cut it in intervals by changing the time and recording to CSV parameter. 
You can use for it the file "File_cut_in_parts_t_CSV.py"
Choose the paramters and run and make new CSV files in intervals that You need.

Then You can take off the big filel from the folder "data" and put these interval files instead and run again with file "Anomaly_detection_2021_ver_1.py"

Before file "Anomaly_detection_2021_ver_1.py" run be shore that convert_to_csv = 'n', because it makes slow and takes a lot of storage.

## If the system finds anomaly automatically makes the file of CSV of the interval. Usually I use 3-5 minutes intervals to run in a files of several hours. 
## This 3-5 minutes files with data_zero.CSV you can put in to the folder "Deep_reports" and run Jupyter Notepooks again. 

Best regards

Mindaugas Bereiša

