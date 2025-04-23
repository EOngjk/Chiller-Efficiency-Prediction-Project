import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
import requests
import fastparquet as fp

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
lat = 22.4827305710136
lon = 114.11533654233023
API_key = "3a10de7122a9154d1694fcd1e5e00fb4"



@ st.cache_data
def load_chillers_data():
    df_chiller_1 = pd.read_parquet("Chiller_1_data.parquet", engine='fastparquet')
    df_chiller_2 = pd.read_parquet("Chiller_2_data.parquet", engine='fastparquet')
    df_chiller_3 = pd.read_parquet("Chiller_3_data.parquet", engine='fastparquet')
    
    return df_chiller_1, df_chiller_2, df_chiller_3

def create_features_coolingload(df):
    """
    Create time series features based on time series index.
    """
    df['hour'] = df['record_timestamp'].dt.hour
    df['dayofweek'] = df['record_timestamp'].dt.day_of_week
    df['quarter'] = df['record_timestamp'].dt.quarter
    df['month'] = df['record_timestamp'].dt.month
    df['dayofyear'] = df['record_timestamp'].dt.day_of_year
    df['Total Cooling Output (kW)'] = (df['Chiller_1_Status']*df['Current Cooling Output (kW)_1'])+ (df['Chiller_2_Status']*df['Current Cooling Output (kW)_2']) + (df['Chiller_3_Status']*df['Current Cooling Output (kW)'])
    df['Total Power Supply (kW)'] = (df['Chiller_1_Status']*df['Power Supply (kW)_1'])+ (df['Chiller_2_Status']*df['Power Supply (kW)_2']) + (df['Chiller_3_Status']*df['Power Supply (kW)'])
    return df

@st.cache_data
def load_merged_chillers_data():
    df_chiller_1, df_chiller_2, df_chiller_3 = load_chillers_data()
    # merge dataframes together to get the sum of cooling output needed
    df_joined = pd.merge(df_chiller_1, df_chiller_2, on = 'record_timestamp', how='outer', suffixes=('_1', '_2'))
    df_joined = pd.merge(df_joined, df_chiller_3, on='record_timestamp', how='outer', suffixes=(None, "_3"))

    # keep relevant columns
    df_joined = df_joined[['record_timestamp', 'Current Cooling Output (kW)_1', 'Current Cooling Output (kW)_2', "Current Cooling Output (kW)", 
                        'Power Supply (kW)_1', 'Power Supply (kW)_2', 'Power Supply (kW)',
                        'Air Temperature (Celsius)','Relative Humidity (%)', 'Chiller_1_Status', 'Chiller_2_Status', 'Chiller_3_Status']]

    # Handle NaN value for Current Cooling Output (kW)_1, Current Cooling Output (kW)_2, Current Cooling Output (kW) and Chiller_1_Status, Chiller_2_Status, Chiller_3_Status
    df_joined[['Chiller_1_Status', 'Current Cooling Output (kW)_1']] = df_joined[['Chiller_1_Status', 'Current Cooling Output (kW)_1']].fillna(0)
    df_joined[['Chiller_2_Status', 'Current Cooling Output (kW)_2']] = df_joined[['Chiller_2_Status', 'Current Cooling Output (kW)_2']].fillna(0)
    df_joined[['Chiller_3_Status', 'Current Cooling Output (kW)']] = df_joined[['Chiller_3_Status', 'Current Cooling Output (kW)']].fillna(0)

    # Create time features and Total Cooling Output
    df_joined = create_features_coolingload(df_joined)

    # drop other unnecessary columns
    df_joined = df_joined.drop(columns=['Current Cooling Output (kW)_1', 'Current Cooling Output (kW)_2', 'Current Cooling Output (kW)',
                                        'Power Supply (kW)_1', 'Power Supply (kW)_2', 'Power Supply (kW)',
                                        'Chiller_1_Status', 'Chiller_2_Status', 'Chiller_3_Status'])

    return df_joined


def load_real_time_data():
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={API_key}"
    response = requests.get(url)
    response_json = response.json()

    real_time_epoch = response_json['dt']
    real_time_time = unix_to_datetime(real_time_epoch)
    data = response_json['main']
    real_time_temperature = data['temp']
    real_time_humidity = data['humidity']
    
    return real_time_time, real_time_temperature, real_time_humidity

# Returns the dataframe that shows the perdicted cooling output needed
@st.cache_data
def load_predicted_cooling_output():
    df_predicted_cooling_output = pd.read_parquet("Predicted_Total_Cooling_Output.parquet", engine='fastparquet')
    return df_predicted_cooling_output

# Returns the dataframe with the optimized chiller sequence with the power used and COP breakdown
@st.cache_data
def load_simulated_optimized_chillers():
    df_simulated_optimized_chillers = pd.read_parquet("Simulated_Optimized_Chiller.parquet", engine='fastparquet')
    df_simulated_optimized_chillers['record_timestamp'] = pd.to_datetime(df_simulated_optimized_chillers['record_timestamp'])
    return df_simulated_optimized_chillers

@st.cache_data
def load_past_temp_humid_data():
    df_temp_humid = pd.read_parquet("Temperature_Humidity.parquet", engine='fastparquet')
    df_temp_humid['record_timestamp'] = pd.to_datetime(df_temp_humid['record_timestamp'])
    df_temp_humid.dropna(inplace=True)
    return df_temp_humid

def unix_to_datetime(unix_input):
    return datetime.datetime.fromtimestamp(int(unix_input)) #.strftime('%Y-%m-%d %H:%M:%S')

# Data from 2025/02/01 to the day before today
def load_past_temp_humid_data_api(start_date):
    # change to UNIX
    start_date_epoch = start_date.timestamp()
    print("The input time", start_date)
    print("Unix for input time:", start_date_epoch)

    past_url = f"https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start_date_epoch}&units=metric&appid={API_key}"
    past_response = requests.get(past_url)
    past_response_json = past_response.json()
    # print(past_response_json)
    extracted_past_data = []
    for data in past_response_json['list']:
        dt = data['dt']
        dt = unix_to_datetime(dt)
        # Temperature Celclius
        temp = data['main']['temp']
        # Humidity
        humidity = data['main']['humidity']
        extracted_past_data.append({'record_timestamp': dt, 'Temperature': temp, 'Humidity': humidity})

    # Creating the DataFrame
    past_data_df = pd.DataFrame(extracted_past_data)
    past_data_df['record_timestamp'] = pd.to_datetime(past_data_df['record_timestamp'])
    return past_data_df


def load_4_days_forecast_data():
    forecast_url = f"https://pro.openweathermap.org/data/2.5/forecast/hourly?lat={lat}&lon={lon}&units=metric&appid={API_key}"
    forecast_response = requests.get(forecast_url)
    forecast_response_json = forecast_response.json()

    extracted_data = []
    for data in forecast_response_json['list']:
        dt_txt = data['dt_txt']
        # Temperature Celclius
        temp = data['main']['temp']
        # Humidity
        humidity = data['main']['humidity']
        extracted_data.append({'record_timestamp': dt_txt, 'Temperature': temp, 'Humidity': humidity})

    # Creating the DataFrame
    forecast_df = pd.DataFrame(extracted_data)
    forecast_df['record_timestamp'] = pd.to_datetime(forecast_df['record_timestamp'])
    # Display the DataFrame
    return forecast_df