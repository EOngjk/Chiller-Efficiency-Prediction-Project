import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
import statsmodels.api as sm
import warnings
import pickle
import plotly.graph_objs as go
from datetime import date, datetime, timedelta, time
warnings.filterwarnings('ignore')
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')
from Get_Data import load_chillers_data, load_merged_chillers_data, load_real_time_data, load_past_temp_humid_data, load_past_temp_humid_data_api, load_4_days_forecast_data

data_chiller_1, data_chiller_2, data_chiller_3 = load_chillers_data()

real_time, real_temp, real_humid = load_real_time_data()

new_order = ["record_timestamp","Air Temperature (Celsius)","Relative Humidity (%)","Predicted Cooling Load (kW)","Predicted Power Used (kW)",
             "Predicted Power Breakdown (kW)", "Predicted COP Breakdown", "Strategy","Main_Strategy","Details"]

# Define the data as a list of dictionaries
sample_data = [
    {"record_timestamp": "4/1/2024 0:00", "Temperature": 27, "Humidity": 83},
    {"record_timestamp": "4/1/2024 0:10", "Temperature": 27, "Humidity": 83},
    {"record_timestamp": "4/1/2024 0:20", "Temperature": 27, "Humidity": 83},
    {"record_timestamp": "7/19/2024 14:40", "Temperature": 29, "Humidity": 92},
    {"record_timestamp": "7/19/2024 14:50", "Temperature": 29, "Humidity": 91},
    {"record_timestamp": "8/6/2024 3:40", "Temperature": 29, "Humidity": 87},
    {"record_timestamp": "8/6/2024 3:50", "Temperature": 28, "Humidity": 87},
    {"record_timestamp": "8/6/2024 4:00", "Temperature": 28, "Humidity": 88},
    {"record_timestamp": "8/6/2024 4:10", "Temperature": 28, "Humidity": 88},
    {"record_timestamp": "1/31/2025 23:10", "Temperature": 18, "Humidity": 83},
    {"record_timestamp": "1/31/2025 23:20", "Temperature": 18, "Humidity": 84},
    {"record_timestamp": "1/31/2025 23:30", "Temperature": 18, "Humidity": 84},
    {"record_timestamp": "1/31/2025 23:40", "Temperature": 18, "Humidity": 84},
    {"record_timestamp": "1/31/2025 23:50", "Temperature": 18, "Humidity": 85},
]

# Create the DataFrame
sample_df = pd.DataFrame(sample_data)
sample_df['record_timestamp'] = pd.to_datetime(sample_df['record_timestamp'], format='%m/%d/%Y %H:%M')

def create_features_coolingload(df):
    """
    Create time series features based on time series index.
    """
    df['hour'] = df['record_timestamp'].dt.hour
    df['dayofweek'] = df['record_timestamp'].dt.day_of_week
    df['quarter'] = df['record_timestamp'].dt.quarter
    df['month'] = df['record_timestamp'].dt.month
    df['dayofyear'] = df['record_timestamp'].dt.day_of_year
    return df

def get_strategy_summary(df):
    strategy_stats = df.groupby(['Main_Strategy', 'Details']).agg(count=('Main_Strategy', 'size'),avg_power=('Predicted Power Used (kW)', 'mean')).reset_index()
    
    # Round up to 2 dp
    strategy_stats['avg_power'] = strategy_stats['avg_power'].round(2)

    most_popular = strategy_stats.loc[strategy_stats['count'].idxmax()]
    least_popular = strategy_stats.loc[strategy_stats['count'].idxmin()]
    most_cost_saving = strategy_stats.loc[strategy_stats['avg_power'].idxmin()]
    least_cost_saving = strategy_stats.loc[strategy_stats['avg_power'].idxmax()]

    return strategy_stats, most_popular, least_popular, most_cost_saving, least_cost_saving

# merge dataframes together to get the sum of cooling output needed
df_joined = load_merged_chillers_data()

# keep the datarows that are turned on only
df_joined_model_cooling = df_joined.copy()

# drop na
df_joined_model_cooling = df_joined_model_cooling.dropna()

# Filter total cooling output > 0
df_joined_model_cooling = df_joined_model_cooling[df_joined_model_cooling['Total Cooling Output (kW)'] > 0]

# print(df_joined_model_cooling.info())

# Remove Outlier
Q1 = df_joined_model_cooling['Total Cooling Output (kW)'].quantile(0.25)
Q3 = df_joined_model_cooling['Total Cooling Output (kW)'].quantile(0.75)
IQR = Q3 - Q1

# print("Quartile 1:", Q1)
# print("Quartile 3:", Q3)
# print("InterQuartile Range", IQR)

print("Number of rows BEFORE removing outliers:", len(df_joined_model_cooling))
df_joined_model_cooling = df_joined_model_cooling[~((df_joined_model_cooling['Total Cooling Output (kW)'] < (Q1 - 1.5 * IQR)) | (df_joined_model_cooling['Total Cooling Output (kW)'] > (Q3 + 1.5 * IQR)))]

print("Number of rows AFTER removing outliers:", len(df_joined_model_cooling))

# Modelling
# Feature is Total Cooling Output (kW)
seed = 42
test_size = 0.3
cooling_features = ['Air Temperature (Celsius)','Relative Humidity (%)', 'hour', 'dayofweek', 'quarter', 'month', 'dayofyear']
cooling_target = 'Total Cooling Output (kW)'

# Get the Cooling Load model
with open("cooling_load_model.pkl", 'rb') as file:  
    cooling_load_model = pickle.load(file)

def get_cooling_load_model():
    """
    This is to get the cooling load model
    """
    return cooling_load_model


# Get COP model for the Chillers
# Feature is COP_Own
seed = 42
test_size = 0.3
cop_features = ['Air Temperature (Celsius)','Relative Humidity (%)', 'Percentage of Loading (%)', 'hour', 'dayofweek', 'quarter', 'month', 'dayofyear']
cop_target = 'COP_Own'

def create_features_cop(df):
    """
    Create time series features based on time series index.
    """
    df['hour'] = df['record_timestamp'].dt.hour
    df['dayofweek'] = df['record_timestamp'].dt.day_of_week
    df['quarter'] = df['record_timestamp'].dt.quarter
    df['month'] = df['record_timestamp'].dt.month
    df['dayofyear'] = df['record_timestamp'].dt.day_of_year
    return df

# Chiller 1 COP model
with open("cop_chiller_1_model.pkl", 'rb') as file:  
    cop_1_model = pickle.load(file)

# Chiller 2 COP model
with open("cop_chiller_2_model.pkl", 'rb') as file:  
    cop_2_model = pickle.load(file)

# Chiller 3 COP model
with open("cop_chiller_3_model.pkl", 'rb') as file:  
    cop_3_model = pickle.load(file)

def get_cop_model():
    """
    This is to get the cop models for each chiller
    """
    return cop_1_model, cop_2_model, cop_3_model


# Function to create prediction input
def create_pred_input(temp_object, humidity_object, cooling_load, hour, day_of_week_number, quarter, month, day_of_year):
    input_data_cop = pd.DataFrame({
    'Air Temperature (Celsius)': [temp_object],
    'Relative Humidity (%)': [humidity_object],
    'Percentage of Loading (%)': [cooling_load/150*100],
    'hour': [hour],
    'dayofweek': [day_of_week_number],
    'quarter': [quarter],
    'month': [month],
    'dayofyear': [day_of_year]})

    return input_data_cop

# Functions for the optimization

# Calcualte the Power Used
def calculate_power(cop, cooling_load):
    return round(cooling_load/cop, 2)

# One Chiller usage
def one_chiller_cop(cooling_load, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_model_1, cop_model_2, cop_model_3, combination_cop_results, combination_power_results, combination_tag_results):
    
    chillers = ["Chiller 1", "Chiller 2", "Chiller 3"]
    
    # Operational limit: cooling load should not be > 150 for a single chiller combination
    if cooling_load > 150:
        return  # No valid configuration for one chiller if load is above 150

    for chiller in chillers:   
        input_data_cop = create_pred_input(temperature_input, humidity_input, cooling_load, hour, day_of_week_number, quarter, month, day_of_year)

        # make prediction
        if "1" in chiller:
            cop = cop_model_1.predict(input_data_cop)
        if "2" in chiller:
            cop = cop_model_2.predict(input_data_cop)
        else:
            cop = cop_model_3.predict(input_data_cop)
        
        # update the COP combination results dictionary
        combination_cop_results[chiller] = round(cop[0], 2) 

        # Update the combination Power results dictionary
        combination_power_results[chiller] = calculate_power(cop[0], cooling_load)

        # Update the tag 
        combination_tag_results[chiller] = f"One Chiller: {chiller}"

# Two Chiller usage
def two_chillers_cop(cooling_load, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_model_1, cop_model_2, cop_model_3, combination_cop_results, combination_power_results, combination_tag_results):
    
    # operational limit for two chillers
    cooling_load_div_2 = cooling_load/2 # Equal Load Sharing

    chillers = ["Chiller 1 & 2", "Chiller 1 & 3", "Chiller 2 & 3"]

    for chiller in chillers:
        # Chiller 1 and 2
        if chiller == "Chiller 1 & 2":
            # Predict
            cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, cooling_load_div_2, hour, day_of_week_number, quarter, month, day_of_year))
            cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, cooling_load_div_2, hour, day_of_week_number, quarter, month, day_of_year))

            # Check if the cooling load is above the minimum requirement
            if cooling_load_div_2 / 150 >= 0.8:
                combination_cop_results[f"{chiller} Optimal Equal Load Percentage: {cooling_load_div_2/150*100:.2f}%"] = [round(cop_1[0],2), round(cop_2[0],2)]
                combination_power_results[f"{chiller} Optimal Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = [calculate_power(cop_1[0], cooling_load_div_2), calculate_power(cop_2[0], cooling_load_div_2)]
                combination_tag_results[f"{chiller} Optimal Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = f"Two Chillers Optimal Equal Load: {chiller}"

            else:                
                combination_cop_results[f"{chiller} Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = [round(cop_1[0],2), round(cop_2[0],2)]
                combination_power_results[f"{chiller} Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = [calculate_power(cop_1[0], cooling_load_div_2), calculate_power(cop_2[0], cooling_load_div_2)] 
                combination_tag_results[f"{chiller} Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = f"Two Chillers Equal Load: {chiller}"
            
        # Chiller 1 and 3
        elif chiller == "Chiller 1 & 3":
            # Predict
            cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, cooling_load_div_2, hour, day_of_week_number, quarter, month, day_of_year))
            cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, cooling_load_div_2, hour, day_of_week_number, quarter, month, day_of_year))
        
            # Check if the cooling load is above the minimum requirement
            if cooling_load_div_2 / 150 >= 0.8:
                combination_cop_results[f"{chiller} Optimal Equal Load Percentage: {cooling_load_div_2/150*100:.2f}%"] = [round(cop_1[0],2), round(cop_3[0],2)]
                combination_power_results[f"{chiller} Optimal Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = [calculate_power(cop_1[0], cooling_load_div_2), calculate_power(cop_3[0], cooling_load_div_2)]
                combination_tag_results[f"{chiller} Optimal Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = f"Two Chillers Optimal Equal Load: {chiller}"
            else:
                combination_cop_results[f"{chiller} Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = [round(cop_1[0],2), round(cop_3[0],2)]
                combination_power_results[f"{chiller} Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = [calculate_power(cop_1[0], cooling_load_div_2), calculate_power(cop_3[0], cooling_load_div_2)]
                combination_tag_results[f"{chiller} Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = f"Two Chillers Equal Load: {chiller}"

        # Chiller 2 and 3
        else:
            # Predict
            cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, cooling_load_div_2, hour, day_of_week_number, quarter, month, day_of_year))
            cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, cooling_load_div_2, hour, day_of_week_number, quarter, month, day_of_year))
            
            # Check if the cooling load is above the minimum requirement
            if cooling_load_div_2 / 150 >= 0.8:
                combination_cop_results[f"{chiller} Optimal Equal Load Percentage: {cooling_load_div_2/150*100:.2f}%"] = [round(cop_2[0],2), round(cop_3[0],2)]
                combination_power_results[f"{chiller} Optimal Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = [calculate_power(cop_2[0], cooling_load_div_2), calculate_power(cop_3[0], cooling_load_div_2)]
                combination_tag_results[f"{chiller} Optimal Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = f"Two Chillers Optimal Equal Load: {chiller}"
            else:
                combination_cop_results[f"{chiller} Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = [round(cop_2[0],2), round(cop_3[0],2)]
                combination_power_results[f"{chiller} Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = [calculate_power(cop_2[0], cooling_load_div_2), calculate_power(cop_3[0], cooling_load_div_2)]
                combination_tag_results[f"{chiller} Equal Load Percentage: {(cooling_load_div_2/150*100):.2f}%"] = f"Two Chillers Equal Load: {chiller}"
   
    if cooling_load >= 150:
        for chiller in chillers:
            # Idea: Full load and Part load combination
            part_load = (cooling_load - 150) # Calculate part load 
            part_load_perc = part_load / 150 * 100
            
            if chiller == "Chiller 1 & 2":
                # Chiller 1 full load & Chiller 2 part load
                cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
                cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, part_load, hour, day_of_week_number, quarter, month, day_of_year))
                combination_cop_results[f"Chiller 1: 100%, Chiller 2: {part_load_perc:.2f}%"] = [round(cop_1[0],2) , round(cop_2[0],2)]
                combination_power_results[f"Chiller 1: 100%, Chiller 2: {part_load_perc:.2f}%"] = [calculate_power(cop_1[0], 150), calculate_power(cop_2[0], part_load)]
                combination_tag_results[f"Chiller 1: 100%, Chiller 2: {part_load_perc:.2f}%"] = f"Two Chillers One Full Load, One Part Load: Chiller 1 Full, Chiller 2 Part"

                # Chiller 1 part load & Chiller 2 full load
                cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, part_load, hour, day_of_week_number, quarter, month, day_of_year))
                cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
                combination_cop_results[f"Chiller 1: {part_load_perc:.2f}%, Chiller 2: 100%"] = [round(cop_1[0],2), round(cop_2[0],2)]
                combination_power_results[f"Chiller 1: {part_load_perc:.2f}%, Chiller 2: 100%"] = [calculate_power(cop_1[0], part_load), calculate_power(cop_2[0], 150)]
                combination_tag_results[f"Chiller 1: {part_load_perc:.2f}%, Chiller 2: 100%"] = f"Two Chillers One Full Load, One Part Load: Chiller 1 Part, Chiller 2 Full"

            elif chiller == "Chiller 1 & 3":
                # Chiller 1 full load & Chiller 3 part load
                cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
                cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, part_load, hour, day_of_week_number, quarter, month, day_of_year))
                combination_cop_results[f"Chiller 1: 100%, Chiller 3: {part_load_perc:.2f}%"] = [round(cop_1[0],2), round(cop_3[0],2)]
                combination_power_results[f"Chiller 1: 100%, Chiller 3: {part_load_perc:.2f}%"] = [calculate_power(cop_1[0], 150), calculate_power(cop_3[0], part_load)]
                combination_tag_results[f"Chiller 1: 100%, Chiller 3: {part_load_perc:.2f}%"] = f"Two Chillers One Full Load, One Part Load: Chiller 1 Full, Chiller 3 Part"

                # Chiller 1 part load & Chiller 3 full load
                cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, part_load, hour, day_of_week_number, quarter, month, day_of_year))
                cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
                combination_cop_results[f"Chiller 1: {part_load_perc:.2f}%, Chiller 3: 100%"] = [round(cop_1[0],2), round(cop_3[0],2)]
                combination_power_results[f"Chiller 1: {part_load_perc:.2f}%, Chiller 3: 100%"] = [calculate_power(cop_1[0], part_load), calculate_power(cop_3[0], 150)]
                combination_tag_results[f"Chiller 1: {part_load_perc:.2f}%, Chiller 3: 100%"] = f"Two Chillers One Full Load, One Part Load: Chiller 1 Part, Chiller 3 Full"
            
            else:
                # Chiller 2 full load & Chiller 3 part load
                cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
                cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, part_load, hour, day_of_week_number, quarter, month, day_of_year))
                combination_cop_results[f"Chiller 2: 100%, Chiller 3: {part_load_perc:.2f}%"] = [round(cop_2[0],2), round(cop_3[0],2)]
                combination_power_results[f"Chiller 2: 100%, Chiller 3: {part_load_perc:.2f}%"] = [calculate_power(cop_2[0], 150), calculate_power(cop_3[0], part_load)]
                combination_tag_results[f"Chiller 2: 100%, Chiller 3: {part_load_perc:.2f}%"] = f"Two Chillers One Full Load, One Part Load: Chiller 2 Full, Chiller 3 Part"
                
                # Chiller 2 part load & Chiller 3 full load
                cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, part_load, hour, day_of_week_number, quarter, month, day_of_year))
                cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
                combination_cop_results[f"Chiller 2: {part_load_perc:.2f}%, Chiller 3: 100%"] = [round(cop_2[0],2), round(cop_3[0],2)]
                combination_power_results[f"Chiller 2: {part_load_perc:.2f}%, Chiller 3: 100%"] = [calculate_power(cop_2[0], part_load), calculate_power(cop_3[0], 150)]
                combination_tag_results[f"Chiller 2: {part_load_perc:.2f}%, Chiller 3: 100%"] = f"Two Chillers One Full Load, One Part Load: Chiller 2 Part, Chiller 3 Full"

# Three Chiller usage
def three_chillers_cop(cooling_load, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_model_1, cop_model_2, cop_model_3, combination_cop_results, combination_power_results, combination_tag_results):
    # Divide the cooling load equally
    cooling_load_div_3 = cooling_load/3

    # Equal Load Prediction
    cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, cooling_load_div_3, hour, day_of_week_number, quarter, month, day_of_year))
    cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, cooling_load_div_3, hour, day_of_week_number, quarter, month, day_of_year))
    cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, cooling_load_div_3, hour, day_of_week_number, quarter, month, day_of_year))

    if cooling_load_div_3 / 150 >= 0.8:
        combination_cop_results["Chiller 1, 2 & 3 Optimal Equal Load Percentage: {:.2f}%".format(cooling_load_div_3/150*100)] = [round(cop_1[0], 2), round(cop_2[0], 2), round(cop_3[0],2)]
        combination_power_results["Chiller 1, 2 & 3 Optimal Equal Load Percentage: {:.2f}%".format(cooling_load_div_3/150*100)] = [calculate_power(cop_1[0], cooling_load_div_3), calculate_power(cop_2[0], cooling_load_div_3), calculate_power(cop_3[0], cooling_load_div_3)]
        combination_tag_results["Chiller 1, 2 & 3 Optimal Equal Load Percentage: {:.2f}%".format(cooling_load_div_3/150*100)] = f"Three Chillers Optimal Equal Load: Chiller 1, 2 & 3"
    else:
        combination_cop_results["Chiller 1, 2 & 3 Equal Load Percentage: {:.2f}%".format(cooling_load_div_3/150*100)] = [round(cop_1[0],2), round(cop_2[0],2), round(cop_3[0],2)]
        combination_power_results["Chiller 1, 2 & 3 Equal Load Percentage: {:.2f}%".format(cooling_load_div_3/150*100)] = [calculate_power(cop_1[0], cooling_load_div_3), calculate_power(cop_2[0], cooling_load_div_3), calculate_power(cop_3[0], cooling_load_div_3)]
        combination_tag_results["Chiller 1, 2 & 3 Equal Load Percentage: {:.2f}%".format(cooling_load_div_3/150*100)] = f"Three Chillers Equal Load: Chiller 1, 2 & 3"

    if cooling_load >= 300:
        # Idea: Full load and Part load combination
        part_load = (cooling_load - 300) # Calculate part load 
        part_load_perc = part_load/ 150 * 100 

        # Chiller 1 full load, Chiller 2 full load, Chiller 3 part load
        cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
        cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
        cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, part_load, hour, day_of_week_number, quarter, month, day_of_year))
        combination_cop_results[f"Chiller 1: 100%, Chiller 2: 100%, Chiller 3: {part_load_perc:.2f}%"] = [round(cop_1[0],2), round(cop_2[0],2), round(cop_3[0],2)]
        combination_power_results[f"Chiller 1: 100%, Chiller 2: 100%, Chiller 3: {part_load_perc:.2f}%"] = [calculate_power(cop_1[0], 150), calculate_power(cop_2[0], 150), calculate_power(cop_3[0], part_load)]
        combination_tag_results[f"Chiller 1: 100%, Chiller 2: 100%, Chiller 3: {part_load_perc:.2f}%"] = f"Three Chillers Two Full Load, One Part Load: Chiller 1 Full, Chiller 2 Full, Chiller 3 Part"

        # Chiller 1 full load, Chiller 2 part load, Chiller 3 full load
        cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
        cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, part_load, hour, day_of_week_number, quarter, month, day_of_year))
        cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
        combination_cop_results[f"Chiller 1: 100%, Chiller 2: {part_load_perc:.2f}%, Chiller 3: 100%"] = [round(cop_1[0],2), round(cop_2[0],2), round(cop_3[0],2)]
        combination_power_results[f"Chiller 1: 100%, Chiller 2: {part_load_perc:.2f}%, Chiller 3: 100%"] = [calculate_power(cop_1[0], 150), calculate_power(cop_2[0], part_load), calculate_power(cop_3[0], 150)]
        combination_tag_results[f"Chiller 1: 100%, Chiller 2: {part_load_perc:.2f}%, Chiller 3: 100%"] = f"Three Chillers Two Full Load, One Part Load: Chiller 1 Full, Chiller 2 Part, Chiller 3 Full"

        # Chiller 1 part load, Chiller 2 full load, Chiller 3 full load
        cop_1 = cop_model_1.predict(create_pred_input(temperature_input, humidity_input, part_load, hour, day_of_week_number, quarter, month, day_of_year))
        cop_2 = cop_model_2.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
        cop_3 = cop_model_3.predict(create_pred_input(temperature_input, humidity_input, 150, hour, day_of_week_number, quarter, month, day_of_year))
        combination_cop_results[f"Chiller 1: {part_load_perc:.2f}%, Chiller 2: 100%, Chiller 3: 100%"] = [round(cop_1[0],2), round(cop_2[0],2), round(cop_3[0],2)]
        combination_power_results[f"Chiller 1: {part_load_perc:.2f}%, Chiller 2: 100%, Chiller 3: 100%"] = [calculate_power(cop_1[0], part_load), calculate_power(cop_2[0], 150), calculate_power(cop_3[0], 150)]
        combination_tag_results[f"Chiller 1: {part_load_perc:.2f}%, Chiller 2: 100%, Chiller 3: 100%"] = f"Three Chillers Two Full Load, One Part Load: Chiller 1 Part, Chiller 2 Full, Chiller 3 Full"


# Function to choose best stratetgy
def best_chiller_strat_power(power_dict, cop_dict, tag_dict):
    lowest_power = float('inf')  # start with infinity
    best_chiller = None  # To keep track of the chiller configuration with the BEST COP 
    power_breakdown = None

    for chiller, power in power_dict.items():
        print(chiller, power)
        if isinstance(power, list):
            # If the value is a list, check for negative values and skip if any are found. 
            # Else, find the minimum in the list
            if any(p < 0 for p in power):
                continue
            total_power = sum(power)
            if total_power < lowest_power:
                lowest_power = total_power 
                best_chiller = chiller
                power_breakdown = power
        else:
            # If the value is a float, check if its negative.
            # Else, compare it directly
            if power < 0:
                continue
            if power < lowest_power:
                lowest_power = power 
                best_chiller = chiller
                power_breakdown = power
    
    cop_breakdown = cop_dict[best_chiller]
    tag = tag_dict[best_chiller]
    return best_chiller, round(lowest_power,2), power_breakdown, cop_breakdown, tag


def get_temp_humid_from_data(datetime_input, df):
    closest_idx = (df['record_timestamp'] - datetime_input).abs().idxmin()
    closest_datetime = df['record_timestamp'].loc[closest_idx]
    df_filter = df[df['record_timestamp'] == closest_datetime]
    # print(df_filter)
    temp = df_filter['Temperature'].values[0]
    humid = df_filter['Humidity'].values[0]
    return temp, humid

def convert_to_string(input_value):
    
    if isinstance(input_value, list):  # Check if the input is a list
        # Convert each element to string and join with commas
        return ', '.join(map(str, input_value))
    else:  # Check if the input is a float
        return str(input_value)

def optimized_strategy_row_process(row):
    # Extract necessary data from the row
    timestamp = row['record_timestamp']
    cooling_load = row["Predicted Cooling Load (kW)"]
    temperature_input = row["Air Temperature (Celsius)"]
    humidity_input = row["Relative Humidity (%)"]
    hour = row["hour"]
    day_of_week_number = row["dayofweek"]
    quarter = row["quarter"]
    month = row["month"]
    day_of_year = row["dayofyear"]

    combination_cop_results = {}
    combination_power_results = {}
    combination_tag_results = {}


    if cooling_load < 150:
        one_chiller_cop(cooling_load, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
        two_chillers_cop(cooling_load, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
        three_chillers_cop(cooling_load, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
    elif 150 <= cooling_load < 300:
        two_chillers_cop(cooling_load, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
        three_chillers_cop(cooling_load, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
    elif cooling_load >= 300:
        three_chillers_cop(cooling_load, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)

    strat, power_used, power_breakdown, cop_breakdown, tag = best_chiller_strat_power(combination_power_results, combination_cop_results, combination_tag_results)
    print()
    combined_results = {
        'record_timestamp': timestamp,
        'Strategy': strat,
        'Strategy Tag': tag,
        'Predicted Power Used (kW)': power_used,
        'Predicted Power Breakdown (kW)': convert_to_string(power_breakdown),
        'Predicted COP Breakdown': convert_to_string(cop_breakdown)
    }
    print(combined_results)

    return combined_results


def show_predict_page():
    # Load past temperature & humidity data
    df_temp_humid_past = load_past_temp_humid_data()

    # load next 4 days data
    df_forecast_4_days = load_4_days_forecast_data()

    # earliest date allowed
    start_date = df_temp_humid_past['record_timestamp'].min()
    start_date_str = start_date.strftime('%Y-%m-%d')

    # end date for downloaded data
    end_date_str = "2025/01/31"

    # maximum date and time that allows the automated time
    max_date = df_forecast_4_days['record_timestamp'].max()
    # formatted_max_date = max_date.strftime('%Y-%m-%d')  # Format it to a string

    st.markdown("<h2 style = 'text-align:center;'>Total Cooling Output and Coefficient of Performance Prediction 🔮</h2>", unsafe_allow_html=True)  

    st.markdown("<h3 style = 'text-align:center;'>Real-Time/Custom Input Strategy Prediction</h3>", unsafe_allow_html=True) 
    
    # Input Prediction section
    input_container = st.container(border=True, key="input_container")
    with input_container:
        st.markdown(f"<h4 style = 'text-align:center;'>Please input relevant information for prediction. Date range starts from {start_date} to {max_date}</h4>", unsafe_allow_html=True)

        # Get the input
        date_input = st.date_input("Date", value = 'today', key = "date_input")
        time_input = st.time_input("Time", key='time_input')

        # Convert the range strings to datetime.date objects
        start_date_hist = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date_hist = datetime.strptime(end_date_str, "%Y/%m/%d").date()
        # end_date_training = datetime.strptime(end_date_str, "%Y/%m/%d").date()

        # convert to date time format
        combined_datetime = pd.to_datetime(f"{date_input} {time_input}")
        print("Input Date", date_input)
        print("Input Time", time_input)

        # If the date input is earlier than the earliest training data, return error
        if date_input < start_date_hist:
            st.write("Please manually input the temperature and humidity")
            temp_input = 0.00
            humid_input = 0.00

        # If it is today's date
        elif date_input == date.today():
            # compare the real time and the time_input
            # if the input is earlier than current time and date
            if time_input <= datetime.now().time():
                print("Time input is less than datetime.now()")
                try:
                    df_past_data_api = load_past_temp_humid_data_api(datetime.combine(date_input, time(0,0,0)))

                    # real time api vs past data api. which one is closer to the input
                    real_time_diff = abs(real_time-combined_datetime)
                    past_data_diff = (df_past_data_api['record_timestamp'] - combined_datetime).abs().min()

                    # real time is closer to the input 
                    if real_time_diff <= past_data_diff:
                        print(f"real_time_diff ({real_time_diff}) is closer to the input than past_data_diff ({past_data_diff})")
                        temp_input = real_temp
                        humid_input = real_humid
                    # past data api is closer to the input
                    else:
                        print(f"past_data_diff ({past_data_diff}) is closer to the input than real_time_diff ({real_time_diff})")
                        temp_input, humid_input = get_temp_humid_from_data(combined_datetime,df_past_data_api)
                except Exception as e:
                    temp_input = real_temp
                    humid_input = real_humid
            #  Input is after the current time and date
            else:
                # real time api vs forecast data api. which one is closer to the input
                real_time_diff = abs(real_time-combined_datetime)
                forecast_data_diff = (df_forecast_4_days['record_timestamp'] - combined_datetime).abs().min()

                # real time is closer to the input 
                if real_time_diff <= forecast_data_diff:
                    print(f"real_time_diff ({real_time_diff}) is closer to the input than forecast_data_diff ({forecast_data_diff})")
                    temp_input = real_temp
                    humid_input = real_humid
                
                # forecast data api is closer to the input
                else:
                    print(f"forecast_data_diff ({forecast_data_diff}) is closer to the input than real_time_diff ({real_time_diff})")
                    temp_input, humid_input = get_temp_humid_from_data(combined_datetime, df_forecast_4_days)


        # If it is from 2022/07/14 t0 2025/01/31: take from the temperature and humidity historical data downloaded (parquet): df_temp_humid_past
        elif date_input >= start_date_hist and date_input <= end_date_hist:
            temp_input, humid_input = get_temp_humid_from_data(combined_datetime, df_temp_humid_past)
            
        # if it is within 1 year prior to today's date, use the API
        elif date_input > end_date_hist and date_input < date.today():
            # if it is 2025/02/01 to yesterday (datetime.today() - timedelta(days=days_to_subtract))
            df_temp_humid_past_api = load_past_temp_humid_data_api(datetime.combine(date_input, time(0,0,0)))
            temp_input, humid_input = get_temp_humid_from_data(combined_datetime, df_temp_humid_past_api)

        # if it is the next 4 days --> Use API
        elif date_input > date.today() and combined_datetime <= (datetime.today() + timedelta(days=4)):
            if combined_datetime <= max_date:
                temp_input, humid_input = get_temp_humid_from_data(combined_datetime, df_forecast_4_days)
            else:
                st.write("Please manually input the temperature and humidity")
                temp_input = 0.00
                humid_input = 0.00

        # if it is more than 4 days --> manually input
        else:
            st.write("Please manually input the temperature and humidity")
            temp_input = 0.00
            humid_input = 0.00
        
        temperature_input = st.number_input("Temperature (°C)", value = temp_input, key='temperature_input')
        humidity_input = st.number_input("Humidity (%)", value = humid_input, key = 'humidity_input')
        
        # Extract the time features of the date
        hour = time_input.hour 
        day_of_week = date_input.strftime('%A') # Full weekday name 
        day_of_week_number = date_input.weekday() # Monday is 0 and Sunday is 6 
        month = date_input.month 
        day_of_year = date_input.timetuple().tm_yday

        # Calculating the quarter 
        if 1 <= date_input.month <= 3: 
            quarter = 1 
        elif 4 <= date_input.month <= 6: 
            quarter = 2 
        elif 7 <= date_input.month <= 9: 
            quarter = 3 
        else: 
            quarter = 4

        input_data_cooling = pd.DataFrame({
        'Air Temperature (Celsius)': [temperature_input],
        'Relative Humidity (%)': [humidity_input],
        'hour': [hour],
        'dayofweek': [day_of_week_number],
        'quarter': [quarter],
        'month': [month],
        'dayofyear': [day_of_year]
        })

        ok_input = st.button("Predict 🔮", key = "ok_button") 
        if ok_input:
            predict_container = st.container(border=True, key="predict_container")
            with predict_container:

                combination_cop_results = {}
                combination_power_results = {}
                combination_tag_results = {}
                
                # Predict
                cooling_load_pred  = cooling_load_model.predict(input_data_cooling)[0]
                st.markdown("""
                    <style>
                    [data-testid="stMetricLabel"] {
                        font-size: 60px;
                    }
                    
                    [data-testid="stMetricValue"] {
                        font-size: 30px;
                        word-wrap: normal;
                    }
                    </style>
                """, unsafe_allow_html=True)

                if cooling_load_pred < 150:
                    one_chiller_cop(cooling_load_pred, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
                    two_chillers_cop(cooling_load_pred, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
                    three_chillers_cop(cooling_load_pred, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)

                elif 150 <= cooling_load_pred < 300:
                    two_chillers_cop(cooling_load_pred, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
                    three_chillers_cop(cooling_load_pred, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
                elif cooling_load_pred >= 300:
                    # (operational limit)
                    three_chillers_cop(cooling_load_pred, temperature_input, humidity_input, hour, day_of_week_number, quarter, month, day_of_year, cop_1_model, cop_2_model, cop_3_model, combination_cop_results, combination_power_results, combination_tag_results)
                
                # best_chiller, round(lowest_power,2), power_breakdown, cop_breakdown, tag
                strat, power_used, power_breakdown , cop_breakdown, tag = best_chiller_strat_power(combination_power_results, combination_cop_results, combination_tag_results)

                # estimated cost --> based on 1 jan 2024 tariff
                estimated_cost = power_used*1.523

                cola, colb = st.columns(2)
                with cola:
                    st.markdown(f"<h5>Amount of Cooling Output needed (kW)</strong>: {round(cooling_load_pred,2)}</h5>", unsafe_allow_html=True)
                with colb:
                    st.markdown(f"<h5>Best Strategy</strong>: {strat}</h5>", unsafe_allow_html=True)
    
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"<h5>Estimated Total Power Used (kW): {power_used:.2f}</h5>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<h5>Estimated Power Breakdown (kW): {power_breakdown}</h5>", unsafe_allow_html=True)
                
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(f"<h5>Estimated COP Breakdown: {cop_breakdown}</h5>", unsafe_allow_html=True)
                    # st.metric(label="Estimated Power Breakdown (kW)", value=f"{power_breakdown}")
                with col4:
                    st.markdown(f"<h5>Estimated Cost (HK$ per hour): {round(estimated_cost,2)}</h5>", unsafe_allow_html=True)
                    # st.metric(label="Estimated COP Breakdown", value=f"{cop_breakdown}")
    
    st.markdown("""---""")

    # API Forecast Data Section Prediction
    st.markdown("<h3 style = 'text-align:center;'>Weather-Based Strategy Prediction (Next 4 Days)</h3>", unsafe_allow_html=True) 
    # Create a copy
    df_forecast_4_days_pred = df_forecast_4_days.copy()
    # Rename the columns
    df_forecast_4_days_pred.rename(columns={'Temperature': "Air Temperature (Celsius)", "Humidity":"Relative Humidity (%)"}, inplace = True)

    # Filter the dataframe
    df_forecast_4_days_pred = df_forecast_4_days_pred[df_forecast_4_days_pred["record_timestamp"]>= datetime.now()] 

    min_forecast_date = df_forecast_4_days_pred['record_timestamp'].min().strftime('%Y-%m-%d')
    max_forecast_date = df_forecast_4_days_pred['record_timestamp'].max().strftime('%Y-%m-%d')

    forecast_container = st.container(border=True, key="forecast_container")
    with forecast_container:
        row1_space1, row1_1, row1_space2, row1_2, row3_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
        with row1_1:
            st.subheader(f"Hong Kong's Temperature Forecast (°C) from {min_forecast_date} to {max_forecast_date}")
            # create a Plotly figure
            fig = go.Figure()

            # Temperature line
            fig.add_trace(go.Scatter(
                x=df_forecast_4_days_pred['record_timestamp'],
                y=df_forecast_4_days_pred['Air Temperature (Celsius)'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='blue'),  
                marker=dict(size=2)  # set marker size for better visibility
            ))

            # update layout
            fig.update_layout(
                title='Temperature Forecast (°C)',
                xaxis_title='Date',
                yaxis_title='Temperature (°C)',
                legend_title = "Legend",
                xaxis = dict(showgrid=True),
                yaxis = dict(showgrid=True),
                template = 'plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        with row1_2:
            st.subheader(f"Hong Kong's Relative Humidity (%) Forecast from {min_forecast_date} to {max_forecast_date}")
            # create a Plotly figure
            fig = go.Figure()
            
            # Humidity line on y-axis
            fig.add_trace(go.Scatter(
                x=df_forecast_4_days_pred['record_timestamp'],
                y=df_forecast_4_days_pred['Relative Humidity (%)'],
                mode='lines+markers',
                name='Humidity (%)',
                line=dict(color='green'),
                marker=dict(size=2)  # Set marker size for better visibility
            ))

            # update layout
            fig.update_layout(
                title='Humidity Forecast (%)',
                xaxis_title='Date',
                yaxis_title='Humidity (%)',
                legend_title='Legend',
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
                template='plotly_white'  # Optional: use a clean template
            )

            # display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)    
        
        predict_forecast = st.button("Predict the next 4 days 🔮", key = "predict_button") 
        if predict_forecast:
            api_forecast_input_container = st.container(border=True, key="api_forecast_input_container")
            with api_forecast_input_container:
                with st.spinner("Predicting next 4 days"):
                    # Get necessary Features for Cooling Load Prediction
                    df_forecast_4_days_pred = create_features_coolingload(df_forecast_4_days_pred)
                    
                    # Keep only the relevant features
                    df_forecast_4_days_pred_copy = df_forecast_4_days_pred[cooling_features]
                    df_forecast_4_days_pred_cooling = cooling_load_model.predict(df_forecast_4_days_pred_copy)
                    df_forecast_4_days_pred['Predicted Cooling Load (kW)'] = df_forecast_4_days_pred_cooling

                    # Now Predict for the COP optimization
                    # Have to go through each row and pred
                    # Initialize the results dictionaries
                    combination_cop_results = {}
                    combination_power_results = {}
                    combination_tag_results = {}

                    results = df_forecast_4_days_pred.apply(optimized_strategy_row_process, axis=1)
                    df_forecast_4_days_strategy =  pd.DataFrame(results.tolist())
                    df_joined = pd.merge(df_forecast_4_days_pred,df_forecast_4_days_strategy,  on = 'record_timestamp', how = 'inner')
                    df_joined[["Main_Strategy", "Details"]] = df_joined['Strategy Tag'].str.split(": ", expand=True)
                    strategy_stats, most_popular, least_popular, most_cost_saving, least_cost_saving = get_strategy_summary(df_joined)


                    # plot the predicted power usage for the next 4 days
                    fig = go.Figure()

                    # Predicted power used line
                    fig.add_trace(go.Scatter(
                        x=df_joined['record_timestamp'],
                        y=df_joined['Predicted Power Used (kW)'],
                        mode='lines',
                        name='Predicted Total Power Used',
                        line=dict(color='blue')
                    ))

                    # Update layout
                    fig.update_layout(
                        title=f'Predicted Power Used (kW) from {min_forecast_date} to {max_forecast_date}.',
                        xaxis_title='Date',
                        yaxis_title='Power (kW)',
                        legend_title='Legend',
                        xaxis=dict(showgrid=True),
                        yaxis=dict(showgrid=True),
                        template='plotly_white',
                        xaxis_tickangle=-45 
                    )

                    st.plotly_chart(fig, use_container_width=True)    

                    st.markdown("---")

                    st.write("This is the operational dataframe of the forecast data")
                    df_joined = df_joined[new_order]
                    st.dataframe(df_joined, hide_index = True)

                    st.markdown("""---""")
                    st.write("This is the strategy summary of the forecast data")
                    st.dataframe(strategy_stats, hide_index=True)

                    st.markdown("""---""")

                    st.markdown("""
                    <style>
                    [data-testid="stMetricLabel"] {
                        font-size: 60px;
                    }
                    
                    [data-testid="stMetricValue"] {
                        font-size: 30px;
                        word-wrap: normal;
                    }
                    </style>
                """, unsafe_allow_html=True)
                    
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown(f"<h5 style='color: green;'><strong>Most Used Strategy</strong>: {most_popular['Main_Strategy']} {most_popular['Details']}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5><strong>Average Power Used</strong>: {most_popular['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                        # st.metric(label=f"Most Used Strategy", value=f"{most_popular['Main_Strategy']} {most_popular['Details']}")
                        # st.metric(label=f"Average Power Used", value=f"{most_popular['avg_power']:.2f} kW")
                    with col4:
                        st.markdown(f"<h5 style='color: green;'><strong>Most Cost Saving Strategy</strong>: {most_cost_saving['Main_Strategy']} {most_cost_saving['Details']}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5><strong>Average Power Used</strong>: {most_cost_saving['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                        # st.metric(label=f"Most Cost Saving Strategy", value=f"{most_cost_saving['Main_Strategy']}  {most_cost_saving['Details']}")
                        # st.metric(label=f"Average Power Used", value=f"{most_cost_saving['avg_power']:.2f} kW")

                    st.markdown("""---""")
                    col5, col6 = st.columns(2)
                    with col5:
                        st.markdown(f"<h5 style='color: red;'><strong>Least Used Strategy</strong>: {least_popular['Main_Strategy']} {least_popular['Details']}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5><strong>Average Power Used</strong>: {least_popular['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)
                        
                    with col6:
                        st.markdown(f"<h5 style='color: red;'><strong>Least Cost Saving Strategy</strong>: {least_cost_saving['Main_Strategy']}: {least_cost_saving['Details']}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5><strong>Average Power Used</strong>: {least_cost_saving['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)
                        
                   
    st.markdown("""---""")
    st.markdown("<h3 style = 'text-align:center;'>File Upload Prediction</h3>", unsafe_allow_html=True) 
    # Section whereby people can input CSV File to predict
    csv_input_container = st.container(border=True, key="csv_input_container")
    with csv_input_container:
        st.markdown("<h4 style = 'text-align:center;'>Upload CSV or Excel file with date, time, temperature and humidity to see the optimised operations</h4>", unsafe_allow_html=True)
        st.write("Here is a sample data file. You may download it for reference.")
        st.dataframe(sample_df)
        st.markdown("---")

        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type = ['csv', 'xlsx'])

        if uploaded_file is not None:
            print("File uploaded")
            with st.spinner("Predicting uploaded file."):

                # CSV file read            
                try:
                    # read CSV file
                    df_uploaded = pd.read_csv(uploaded_file)
                except Exception as e:
                    df_uploaded = pd.read_excel(uploaded_file)
                df_uploaded.rename(columns={"Temperature": "Air Temperature (Celsius)", "Humidity": "Relative Humidity (%)"}, inplace = True)
                
                # Change to proper data type
                df_uploaded['record_timestamp'] = pd.to_datetime(df_uploaded['record_timestamp'])
                
                # Get relevant features for cooling load predictions
                df_uploaded = create_features_coolingload(df_uploaded)
                df_uploaded_copy = df_uploaded[cooling_features]
                df_uploaded_pred_cooling = cooling_load_model.predict(df_uploaded_copy)
                df_uploaded['Predicted Cooling Load (kW)'] = df_uploaded_pred_cooling

                # Now Predict for the COP optimization
                # Have to go through each row and pred
                # Initialize the results dictionaries
                combination_cop_results = {}
                combination_power_results = {}
                combination_tag_results = {}

                results = df_uploaded.apply(optimized_strategy_row_process, axis=1)
                df_uploaded_strategy =  pd.DataFrame(results.tolist())
                df_joined = pd.merge(df_uploaded,df_uploaded_strategy,  on = 'record_timestamp', how = 'inner')
                df_joined[["Main_Strategy", "Details"]] = df_joined['Strategy Tag'].str.split(": ", expand=True)
                strategy_stats, most_popular, least_popular, most_cost_saving, least_cost_saving = get_strategy_summary(df_joined)
                
                st.write("This is the operational dataframe of the uploaded data file.")
                df_joined = df_joined[new_order]
                st.dataframe(df_joined, hide_index=True)

                st.markdown("""---""")
                st.write("This is the strategy summary of the uploaded data file.")
                st.dataframe(strategy_stats, hide_index=True)

                st.markdown("""---""")
                col7, col8 = st.columns(2)
                with col7:
                    st.markdown(f"<h5 style='color: green;'><strong>Most Used Strategy</strong>: {most_popular['Main_Strategy']} {most_popular['Details']}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5><strong>Average Power Used</strong>: {most_popular['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                with col8:
                    st.markdown(f"<h5 style='color: green;'><strong>Most Cost Saving Strategy</strong>: {most_cost_saving['Main_Strategy']} {most_cost_saving['Details']}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5><strong>Average Power Used</strong>: {most_cost_saving['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                st.markdown("""---""")
                col9, col10 = st.columns(2)
                with col9:
                    st.markdown(f"<h5 style='color: red;'><strong>Least Used Strategy</strong>: {least_popular['Main_Strategy']} {least_popular['Details']}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5><strong>Average Power Used</strong>: {least_popular['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                with col10:
                    st.markdown(f"<h5 style='color: red;'><strong>Least Cost Saving Strategy</strong>: {least_cost_saving['Main_Strategy']}: {least_cost_saving['Details']}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5><strong>Average Power Used</strong>: {least_cost_saving['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)
