import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from Get_Data import load_chillers_data

# Get the chiller data
df_chiller_1, df_chiller_2, df_chiller_3 = load_chillers_data()

df_chiller_1_copy = df_chiller_1.copy()
df_chiller_2_copy = df_chiller_2.copy()
df_chiller_3_copy = df_chiller_3.copy()

# Create a chiller number
df_chiller_1_copy['Chiller_Num'] = 1
df_chiller_2_copy['Chiller_Num'] = 2
df_chiller_3_copy['Chiller_Num'] = 3

# change chiller status name
df_chiller_1_copy.rename(columns={"Chiller_1_Status": "Chiller_Status"}, inplace = True)
df_chiller_2_copy.rename(columns={"Chiller_2_Status": "Chiller_Status"}, inplace = True)
df_chiller_3_copy.rename(columns={"Chiller_3_Status": "Chiller_Status"}, inplace = True)

# concat dataframes together (stacking on top of each other to get aggregated results)
df_concat = pd.concat([df_chiller_1_copy, df_chiller_2_copy, df_chiller_3_copy], ignore_index=True)

# concat dataframes together (stacking on top of each other to get aggregated results)
df_merged = pd.merge(df_chiller_1_copy, df_chiller_2_copy, on = 'record_timestamp', how='outer', suffixes=('_1', '_2'))
df_merged = pd.merge(df_merged, df_chiller_3_copy, on='record_timestamp', how='outer', suffixes=(None, "_3"))

# Get the number of chillers used
def get_strategy_tag(row):
    # Extract loading percentages for each chiller
    loading_1 = round(row['Percentage of Loading (%)_1'],2)
    loading_2 = round(row['Percentage of Loading (%)_2'],2)
    loading_3 = round(row['Percentage of Loading (%)'], 2)
    
    # Create tags based on the loading values
    chillers_used = []
    
    if pd.notna(loading_1) and loading_1 > 0:
        chillers_used.append(f"Chiller 1")
    if pd.notna(loading_2) and loading_2 > 0:
        chillers_used.append(f"Chiller 2")
    if pd.notna(loading_3) and loading_3 > 0:
        chillers_used.append(f"Chiller 3")
    
    # Determine the Strategy Tag based on the number of chillers used
    num_chillers = len(chillers_used)
    if num_chillers == 1:
        return f"One Chiller: {chillers_used[0]}"
    elif num_chillers == 2:
        return f"Two Chillers: {', '.join(chillers_used)}"
    elif num_chillers == 3:
        return f"Three Chillers: {', '.join(chillers_used)}"
    else:
        return "No Chillers in Use"

# Apply the function to each row in the DataFrame
df_merged['Strategy Tag'] = df_merged.apply(get_strategy_tag, axis=1)
df_merged[['Main_Strategy', 'Details']] = df_merged['Strategy Tag'].str.split(":", expand= True)

# Count the occurrences of each Main_Strategy
strategy_counts_ori = df_merged['Main_Strategy'].value_counts().reset_index()
strategy_counts_ori.columns = ['Main_Strategy', 'Count']

# month list
months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
month_mapping = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
    }

def plot_operation(month, year):
    month_numeric = month_mapping[month]
    df_chiller_1_operation = df_chiller_1_copy.copy()
    df_chiller_2_operation = df_chiller_2_copy.copy()
    df_chiller_3_operation = df_chiller_3_copy.copy()

    df_chiller_1_operation = df_chiller_1_operation[(df_chiller_1_operation['record_timestamp'].dt.month == month_numeric) & (df_chiller_1_operation['record_timestamp'].dt.year == year)]
    
    df_chiller_2_operation = df_chiller_2_operation[(df_chiller_2_operation['record_timestamp'].dt.month == month_numeric) & (df_chiller_2_operation['record_timestamp'].dt.year == year)]    
    
    df_chiller_3_operation = df_chiller_3_operation[(df_chiller_3_operation['record_timestamp'].dt.month == month_numeric) & (df_chiller_3_operation['record_timestamp'].dt.year == year)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x = df_chiller_1_operation['record_timestamp'], y = df_chiller_1_operation['COP_Own'], mode ='lines', name="Chiller 1", opacity=0.4))
    fig.add_trace(go.Scatter(x = df_chiller_2_operation['record_timestamp'], y = df_chiller_2_operation['COP_Own'], mode ='lines', name="Chiller 2", opacity=0.4))
    fig.add_trace(go.Scatter(x = df_chiller_3_operation['record_timestamp'], y = df_chiller_3_operation['COP_Own'], mode ='lines', name="Chiller 3", opacity=0.4))

    fig.update_layout(
        title=f"TVLV Chiller Operational Timeframe - {month} {year}",
        xaxis_title='DateTime',
        yaxis_title='COP_Own',
        legend_title='Chillers'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    print(f"{month} {year} has been plotted")
    

def show_explore_page():
    st.markdown("<h2 style = 'text-align:center;'>Explore Page 🔎</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style = 'text-align:center;'>Snapshot of each chiller's data</h3>", unsafe_allow_html=True)
    # st.write("""
    #          ### Snapshot of the chiller data""")
    option = st.selectbox(
    "Choose your chiller",
    ("Chiller 1", "Chiller 2", "Chiller 3"),
    )

    if option == "Chiller 1":
        st.write("Chiller 1 Data")    
        st.dataframe(df_chiller_1_copy, hide_index=True)
    elif option == "Chiller 2":
        st.write("Chiller 2 Data")
        st.dataframe(df_chiller_2_copy, hide_index=True)
    else:
        st.write("Chiller 3 Data")
        st.dataframe(df_chiller_3_copy, hide_index=True)
    
    st.markdown("""---""")


    st.markdown("<h3 style = 'text-align:center;'>Three Chillers Operation Timeline</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style = 'text-align:center;'>Date range starts from July 2022 to March 2024</h4>", unsafe_allow_html=True)
    year_input = st.selectbox('Year', (2022, 2023, 2024), key = 'year_selectbox')
    month_input = st.selectbox('Month', (months) , key='month_selectbox')

    plot_button = st.button("Plot 📈", key = 'plot_button')
    if plot_button:
        month_numeric = month_mapping[month_input]

        if (year_input == 2022 and month_numeric < 7) or (year_input == 2024 and month_numeric > 3):
            st.error("Invalid month/year. Please select from July 2022 and March 2024.")
        else:
            plot_container = st.container(border= True)
            with plot_container:
                plot_operation(month_input, year_input)
                st.markdown("""Tai Lung Veterinary Lab does not take a specific strategy in operating their chillers. They either use a **Round-Robin style** whereby chillers are operated one by one (e.g., November 2022) with no overlap of each chiller's line chart
                             OR **multiple chillers operating simultaneously** (e.g., July 2022), with the chillers' line charts overlapping    .""")

    st.markdown("""---""")
    st.markdown("<h3 style = 'text-align:center;'>Tai Lung Veterinary Lab Chiller Operation Strategy Breakdown</h3>", unsafe_allow_html=True)

    # Look at the chiller operation strategy breakdown of the Tai Lung Veterinary Lab
    # Create a Plotly bar chart
    fig = px.bar(strategy_counts_ori, x='Main_Strategy', y='Count', title='TVLV Chiller Strategy Analysis',
                labels={'Main_Strategy': 'Main Strategy', 'Count': 'Count'},
                color='Count', text='Count')

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)


    st.plotly_chart(fig)

    st.write("""Here is the detailed breakdown:""")
    st.write("""
             - One Chiller Used: **45518** 
                - Chiller 1: 16378
                - Chiller 3: 15028
                - Chiller 2: 14112
             - No Chillers Used: **2370**
             - Two Chillers Used: **1663**
                - Chiller 1, Chiller 3: 705
                - Chiller 1, Chiller 2: 566
                - Chiller 2, Chiller 3: 392
             - Three Chillers Used: **3**""")

    st.markdown("""---""")

    # Features to see correlation heatmap
    heatmap_features = ['COP_Own', 'Power Supply (kW)', 'Current Cooling Output (kW)','Water Flow Data (L/s)','Return Water Temp', 'Supply Water Temp','Percentage of Loading (%)', 'Air Temperature (Celsius)','Relative Humidity (%)']

    st.markdown("<h3 style = 'text-align:center;'>Correlation Heatmap of all three chillers' features</h3>", unsafe_allow_html=True)

    st.markdown("""According to [Ho & Yu (2021)](https://www.sciencedirect.com/science/article/pii/S036054422032483X?via%3Dihub) and [Yu et al. (2017)](https://www.sciencedirect.com/science/article/abs/pii/S0378778816309860?via=ihub), there are a few factors that affect the COP of chillers:""")
    st.markdown("1. **Outdoor Temperature and Humidity**: High cooling load is required to cool down and more energy is needed to remove the moisture from the air.")
    st.markdown("2. **Percentage of Load**: Chillers have performance curve. The percentage of load that are below the threshold leads to inefficiencies.")
    st.markdown("3. **Chiller Water Flow Rate and its Temperature**: The heat exchange process is affected thus affecting the evaporating pressure.")

    chiller_option = st.selectbox("Choose the chiller(s)",
    ("Chiller 1", "Chiller 2", "Chiller 3", "All 3 Chillers Aggregated"), key = 'chiller_option')

    if chiller_option == "Chiller 1":
        heatmap_df_temp = df_chiller_1_copy[heatmap_features]
    elif chiller_option == "Chiller 2":
        heatmap_df_temp = df_chiller_2_copy[heatmap_features]
    elif chiller_option == "Chiller 3":
        heatmap_df_temp = df_chiller_3_copy[heatmap_features]
    else:
        heatmap_df_temp = df_concat[heatmap_features]


    # # visualization of correlation coefficient of chillers
    corr_df = heatmap_df_temp.corr()  # Calculate correlation
        
    # create a heatmap using Plotly
    fig = px.imshow(corr_df,
                    color_continuous_scale='Reds',
                    title=f'Correlation Heatmap - {chiller_option}',
                    labels=dict(x='Features', y='Features', color='Correlation'),
                    aspect='auto',  # Adjusting the aspect ratio
                    text_auto=True)  # Adding correlation values as text annotations

    # update layout for better readability
    fig.update_layout(font=dict(size=16))  # Adjust font size for the heatmap

    # display the heatmap in Streamlit
    st.plotly_chart(fig)

    st.write("""**:blue[Water Flow, Temperature of Return and Supply Water and Percentage of Loading]** have **stronger correlation with the COP**.
            However, **:red[Air Temperature and Relative Humidity]** does not which does not align with the research findings by [Ho & Yu (2021)](https://www.sciencedirect.com/science/article/pii/S036054422032483X?via%3Dihub) and [Yu et al. (2017)](https://www.sciencedirect.com/science/article/abs/pii/S0378778816309860?via=ihub).""")
    st.write("""The following few plots will show how **:red[Air Temperature and Relative Humidity]** have an impact on the Chiller's COP.""")
    
    # Hong Kong's average climate and weather 
    st.markdown("""---""")
    st.markdown("<h3 style = 'text-align:center;'>Climate and Weather Averages in Hong Kong through the year</h3>", unsafe_allow_html=True)
    left_co, cent_co,last_co = st.columns([0.2, 5, 0.2])
    cent_co.image("https://www.hko.gov.hk/en/cis/images/climte_hk_fig_combine.jpg", caption="Climate of Hong Kong, source: Hong Kong Observatory", use_container_width = True)
    st.write("""Hong Kong has four seasons: """)
    st.write("""
             - **Spring (March to May)** 
             - **Summer (June to August)**
             - **Autumn (September to November)**
             - **Winter (December to February)**""")
    st.write("""**Temperature** starts to rise drastically from March, transitioning from winter to spring, and peaks in July during the summer, before falling in September as autumn arrives.""") 
    st.write("""**Relative humidity** is high from March to August before dropping in September as autumn arrives.""")

    st.markdown("""---""")

    # Features against time (month and hour)
    st.markdown("<h3 style = 'text-align:center;'>Cooling Load needed, Power Used and COP over time</h3>", unsafe_allow_html=True)
    
    seasons = {"Spring": [3,4,5], "Summer": [6,7,8], "Autumn": [9,10,11], "Winter":[12,1,2]}

    # Trend by time: how much cooling is needed throughout the day and year
    cooling_container = st.container(border=True, key="cooling_container")
    with cooling_container:
        row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
        # Power by hour
        with row4_1:
            st.subheader("Cooling Load needed against hours of the day")

            chiller_option_cool_hour = st.selectbox("Choose the chiller(s)",
            ("Chiller 1", "Chiller 2", "Chiller 3", "All 3 Chillers Aggregated"), key = 'chiller_option_cool_hour')
            seasons_option_cool = st.selectbox("Choose the season",
            ("Spring", "Summer", "Autumn", "Winter"), key = 'seasons_option_cool')

            if chiller_option_cool_hour == "Chiller 1":
                avg_cool_chiller_hour = df_chiller_1_copy
            elif chiller_option_cool_hour == "Chiller 2":
                avg_cool_chiller_hour = df_chiller_2_copy
            elif chiller_option_cool_hour == "Chiller 3":
                avg_cool_chiller_hour = df_chiller_3_copy
            else:
                avg_cool_chiller_hour = df_concat
            
            # Extract hour
            avg_cool_chiller_hour['hour'] = avg_cool_chiller_hour['record_timestamp'].dt.hour  
            # Extract month
            avg_cool_chiller_hour['month'] = avg_cool_chiller_hour['record_timestamp'].dt.month

            # Filter the dataframe into the respective months
            avg_cool_chiller_hour = avg_cool_chiller_hour[avg_cool_chiller_hour['month'].isin(seasons[seasons_option_cool])]

            avg_cool_chiller_hour_plot = avg_cool_chiller_hour.groupby('hour')['Current Cooling Output (kW)'].mean().reset_index()

            average_cool_hour = round(avg_cool_chiller_hour_plot['Current Cooling Output (kW)'].mean(),2)

            fig_cooling_hour = go.Figure()
            fig_cooling_hour.add_trace(go.Scatter(x=avg_cool_chiller_hour_plot['hour'], 
                                        y=avg_cool_chiller_hour_plot['Current Cooling Output (kW)'], 
                                        mode='lines+markers', 
                                        name='Average Cooling Used (kW)',
                                        line=dict(color='blue')))

            fig_cooling_hour.update_layout(title=f'Average Hourly Cooling Produced by {chiller_option_cool_hour} during {seasons_option_cool}',
                                xaxis_title='Hour',
                                yaxis_title='Average Cooling Used (kW)',
                                xaxis=dict(tickvals=list(range(24))),  # Show all 12 hours
                                template='plotly_white')
            fig_cooling_hour.update_xaxes(tickangle=30)  # Rotate x-axis labels

            # Show the hourly chart in Streamlit
            st.plotly_chart(fig_cooling_hour)
            st.markdown(f"The average hourly cooling load needed for **{chiller_option_cool_hour.lower()}** during **{seasons_option_cool.lower()}** season is **{average_cool_hour} kW**.")
            st.markdown("The average amount of cooling needed during the **:red[summer]** is **:red[higher]** as the atmospheric temperature and relative humidity is **:red[higher]**, while the average cooling load needed is **:blue[lower]** during **:blue[winter]** as the temperature and relative humidity is **:blue[lower]**.")
            st.markdown("The high atmospheric temperature **:red[increases the rate of heat transfer and the amount of heat entering the laboratory]**, requiring more cooling to offset this heat.")
            st.markdown("Additionally, a **:red[high relative humidity]** means the **:red[air contains more moisture]**. The chiller must produce **:red[more cooling load]** to **cool the air** and **remove the moisture** from it.")
            
        with row4_2:
            st.subheader("Cooling Load needed against months of the year")
            chiller_option_cool_month = st.selectbox("Choose the chiller(s)",
            ("Chiller 1", "Chiller 2", "Chiller 3", "All 3 Chillers Aggregated"), key = 'chiller_option_cool_month')

            if chiller_option_cool_month == "Chiller 1":
                avg_cool_chiller_month = df_chiller_1_copy
            elif chiller_option_cool_month == "Chiller 2":
                avg_cool_chiller_month = df_chiller_2_copy
            elif chiller_option_cool_month == "Chiller 3":
                avg_cool_chiller_month = df_chiller_3_copy
            else:
                avg_cool_chiller_month = df_concat

            # Extract month
            avg_cool_chiller_month['month'] = avg_cool_chiller_month['record_timestamp'].dt.month

            avg_cool_chiller_month_plot = avg_cool_chiller_month.groupby('month')['Current Cooling Output (kW)'].mean().reset_index()
            
            average_cool_month = round(avg_cool_chiller_month_plot['Current Cooling Output (kW)'].mean(),2)
            
            fig_cooling_month = go.Figure()
            fig_cooling_month.add_trace(go.Scatter(x=avg_cool_chiller_month_plot['month'], 
                                        y=avg_cool_chiller_month_plot['Current Cooling Output (kW)'], 
                                        mode='lines+markers', 
                                        name='Average Cooling Used (kW)',
                                        line=dict(color='blue')))

            fig_cooling_month.update_layout(title=f'Average Monthly Cooling Produced by {chiller_option_cool_month}',
                                xaxis_title='Month',
                                yaxis_title='Average Cooling Used (kW)',
                                xaxis=dict(tickvals=list(range(12))),  # Show all 12 hours
                                template='plotly_white')
            fig_cooling_month.update_xaxes(tickangle=30)  # Rotate x-axis labels

            # Show the hourly chart in Streamlit
            st.plotly_chart(fig_cooling_month)
            st.markdown(f"The average monthly cooling load needed for **{chiller_option_cool_month.lower()}** is **{average_cool_month} kW**.")
            st.markdown("Hong Kong's **:red[atmospheric temperature]** and **:red[relative humidity]** rises from **:red[March]**, thus the average cooling load needed **:red[increases more drastically from March]**.")
            st.markdown("Also, **:red[higher atmospheric temperatures]** and **:red[relative humidity]** is recorded during  **:red[summer (June to August)]**, which requires **:red[more cooling load]** to cool down the building.")

        # Trend by time: how much Power is used throughout the day and year
    power_container = st.container(border=True, key="power_container")
    with power_container:
        row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))

        # Power by hour
        with row3_1:
            st.subheader("Power Used against hours of the day")
            chiller_option_power_hour = st.selectbox("Choose the chiller(s)",
            ("Chiller 1", "Chiller 2", "Chiller 3", "All 3 Chillers Aggregated"), key = 'chiller_option_power_hour')
            seasons_option_power = st.selectbox("Choose the season",
            ("Spring", "Summer", "Autumn", "Winter"), key = 'seasons_option_power')

            if chiller_option_power_hour == "Chiller 1":
                avg_power_chiller_hour = df_chiller_1_copy
            elif chiller_option_power_hour == "Chiller 2":
                avg_power_chiller_hour = df_chiller_2_copy
            elif chiller_option_power_hour == "Chiller 3":
                avg_power_chiller_hour = df_chiller_3_copy
            else:
                avg_power_chiller_hour = df_concat
            
            # Extract hour
            avg_power_chiller_hour['hour'] = avg_power_chiller_hour['record_timestamp'].dt.hour  
            # Extract month
            avg_power_chiller_hour['month'] = avg_power_chiller_hour['record_timestamp'].dt.month

            # Filter the dataframe into the respective months
            avg_power_chiller_hour = avg_power_chiller_hour[avg_power_chiller_hour['month'].isin(seasons[seasons_option_power])]

            avg_power_chiller_hour_plot = avg_power_chiller_hour.groupby('hour')['Power Supply (kW)'].mean().reset_index()

            average_power_hour = round(avg_power_chiller_hour_plot['Power Supply (kW)'].mean(),2)

            fig_power_hour = go.Figure()
            fig_power_hour.add_trace(go.Scatter(x=avg_power_chiller_hour_plot['hour'], 
                                        y=avg_power_chiller_hour_plot['Power Supply (kW)'], 
                                        mode='lines+markers', 
                                        name='Average Power Used (kW)',
                                        line=dict(color='blue')))

            fig_power_hour.update_layout(title=f'Average Hourly Power Used by {chiller_option_power_hour} during {seasons_option_power}',
                                xaxis_title='Hour',
                                yaxis_title='Average Power Used (kW)',
                                xaxis=dict(tickvals=list(range(24))),  # Show all months
                                template='plotly_white')
            
            fig_power_hour.update_xaxes(tickangle=30)  # Rotate x-axis labels
            # Show the hourly chart in Streamlit
            st.plotly_chart(fig_power_hour)
            st.markdown(f"The average hourly power usage for **{chiller_option_power_hour.lower()}** during **{seasons_option_power.lower()}** season is **{average_power_hour} kW**.")
            st.markdown("**:red[Higher temperature during the day]** (hours 13 to 17), therefore the amount of power needed to cool the building is **:red[more]**.")

        with row3_2:
            st.subheader("Power Used against months of the year")
            chiller_option_power_month = st.selectbox("Choose the chiller(s)",
            ("Chiller 1", "Chiller 2", "Chiller 3", "All 3 Chillers Aggregated"), key = 'chiller_option_power_month')

            if chiller_option_power_month == "Chiller 1":
                avg_power_chiller_month = df_chiller_1_copy
            elif chiller_option_power_month == "Chiller 2":
                avg_power_chiller_month = df_chiller_2_copy
            elif chiller_option_power_month == "Chiller 3":
                avg_power_chiller_month = df_chiller_3_copy
            else:
                avg_power_chiller_month = df_concat

            # Extract month
            avg_power_chiller_month['month'] = avg_power_chiller_month['record_timestamp'].dt.month

            avg_power_chiller_month_plot = avg_power_chiller_month.groupby('month')['Power Supply (kW)'].mean().reset_index()
            
            average_power_month = round(avg_power_chiller_month_plot['Power Supply (kW)'].mean(),2)

            fig_power_month = go.Figure()
            fig_power_month.add_trace(go.Scatter(x=avg_power_chiller_month_plot['month'], 
                                        y=avg_power_chiller_month_plot['Power Supply (kW)'], 
                                        mode='lines+markers', 
                                        name='Average Power Used (kW)',
                                        line=dict(color='blue')))

            fig_power_month.update_layout(title=f'Average Monthly Power Used by {chiller_option_power_month}',
                                xaxis_title='Month',
                                yaxis_title='Average Power Used (kW)',
                                xaxis=dict(tickvals=list(range(12))),  # Show all months
                                template='plotly_white')
            fig_power_month.update_xaxes(tickangle=30)  # Rotate x-axis labels

            # Show the hourly chart in Streamlit
            st.plotly_chart(fig_power_month)
            st.markdown(f"The average monthly power usage for **{chiller_option_power_month.lower()}** is **{average_power_month} kW**.")
            st.markdown("Hong Kong's atmospheric temperature and relative humidity **:red[rises]** from **:red[March]**, causing the **:red[average power needed to increase more drastically]**, before **:blue[dropping]** when temperature and humidity starts to **:blue[fall during September (autumn)]**.")
            st.markdown("As a **:red[higher cooling is needed]**, the chillers have to **:red[work harder]** to **remove the heat and moisture** from the air.")
   
    # Trend by time: how much COP is used throughout the day and year
    cop_container = st.container(border=True, key="cop_container")
    with cop_container:
        row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))

        # COP by hour
        with row2_1:
            st.subheader("COP against hours of the day")
            chiller_option_cop_hour = st.selectbox("Choose the chiller(s)",
            ("Chiller 1", "Chiller 2", "Chiller 3", "All 3 Chillers Aggregated"), key = 'chiller_option_cop_hour')
            seasons_option_cop = st.selectbox("Choose the season",
            ("Spring", "Summer", "Autumn", "Winter"), key = 'seasons_option_cop')

            if chiller_option_cop_hour == "Chiller 1":
                avg_cop_chiller_hour = df_chiller_1_copy
            elif chiller_option_cop_hour == "Chiller 2":
                avg_cop_chiller_hour = df_chiller_2_copy
            elif chiller_option_cop_hour == "Chiller 3":
                avg_cop_chiller_hour = df_chiller_3_copy
            else:
                avg_cop_chiller_hour = df_concat
            
            print(avg_cop_chiller_hour.columns)
            
            # Extract hour
            avg_cop_chiller_hour['hour'] = avg_cop_chiller_hour['record_timestamp'].dt.hour  
            # Extract month
            avg_cop_chiller_hour['month'] = avg_cop_chiller_hour['record_timestamp'].dt.month

            # Filter the dataframe into the respective months
            avg_cop_chiller_hour = avg_cop_chiller_hour[avg_cop_chiller_hour['month'].isin(seasons[seasons_option_cop])]

            avg_cop_chiller_hour_plot = avg_cop_chiller_hour.groupby('hour')['COP_Own'].mean().reset_index()

            average_cop_hour = round(avg_cop_chiller_hour_plot['COP_Own'].mean(),2)

            # plotly figure
            fig_cop_hour = go.Figure()
            fig_cop_hour.add_trace(go.Scatter(x=avg_cop_chiller_hour_plot['hour'], 
                                        y=avg_cop_chiller_hour_plot['COP_Own'], 
                                        mode='lines+markers', 
                                        name='Average COP',
                                        line=dict(color='blue')))

            fig_cop_hour.update_layout(title=f'Average Hourly COP of {chiller_option_cop_hour} during {seasons_option_cop}',
                                xaxis_title='Hour of the Day',
                                yaxis_title='Average COP',
                                xaxis=dict(tickvals=list(range(24))),  # Show all hours from 0 to 23
                                template='plotly_white')
            
            fig_cop_hour.update_xaxes(tickangle=30)  # Rotate x-axis labels

            # Show the hourly chart in Streamlit
            st.plotly_chart(fig_cop_hour)

            st.markdown(f"The average hourly COP for **{chiller_option_cop_hour.lower()}** during **{seasons_option_cop.lower()} season** is **{average_cop_hour}**.")
            st.markdown("**:red[Higher atmospheric temperature]** during the day (late morning to afternoon), thus the **:red[COP value is lower]** during the hour 10 to 15.")

        # COP by month
        with row2_2:
            st.subheader("COP against months of the year")
            chiller_option_cop_month = st.selectbox("Choose the chiller(s)",
            ("Chiller 1", "Chiller 2", "Chiller 3", "All 3 Chillers Aggregated"), key = 'chiller_option_cop_month')

            if chiller_option_cop_month == "Chiller 1":
                avg_cop_chiller_month = df_chiller_1_copy
            elif chiller_option_cop_month == "Chiller 2":
                avg_cop_chiller_month = df_chiller_2_copy
            elif chiller_option_cop_month == "Chiller 3":
                avg_cop_chiller_month = df_chiller_3_copy
            else:
                avg_cop_chiller_month = df_concat

            # Extract month
            avg_cop_chiller_month['month'] = avg_cop_chiller_month['record_timestamp'].dt.month

            avg_cop_chiller_month_plot = avg_cop_chiller_month.groupby('month')['COP_Own'].mean().reset_index()
            
            average_cop_month = round(avg_cop_chiller_month_plot['COP_Own'].mean(),2)

            # plotly figure
            fig_cop_month = go.Figure()
            fig_cop_month.add_trace(go.Scatter(x=avg_cop_chiller_month_plot['month'], 
                                        y=avg_cop_chiller_month_plot['COP_Own'], 
                                        mode='lines+markers', 
                                        name='Average COP',
                                        line=dict(color='blue')))

            fig_cop_month.update_layout(title=f'Average Monthly COP of {chiller_option_cop_month}',
                                xaxis_title='Month',
                                yaxis_title='Average COP',
                                xaxis=dict(tickvals=list(range(12))),  # Show all months
                                template='plotly_white')

            fig_cop_month.update_xaxes(tickangle=30)  # Rotate x-axis labels

            # Show the monthly chart in Streamlit
            st.plotly_chart(fig_cop_month)
            st.markdown(f"The average monthly COP for **{chiller_option_cop_month.lower()}** is **{average_cop_month}**.")
            st.markdown("**:red[Atmospheric temperature and relative humidity]** picks up starting from the month of March and peaks in **:red[June]**.")
            st.markdown("The chillers have to **:red[consume more energy to maintain the desired cooling output]**, which explains the **:red[reduced efficiency]**, thus **:red[lower COP value]**.")