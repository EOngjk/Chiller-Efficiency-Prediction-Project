import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import statsmodels.api as sm
import warnings
import plotly.graph_objs as go
import plotly.express as px
warnings.filterwarnings('ignore')
from Get_Data import load_chillers_data, load_merged_chillers_data,load_simulated_optimized_chillers

# Get individual chiller data
data_chiller_1, data_chiller_2, data_chiller_3 = load_chillers_data()

# Get merged chiller data
df_joined = load_merged_chillers_data()

# Get Simulated optimized data
df_simulated_data = load_simulated_optimized_chillers()

# Create a dictionary mapping month numbers to month names
month_mapping = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

new_order = ["record_timestamp","Air Temperature (Celsius)","Relative Humidity (%)","Total Cooling Output (kW)", "Predicted Total Cooling Output (kW)","Total Power Supply (kW)","Predicted Total Power Used (kW)",
             "Predicted Power Breakdown (kW)", "Predicted COP Breakdown", "Strategy","Main_Strategy","Details"]

# Function to see strategy summary
def get_strategy_summary(df):
    strategy_stats = df.groupby(['Main_Strategy', 'Details']).agg(count=('Main_Strategy', 'size'),avg_power=('Predicted Total Power Used (kW)', 'mean')).reset_index()
    
    # Round up to 2 dp
    strategy_stats['avg_power'] = strategy_stats['avg_power'].round(2)

    most_popular = strategy_stats.loc[strategy_stats['count'].idxmax()]
    least_popular = strategy_stats.loc[strategy_stats['count'].idxmin()]
    most_cost_saving = strategy_stats.loc[strategy_stats['avg_power'].idxmin()]
    least_cost_saving = strategy_stats.loc[strategy_stats['avg_power'].idxmax()]

    return strategy_stats, most_popular, least_popular, most_cost_saving, least_cost_saving

# This is the simulation for the project
# will be doing by month simulation. So users just have to input the desired month and then click run.
def show_simulation_page():
    st.markdown("<h2 style = 'text-align:center;'>Simulation and Visualization 🕹️📈</h2>", unsafe_allow_html=True)
    
    simulation_df_joined = df_simulated_data.copy()

    st.markdown("<h3 style = 'text-align:center;'>Please select the desired month for simulation</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style = 'text-align:center;'>Date range starts from 07/2022 to 03/2024</h4>", unsafe_allow_html=True)

    period_input = st.selectbox("Which period would you like to simulate?", ("Whole period", "Custom Period"), key = 'period_input')
    if period_input == "Custom Period":
        year_input = st.slider('Year', min_value=2022, max_value=2024, value=2022, key = 'year_slider')
        month_input = st.slider('Month', min_value=1, max_value=12, value=7, step=1, key='month_slider')
    

    simulate_button = st.button("Run simulation 🏃", key='simulate_button')
    if simulate_button:
        # Validate the input
        if period_input == "Whole period":
            container = st.container(border=True)
            with container:
                st.success(f"Running simulation for the whole period (2022 July to 2024 March)!")
                simulation_df_joined = simulation_df_joined.dropna(subset=["Total Power Supply (kW)"])
                simulation_df_joined = simulation_df_joined[simulation_df_joined['Total Cooling Output (kW)'] > 0]
                
                # For now filter the Predicted total Power Used > 0
                simulation_df_joined = simulation_df_joined[simulation_df_joined['Predicted Total Power Used (kW)']> 0]
                
                actual_power = sum(simulation_df_joined['Total Power Supply (kW)'])
                predicted_power = sum(simulation_df_joined['Predicted Total Power Used (kW)'])
                power_saved = actual_power - predicted_power


                # Display results in columns
                st.subheader("Actual vs Simulation Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label=f"Actual Power Supply (2022 July to 2024 March)", value=f"{actual_power:.2f} kW")
                with col2:
                    st.metric(label=f"Predicted Power Supply (2022 July to 2024 March)", value=f"{predicted_power:.2f} kW")

                estimated_savings = power_saved*1.523*15/60

                # Conditional Formatting for power saved
                if power_saved > 0:
                    st.subheader("Power Savings")
                    st.markdown(f"<h5 style='color: green;'>Amount of Power Saved: {power_saved:.2f} kW</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5 style='color: green;'>Percentage of Power Saved: {(power_saved/actual_power)*100:.2f}%</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5 style='color: green; '>Estimated Savings: HK$ {estimated_savings:.2f}</h5>", unsafe_allow_html=True)
                else:
                    st.subheader("Power Wasted")
                    st.markdown(f"<h5 style='color: red;'>Additional of Power Used: {-power_saved:.2f} kW</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5 style='color: red;'>Percentage of Additional Power Used: {(-power_saved/actual_power)*100:.2f}%</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5 style='color: red;'>Estimated Additional Costs: HK$ {-estimated_savings:.2f}</h5>", unsafe_allow_html=True)

            # Visualize the plot (line graph)            
            # a Plotly figure
            with st.container(border=True):

                fig = go.Figure()

                # Actual power used line
                fig.add_trace(go.Scatter(
                    x=simulation_df_joined['record_timestamp'],
                    y=simulation_df_joined['Total Power Supply (kW)'],
                    mode='lines',
                    name='Actual Total Power Used',
                    line=dict(color='red')
                ))

                # Predicted power used line
                fig.add_trace(go.Scatter(
                    x=simulation_df_joined['record_timestamp'],
                    y=simulation_df_joined['Predicted Total Power Used (kW)'],
                    mode='lines',
                    name='Predicted Total Power Used',
                    line=dict(color='blue')
                ))

                # Update layout
                fig.update_layout(
                    title=f'Actual Power Used VS Strategised Power Used (kW) for whole period (2022 July to 2024 March)',
                    xaxis_title='Date',
                    yaxis_title='Power (kW)',
                    legend_title='Legend',
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    template='plotly_white',
                    xaxis_tickangle=-45  # Rotate x-axis labels for better readability
                )

                # Display the Plotly chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

            with st.container(border=True):
                # Display the dataframe
                st.write("This is the operational dataframe of the simulated data")
                simulation_df_joined.drop(columns=['hour', 'dayofweek', 'quarter', 'month', 'dayofyear'], inplace=True)

                # Split the the column "Strategy Tag"
                simulation_df_joined[["Main_Strategy", "Details"]] = simulation_df_joined['Strategy Tag'].str.split(": ", expand=True)

                simulation_df_joined = simulation_df_joined[new_order]

                st.dataframe(simulation_df_joined, hide_index=True)
                # st.write(simulation_df_joined.columns)
                
                strategy_stats, most_popular, least_popular, most_cost_saving, least_cost_saving = get_strategy_summary(simulation_df_joined)
                
            with st.container(border=True):

                st.write("This is the strategy summary of the simulated data")
                st.dataframe(strategy_stats, hide_index=True)
            
            with st.container(border=True):
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
                
                # Print the most used strategy and most cost saving strategy
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(f"<h5 style='color: green;'><strong>Most Used Strategy (2022 July to 2024 March)</strong>: {most_popular['Main_Strategy']} {most_popular['Details']}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5><strong>Average Power Used</strong>: {most_popular['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                    # st.metric(label=f"Most Used Strategy ({month} {year_input})", value=f"{most_popular['Main_Strategy']} {most_popular['Details']}")
                    # st.metric(label=f"Average Power Used", value=f"{most_popular['avg_power']:.2f} kW")
                # st.markdown("""---""")
                with col4:
                    st.markdown(f"<h5 style='color: green;'><strong>Most Cost Saving Strategy (2022 July to 2024 March)</strong>: {most_cost_saving['Main_Strategy']} {most_cost_saving['Details']}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5><strong>Average Power Used</strong>: {most_cost_saving['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)
                    # st.metric(label=f"Most Cost Saving Strategy ({month} {year_input})", value=f"{most_cost_saving['Main_Strategy']}  {most_cost_saving['Details']}")
                    # st.metric(label=f"Average Power Used", value=f"{most_cost_saving['avg_power']:.2f} kW")
                
                st.markdown("""---""")
                
                # Print the least used strategy and the least cost saving strategy
                col5, col6 = st.columns(2)
                with col5:
                    st.markdown(f"<h5 style='color: red;'><strong>Least Used Strategy (2022 July to 2024 March)</strong>: {least_popular['Main_Strategy']} {least_popular['Details']}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5><strong>Average Power Used</strong>: {least_popular['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                    # st.metric(label=f"Least Used Strategy ({month} {year_input})", value=f"{least_popular['Main_Strategy']} {least_popular['Details']}")
                    # st.metric(label=f"Average Power Used", value=f"{least_popular['avg_power']:.2f} kW")
                # st.markdown("""---""")

                with col6:
                    st.markdown(f"<h5 style='color: red;'><strong>Least Cost Saving Strategy (2022 July to 2024 March)</strong>: {least_cost_saving['Main_Strategy']}: {least_cost_saving['Details']}</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h5><strong>Average Power Used</strong>: {least_cost_saving['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                    # st.metric(label=f"Least Cost Saving Strategy ({month} {year_input})", value=f"{least_cost_saving['Main_Strategy']}: {least_cost_saving['Details']}")
                    # st.metric(label=f"Average Power Used", value=f"{least_cost_saving['avg_power']:.2f} kW")
        else:        
            if (year_input == 2022 and month_input < 7) or (year_input == 2024 and month_input > 3):
                st.error("Invalid month/year. Please select from July 2022 and March 2024.")
            else:
                container = st.container(border=True)
                with container:
                    month = month_mapping[month_input]

                    st.success(f"Running simulation for {year_input} {month}!")
                    simulation_df_joined = simulation_df_joined[
                        (simulation_df_joined['month'] == month_input) & 
                        (simulation_df_joined['record_timestamp'].dt.year == year_input)
                    ]
                    
                    # Drop NA & Keep only the cooling output is greater than 0
                    simulation_df_joined = simulation_df_joined.dropna(subset=["Total Power Supply (kW)"])
                    simulation_df_joined = simulation_df_joined[simulation_df_joined['Total Cooling Output (kW)'] > 0]
                    
                    # For now filter the Predicted total Power Used > 0
                    simulation_df_joined = simulation_df_joined[simulation_df_joined['Predicted Total Power Used (kW)']> 0]
                    
                    actual_power = sum(simulation_df_joined['Total Power Supply (kW)'])
                    predicted_power = sum(simulation_df_joined['Predicted Total Power Used (kW)'])
                    power_saved = actual_power - predicted_power


                    # Display results in columns
                    st.subheader("Actual vs Simulation Comparison")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label=f"Actual Power Supply ({month} {year_input})", value=f"{actual_power:.2f} kW")
                    with col2:
                        st.metric(label=f"Predicted Power Supply ({month} {year_input})", value=f"{predicted_power:.2f} kW")

                    estimated_savings = power_saved*1.523*15/60

                    # Conditional Formatting for power saved
                    if power_saved > 0:
                        st.subheader("Power Savings")
                        st.markdown(f"<h5 style='color: green;'>Amount of Power Saved: {power_saved:.2f} kW</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5 style='color: green;'>Percentage of Power Saved: {(power_saved/actual_power)*100:.2f}%</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5 style='color: green; '>Estimated Savings: HK$ {estimated_savings:.2f}</h5>", unsafe_allow_html=True)
                    else:
                        st.subheader("Power Wasted")
                        st.markdown(f"<h5 style='color: red;'>Additional of Power Used: {-power_saved:.2f} kW</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5 style='color: red;'>Percentage of Additional Power Used: {(-power_saved/actual_power)*100:.2f}%</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5 style='color: red;'>Estimated Additional Costs: HK$ {-estimated_savings:.2f}</h5>", unsafe_allow_html=True)

                # Visualize the plot (line graph)            
                # a Plotly figure
                with st.container(border=True):

                    fig = go.Figure()

                    # Actual power used line
                    fig.add_trace(go.Scatter(
                        x=simulation_df_joined['record_timestamp'],
                        y=simulation_df_joined['Total Power Supply (kW)'],
                        mode='lines',
                        name='Actual Total Power Used',
                        line=dict(color='red')
                    ))

                    # Predicted power used line
                    fig.add_trace(go.Scatter(
                        x=simulation_df_joined['record_timestamp'],
                        y=simulation_df_joined['Predicted Total Power Used (kW)'],
                        mode='lines',
                        name='Predicted Total Power Used',
                        line=dict(color='blue')
                    ))

                    # Update layout
                    fig.update_layout(
                        title=f'Actual Power Used VS Strategised Power Used (kW) for {year_input} {month}',
                        xaxis_title='Date',
                        yaxis_title='Power (kW)',
                        legend_title='Legend',
                        xaxis=dict(showgrid=True),
                        yaxis=dict(showgrid=True),
                        template='plotly_white',
                        xaxis_tickangle=-45  # Rotate x-axis labels for better readability
                    )

                    # Display the Plotly chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                with st.container(border=True):
                    # Display the dataframe
                    st.write("This is the operational dataframe of the simulated data")
                    simulation_df_joined.drop(columns=['hour', 'dayofweek', 'quarter', 'month', 'dayofyear'], inplace=True)

                    # Split the the column "Strategy Tag"
                    simulation_df_joined[["Main_Strategy", "Details"]] = simulation_df_joined['Strategy Tag'].str.split(": ", expand=True)

                    simulation_df_joined = simulation_df_joined[new_order]

                    st.dataframe(simulation_df_joined, hide_index=True)
                    # st.write(simulation_df_joined.columns)
                    
                    strategy_stats, most_popular, least_popular, most_cost_saving, least_cost_saving = get_strategy_summary(simulation_df_joined)
                    
                with st.container(border=True):

                    st.write("This is the strategy summary of the simulated data")
                    st.dataframe(strategy_stats, hide_index=True)
                
                with st.container(border=True):
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
                    
                    # Print the most used strategy and most cost saving strategy
                    col3, col4 = st.columns(2)
                    with col3:
                        st.markdown(f"<h5 style='color: green;'><strong>Most Used Strategy ({month} {year_input})</strong>: {most_popular['Main_Strategy']} {most_popular['Details']}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5><strong>Average Power Used</strong>: {most_popular['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                        # st.metric(label=f"Most Used Strategy ({month} {year_input})", value=f"{most_popular['Main_Strategy']} {most_popular['Details']}")
                        # st.metric(label=f"Average Power Used", value=f"{most_popular['avg_power']:.2f} kW")
                    # st.markdown("""---""")
                    with col4:
                        st.markdown(f"<h5 style='color: green;'><strong>Most Cost Saving Strategy ({month} {year_input})</strong>: {most_cost_saving['Main_Strategy']} {most_cost_saving['Details']}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5><strong>Average Power Used</strong>: {most_cost_saving['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)
                        # st.metric(label=f"Most Cost Saving Strategy ({month} {year_input})", value=f"{most_cost_saving['Main_Strategy']}  {most_cost_saving['Details']}")
                        # st.metric(label=f"Average Power Used", value=f"{most_cost_saving['avg_power']:.2f} kW")
                    
                    st.markdown("""---""")
                    
                    # Print the least used strategy and the least cost saving strategy
                    col5, col6 = st.columns(2)
                    with col5:
                        st.markdown(f"<h5 style='color: red;'><strong>Least Used Strategy ({month} {year_input})</strong>: {least_popular['Main_Strategy']} {least_popular['Details']}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5><strong>Average Power Used</strong>: {least_popular['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                        # st.metric(label=f"Least Used Strategy ({month} {year_input})", value=f"{least_popular['Main_Strategy']} {least_popular['Details']}")
                        # st.metric(label=f"Average Power Used", value=f"{least_popular['avg_power']:.2f} kW")
                    # st.markdown("""---""")

                    with col6:
                        st.markdown(f"<h5 style='color: red;'><strong>Least Cost Saving Strategy ({month} {year_input})</strong>: {least_cost_saving['Main_Strategy']}: {least_cost_saving['Details']}</h5>", unsafe_allow_html=True)
                        st.markdown(f"<h5><strong>Average Power Used</strong>: {least_cost_saving['avg_power']:.2f} kW</h5>", unsafe_allow_html=True)

                        # st.metric(label=f"Least Cost Saving Strategy ({month} {year_input})", value=f"{least_cost_saving['Main_Strategy']}: {least_cost_saving['Details']}")
                        # st.metric(label=f"Average Power Used", value=f"{least_cost_saving['avg_power']:.2f} kW")