import streamlit as st
import time

st.set_page_config(
    page_title=" Chiller Equipment Dashboard",
    page_icon="hkbu.png",
    # page_icon="⚙️",
    layout="wide",
)

st.logo(
    image="hkbu.png"    
)

main_container = st.container(border=True, key="main_container")
with main_container:
    # dashboard title
    # st.markdown("<h1 style = 'text-align: center;'><strong>Tai Lung Veterinary Laboratory Chiller Equipment Dashboard ⚡</strong></h1>", unsafe_allow_html=True)
    st.markdown("<h1 style = 'text-align: center;'><strong>Energy Optimization in Tai Lung Veterinary Laboratory: Predictive Modelling for Chiller Equipment Efficiency</strong></h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>Developed by: ONG Jun Kye, 21201749</h4>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: black;'>Hong Kong Baptist University</h4>", unsafe_allow_html=True)

    # st.markdown("<h4 style='text-align: center; color: black;'>Supervisor: Prof. TAI, Samson Kin Hon</h4>", unsafe_allow_html=True)

content = st.empty()
with content:  
    with st.spinner("Importing necessary libraries..."):
        from Predict_Page import show_predict_page
        from Explore_Page import show_explore_page
        from Simulation import show_simulation_page
        # Clear the status text after a short delay
        time.sleep(0.5)
    
page = st.sidebar.selectbox("Page", ("Explore 🔎", "Predict 🔮", "Simulation 🕹️"))
# need to do the dashboard
if page ==  "Explore 🔎":
    show_explore_page()
elif page == "Predict 🔮":    
    show_predict_page()
else:
    show_simulation_page()
