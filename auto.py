from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas as pd
import os 
from sklearn.utils.multiclass import type_of_target

# Try to import ydata_profiling with error handling
try:
    from ydata_profiling import ProfileReport
    from streamlit_pandas_profiling import st_profile_report
    PROFILING_AVAILABLE = True
except ImportError as e:
    st.error(f"ydata_profiling not available: {e}")
    PROFILING_AVAILABLE = False

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    if PROFILING_AVAILABLE:
        try:
            profile = ProfileReport(df, title="Profiling Report", explorative=True)
            st_profile_report(profile)
        except Exception as e:
            st.error(f"Error generating profile report: {e}")
            st.info("Showing basic dataset information instead:")
            st.write(f"Dataset shape: {df.shape}")
            st.write("Dataset columns:", list(df.columns))
            st.write("Missing values:", df.isnull().sum().to_dict())
    else:
        st.error("Profiling feature is not available due to missing dependencies.")
        st.info("Showing basic dataset information instead:")
        st.write(f"Dataset shape: {df.shape}")
        st.write("Dataset columns:", list(df.columns))
        st.write("Missing values:", df.isnull().sum().to_dict())

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")