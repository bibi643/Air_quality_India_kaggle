

# Libraries

import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import ExponentialSmoothing, Prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation,performance_metrics
from sklearn.metrics import mean_absolute_error
import streamlit as st
# add streamlit

st.title('Air Quality in India')

# Import the datas
df= pd.read_csv('archive/DL001.csv',parse_dates=['From Date'],index_col=['From Date'])
df=df.drop(labels='To Date',axis=1) # we drop one date column


df.head()

add_selectbox = st.sidebar.selectbox(
    
    "Structure",
    ("Data Visualisation", "Resampling", "Predictions")

)
# List of the column we can eventually forecast


# Data Viz

if add_selectbox =='Data Visualisation':
    columns=st.selectbox()
    if columns:
        df.columns




