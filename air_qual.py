

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
import plotly.figure_factory as ff
# add streamlit

st.title('Air Quality in India')

# Import the datas
df= pd.read_csv('archive/DL001.csv',parse_dates=['From Date'],index_col=['From Date'])
df=df.drop(labels='To Date',axis=1) # we drop one date column


df.head()
# Cut from 2018 to have more consistent data
df=df['2018-08':]

# Separation in different categories
part_matters= df.filter(regex='^PM')
n_molecules= df.filter(regex='^N')
carbon_o=df.filter(regex='^C')

add_selectbox = st.sidebar.selectbox(
    
    "Structure",
    ("Data Visualisation", "Resampling", "Predictions")

)
# List of the column we can eventually forecast


# Data Analasis

## Lineplot
st.line_chart(df)
## Histplot





#columns name
col_list=df.columns

col_selection= st.selectbox(
    'Which Column(s) you would like to check the distribution?',
    list(df.columns)
)
'Your choice is:', col_selection
group_labels=[col_selection]
fig= ff.create_distplot([df[col_selection]],group_labels)


# Resample, daily, weekly, monthly. give the option. To take ouy the hour.
