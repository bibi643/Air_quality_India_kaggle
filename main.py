

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
from explore_viz import exp_viz
from models import ts_pred
from conclusion import concl


# run in terminal streamlit run main.py to open in the browzer#
#sous fichiers

#from prophet_v2 import prophet_func
#from darts_lastV import darts_graph

def main():
    #liste of pages
    liste_menu=['Data Exploration/Visualisation','Time Series Predictions','Conclusion']
    


    #side bar
    menu=st.sidebar.selectbox('Select your page',liste_menu)
    
    #Page navigation
    if menu==liste_menu[0]:
        exp_viz()
    
    if menu == liste_menu[1]:
        ts_pred()

    if menu== liste_menu[2]:
        concl()

   
        
   


if __name__=='__main__':
    main()







