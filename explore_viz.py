


#Libraries
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def exp_viz():
    #sidebar
    data_to_observe=st.sidebar.selectbox('Data to observe',['Particule matters','N molecules','Carbon molecules'])
    resampling_duration= st.sidebar.radio('Resampling Selection',['Daily','Weekly','Monthly'])
    


    st.header('Data Visualization')
    st.subheader('Data Resampling')
    st.write('The reader can select different paramaters such as the duration of the resampling window and observe the number of lines of the data frame he/she wish.')
    df=pd.read_csv('archive/DL001.csv',parse_dates=['From Date'],index_col=['From Date'])
    df=df.drop(labels='To Date',axis=1)
    df= df['2018-08':]
   
    if resampling_duration=='Daily':
        df=df.resample('D',label='right',closed='right').mean().dropna()
    elif resampling_duration=='Weekly':
        df=df.resample('W',label='right',closed='right').mean().dropna()
    else:
        df=df.resample('ME',label='right',closed='right').mean().dropna()
    #Resampling daily, weekly, monthly from 2018
   

    part_matters= df.filter(regex='^PM')
    n_molecules= df.filter(regex='^N')
    carbon_o=df.filter(regex='^C')

    
    if data_to_observe=='Particule matters':
        
        fig= px.line(part_matters)
        fig2=px.histogram(part_matters)
        st.plotly_chart(fig)
        st.subheader('Distribution')
        st.plotly_chart(fig2)
    elif data_to_observe=='N molecules':
        fig= px.line(n_molecules)
        fig2=px.histogram(n_molecules)
        st.plotly_chart(fig)
        st.subheader('Distribution')
        st.plotly_chart(fig2)

    else:
        fig=px.line(carbon_o)
        fig2=px.histogram(carbon_o)
        st.plotly_chart(fig)
        st.subheader('Distribution')
        st.plotly_chart(fig2)
        


    
