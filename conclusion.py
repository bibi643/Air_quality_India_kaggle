
#Library

import streamlit as st

def concl():
    st.title('Conclusion')

    con_rad= st.sidebar.radio('Objectives',['Data Vis','Time Series Models','Soft skills'])
    if con_rad =='Data Vis':
        st.header('Data Vis')
        st.markdown(
            """
- It has been plotted different groups of molecules grouping those in different categories.
- Datas have been grouped monthly, weekly to eliminate missing datas.

"""
        )

    elif con_rad =='Time Series Models':
        st.header('Time Series Models')
        st.markdown(
            """
- Carbon monoxyde have been selected as the target to predict.
- Different Models have been tried to predict CO values, such as Exponential Smoothing, Prophet and ARIMA.
- ARIMA has the best performance.

"""
        )

    else:
        st.header('Soft skills')
        st.markdown(
            """
            This project has been useful to:
- work on time series project and datas.
- present the result in a streamlit form.
- keep using git and github tools.

"""
        )
       
