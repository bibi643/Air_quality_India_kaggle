


#Libraries
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from darts import TimeSeries
from darts.models import ExponentialSmoothing, Prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation,performance_metrics
from prophet.plot import plot_cross_validation_metric
from darts.metrics import mape, mase
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error,mean_squared_error



        

def ts_pred():
    #sidebar
    m=st.sidebar.radio('Model',['Exponential Smoothing','Prophet','ARIMA'])
    resampling_duration= st.sidebar.radio('Resampling Selection',['Weekly','Monthly'])
    st.header('Predictions')

    df=pd.read_csv('archive/DL001.csv',parse_dates=['From Date'],index_col=['From Date'])
    df=df.drop(labels='To Date',axis=1)
    
    df= df['CO (mg/m3)']
    df=df['2018-08':]
    
    
    if resampling_duration=='Weekly':
        df=df.resample('W',label='right',closed='right').mean().dropna()
    else:
        df=df.resample('ME',label='right',closed='right').mean().dropna()

    df=df.reset_index()
    if m=='Exponential Smoothing':
        series= TimeSeries.from_dataframe(df,time_col='From Date',fill_missing_dates=True, freq=None)
        training, val= series.split_after(0.8)
        model = ExponentialSmoothing()
        model.fit(training)
        pred=model.predict(len(val))
        fig= plt.figure()
        series.plot(label='Data')
        pred.plot(label='predictions',low_quantile=0.05, high_quantile=0.95)
        plt.legend()
        st.pyplot(fig)
        s= st.radio('Score Metrics',['MAPE','MASE'])
        if s=='MAPE':
                s=mape(val,pred)
                st.write('MAPE',s)
                st.write('The Mean Absolute Percentage Error - MAPE, measures the difference of forecast errors and divides it by the actual observation value.')
        else:
            s=mase(val,pred,training)
            st.write('MASE',s)
            st.write('An MASE = 0.5, means that our model has doubled the prediction accuracy. The lower, the better. When MASE > 1, that means the model needs a lot of improvement')
    
    elif m=='Prophet':
         df=df.rename(columns={'From Date':'ds',
                               'CO (mg/m3)':'y'})
         model=Prophet()
         model.fit(df)
         if resampling_duration=='Weekly':
            future=model.make_future_dataframe(periods=154,freq='W')
            forecast=model.predict(future)
            fig=model.plot(forecast)
            st.pyplot(fig)
            df_cv=cross_validation(model,initial='156 w',horizon='52 w')
            df_p = performance_metrics(df_cv)
            fig2= plot_cross_validation_metric(df_cv,metric='mape')
            st.pyplot(fig2)


         else:
            future=model.make_future_dataframe(periods=24,freq='M')
            forecast=model.predict(future)
            fig=model.plot(forecast)
            st.pyplot(fig)
            df_cv=cross_validation(model,initial='48 m',horizon='3 m')
            df_p = performance_metrics(df_cv)
            fig2= plot_cross_validation_metric(df_cv,metric='mape')
            st.pyplot(fig2)

    else:
        st.header('ARIMA')
        df=pd.read_csv('archive/DL001.csv',header=0,parse_dates=['From Date'],index_col=['From Date'])

        df_ts=df.drop(labels='To Date',axis=1)
        df_ts
        series_co= df_ts['CO (mg/m3)'].to_frame()
        series_co=series_co['2018-08':]
        co_w=series_co.resample('W',label='right',closed='right').mean().dropna()

        co_m=series_co.resample('ME',label='right',closed='right').mean().dropna()
        co_d= series_co.resample('3 d',label='right',closed='right').mean().dropna()
        
        if resampling_duration=='Weekly':
            st.subheader('Seasonal decompose')
            sd=seasonal_decompose(co_w,model='multiplicative')
            fig=sd.plot()
            st.pyplot(fig)
            st.write('It can be observed a seasonality of 1 year')
            co_w_log=np.log(co_w)
            co_w_log_diff=co_w_log.diff().dropna()
            co_w_log_diff_season=co_w_log_diff.diff(periods=52).dropna()
            result=adfuller(co_w_log_diff_season)
            print(f'P-Value is {result[1]}')
            if result[1]< 0.05:
                print('La serie est stationnaire')
            else:
                print('The serie is not stationnary')

            st.subheader('Autocorrelation and Partial Autocorrelation')
            fig1=plot_acf(co_w_log_diff_season,lags=104)
            fig2=plot_pacf(co_w_log_diff_season,lags=95)
            st.pyplot(fig1)
            st.pyplot(fig2)

            model_w =sm.tsa.SARIMAX(co_w_log,order=(1,1,1),seasonal_order=(0,1,0,52))

            model_w_fitted=model_w.fit()
            print(model_w_fitted.summary())

            st.write('Datas are not autocorrelated (Q>0.05) and residus are normally distributed (JB >0.05)')

            st.subheader('Predictions')
            pred_w= np.exp(model_w_fitted.predict(200,244))
            co_pred=pd.concat([co_w,pred_w])
            
            pred_fig=plt.figure()
            plt.plot(co_pred)
            st.pyplot(pred_fig)
            st.subheader('Evaluation')
            mape_w=mean_absolute_percentage_error(co_w[200:244],pred_w[:-1])
            st.write('MAPE: ',mape_w)

            st.subheader('Forecasting')
            prediction_w = model_w_fitted.get_forecast(steps =12).summary_frame()  #Prédiction avec intervalle de confiance

            fig, ax = plt.subplots(figsize = (15,5))

            plt.plot(co_w)
            #prediction = np.exp(prediction) #Passage à l'exponentielle
            prediction_w= np.exp(prediction_w)
            prediction_w['mean'].plot(ax = ax, style = 'k--') #Visualisation de la moyenne

            ax.fill_between(prediction_w.index, prediction_w['mean_ci_lower'], prediction_w['mean_ci_upper'], color='k', alpha=0.1); #Visualisation de l'intervalle de confiance

            st.pyplot(fig)


        else:
            
            st.subheader('Seasonal decompose')
            sd=seasonal_decompose(co_m,model='multiplicative')
            fig=sd.plot()
            st.pyplot(fig)
            st.write('It can be observed a seasonality of 1 year')
            co_m_log=np.log(co_m)
            co_m_log_diff=co_m_log.diff().dropna()
            co_m_log_diff_season=co_m_log_diff.diff(periods=12).dropna()
            result=adfuller(co_m_log_diff_season)
            print(f'P-Value is {result[1]}')
            if result[1]< 0.05:
                print('La serie est stationnaire')
            else:
                print('The serie is not stationnary')

            st.subheader('Autocorrelation and Partial Autocorrelation')
            fig1=plot_acf(co_m_log_diff_season,lags=36)
            fig2=plot_pacf(co_m_log_diff_season,lags=21)
            st.pyplot(fig1)
            st.pyplot(fig2)

            model_m =sm.tsa.SARIMAX(co_m_log,order=(0,1,1),seasonal_order=(0,1,0,12))

            model_m_fitted=model_m.fit()
            print(model_m_fitted.summary())

            st.write('Datas are not autocorrelated (Q>0.05) and residus are normally distributed (JB >0.05)')

            st.subheader('Predictions')
            pred_m= np.exp(model_m_fitted.predict(30,55))
            co_pred=pd.concat([co_m,pred_m])
        
            pred_fig=plt.figure()
            plt.plot(co_pred)
            st.pyplot(pred_fig)
            st.subheader('Evaluation')
            mape_m=mean_absolute_percentage_error(co_m[30:55],pred_m[:-1])
            st.write('MAPE: ',mape_m)

            st.subheader('Forecasting')
            prediction_m = model_m_fitted.get_forecast(steps =12).summary_frame()  #Prédiction avec intervalle de confiance

            fig, ax = plt.subplots(figsize = (15,5))

            plt.plot(co_m)
            #prediction = np.exp(prediction) #Passage à l'exponentielle
            prediction_m= np.exp(prediction_m)
            prediction_m['mean'].plot(ax = ax, style = 'k--') #Visualisation de la moyenne

            ax.fill_between(prediction_m.index, prediction_m['mean_ci_lower'], prediction_m['mean_ci_upper'], color='k', alpha=0.1); #Visualisation de l'intervalle de confiance

            st.pyplot(fig)

                        

            
            
            
            
            
            