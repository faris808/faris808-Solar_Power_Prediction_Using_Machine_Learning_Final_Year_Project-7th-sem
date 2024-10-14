import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import warnings
import datetime as dt
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# Load the dataset from your local directory
gen_1 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-1/Plant_1_Generation_Data.csv')
gen_1.drop('PLANT_ID', axis=1, inplace=True)
sens_1 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-1/Plant_1_Weather_Sensor_Data.csv')
sens_1.drop('PLANT_ID', axis=1, inplace=True)

# Format datetime
gen_1['DATE_TIME'] = pd.to_datetime(gen_1['DATE_TIME'], format='%d-%m-%Y %H:%M')
sens_1['DATE_TIME'] = pd.to_datetime(sens_1['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')


import pandas as pd
from pandas.tseries.offsets import DateOffset
from pmdarima.arima import auto_arima
from statsmodels.tsa.stattools import adfuller

pred_gen=gen_1.copy()
pred_gen=pred_gen.groupby('DATE_TIME').sum()
pred_gen=pred_gen['DAILY_YIELD'][-288:].reset_index()
pred_gen.set_index('DATE_TIME',inplace=True)
print(pred_gen.head())
# pred_gen.to_csv('pred_gen_data.csv', index=True)

result = adfuller(pred_gen['DAILY_YIELD'])
print('Augmented Dickey-Fuller Test:')
labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

for value,label in zip(result,labels):
    print(label+' : '+str(value) )
    
if result[1] <= 0.05:
    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
else:
    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    
train=pred_gen[:192]
test=pred_gen[-96:]
plt.figure(figsize=(15,5))
plt.plot(train,label='Train',color='navy')
plt.plot(test,label='Test',color='darkorange')
plt.title('Last 4 days of daily yield',fontsize=17)
plt.legend()
plt.show()

import joblib
# arima_model = auto_arima(train,
#                          start_p=0,d=1,start_q=0,
#                          max_p=4,max_d=4,max_q=4,
#                          start_P=0,D=1,start_Q=0,
#                          max_P=1,max_D=1,max_Q=1,m=96,
#                          seasonal=True,
#                          error_action='warn',trace=True,
#                          supress_warning=True,stepwise=True,
#                          random_state=20,n_fits=1)

# joblib.dump(arima_model, 'arima_model.pkl')

arima_model = joblib.load('arima_model.pkl')

future_dates = [test.index[-1] + DateOffset(minutes=x) for x in range(0, 2910, 15)]

prediction = pd.DataFrame(arima_model.predict(n_periods=96), index=test.index)
prediction.columns = ['predicted_yield']

f_prediction = pd.DataFrame(arima_model.predict(n_periods=194), index=future_dates)
f_prediction.columns = ['predicted_yield']

# Plotting
fig, ax = plt.subplots(ncols=2, dpi=100, figsize=(17, 5))
ax[0].plot(train, label='Train', color='navy')
ax[0].plot(test, label='Test', color='darkorange')
ax[0].plot(prediction, label='Prediction', color='green')
ax[0].legend()
ax[0].set_title('Forecast on test set', size=17)
ax[0].set_ylabel('kW', color='navy', fontsize=17)

ax[1].plot(pred_gen, label='Original data', color='navy')
ax[1].plot(f_prediction, label='Next days forecast', color='green')
ax[1].legend()
ax[1].set_title('Next days forecast', size=17)
plt.show()



# Check for NaN values
print(test.isna().sum())
print(prediction.isna().sum())

# Drop or fill NaN values
test = test.dropna()
prediction = prediction.dropna()

# Ensure the indices are aligned
if not test.index.equals(prediction.index):
    aligned_indices = test.index.intersection(prediction.index)
    test = test.loc[aligned_indices]
    prediction = prediction.loc[aligned_indices]
    
    
# Metrics
print('SARIMAX R2 Score: %f' % r2_score(test['DAILY_YIELD'], prediction['predicted_yield']))
print('-' * 15)
print('SARIMAX MAE Score: %f' % mean_absolute_error(test['DAILY_YIELD'], prediction['predicted_yield']))
print('SARIMAX RMSE Score: %f' % mean_squared_error(test['DAILY_YIELD'], prediction['predicted_yield'], squared=False))

print(arima_model.summary())


# Combine the true and predicted values into a DataFrame for side-by-side comparison
comparison_df = pd.DataFrame({
    'Exact Values (Test)': test['DAILY_YIELD'],
    'Predicted Values': prediction['predicted_yield']
})

# Display the comparison DataFrame
print("Comaprison is:-",comparison_df.head(30))  # Display the first 10 rows for comparison

comparison_df.to_csv('comparison_results_arima_model.csv', index=False)





print('SARIMAX R2 Score: %f' % (r2_score(prediction['predicted_yield'],test['DAILY_YIELD'])))
print('SARIMAX MAE Score: %f' % (mean_absolute_error(prediction['predicted_yield'],test['DAILY_YIELD'])))
print('SARIMAX RMSE Score: %f' % (mean_squared_error(prediction['predicted_yield'],test['DAILY_YIELD'],squared=False)))
