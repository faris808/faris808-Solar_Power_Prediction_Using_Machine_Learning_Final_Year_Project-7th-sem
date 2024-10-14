# #Prophet model training and analysis
# from prophet import Prophet
# from pandas.tseries.offsets import DateOffset
# import matplotlib.pyplot as plt
# import pandas as pd


# # Load the dataset from your local directory
# gen_1 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-1/Plant_1_Generation_Data.csv')
# gen_1.drop('PLANT_ID', axis=1, inplace=True)
# sens_1 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-1/Plant_1_Weather_Sensor_Data.csv')
# sens_1.drop('PLANT_ID', axis=1, inplace=True)

# # Format datetime
# gen_1['DATE_TIME'] = pd.to_datetime(gen_1['DATE_TIME'], format='%d-%m-%Y %H:%M')
# sens_1['DATE_TIME'] = pd.to_datetime(sens_1['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')



# pred_gen=gen_1.copy()
# pred_gen=pred_gen.groupby('DATE_TIME').sum()
# pred_gen=pred_gen['DAILY_YIELD'][-288:].reset_index()
# pred_gen.set_index('DATE_TIME',inplace=True)
# test=pred_gen[-96:]
# test = test.dropna()

# # Preprocessing data for Prophet
# pred_gen2 = gen_1.copy()
# pred_gen2 = pred_gen2.groupby('DATE_TIME')['DAILY_YIELD'].sum().reset_index()
# pred_gen2.rename(columns={'DATE_TIME': 'ds', 'DAILY_YIELD': 'y'}, inplace=True)

# # pred_gen2.to_csv('pred_gen_data2.csv', index=True)

# # Plotting actual DAILY_YIELD
# pred_gen2.plot(x='ds', y='y', figsize=(17, 5))
# plt.legend('')
# plt.title('DAILY_YIELD', size=17)
# plt.show()

# # Initialize and fit Prophet model
# m = Prophet()
# m.fit(pred_gen2)

# # Creating future DataFrame for prediction
# future = [pred_gen2['ds'].iloc[-1:] + DateOffset(minutes=x) for x in range(0, 2910, 15)]
# time1 = pd.DataFrame(future).reset_index().drop('index', axis=1)
# time1.rename(columns={3157: 'ds'}, inplace=True)

# timeline = pd.DataFrame(pred_gen2['ds'])
# fut = pd.concat([timeline, time1], ignore_index=True)

# # Forecasting using the fitted model
# forecast = m.predict(fut)

# # Plotting the forecasted results
# m.plot(forecast, figsize=(15, 7))
# plt.title('Prophet Forecast')
# plt.legend(labels=['Original data', 'Prophet Forecast'])
# plt.show()

# from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
# test2=pd.DataFrame(test.index)
# test2.rename(columns={'DATE_TIME':'ds'},inplace=True)
# test_prophet=m.predict(test2)

# # Creating a comparison table between actual and predicted values
# # Merging the actual and predicted data on the 'ds' column (time)
# comparison_df = pd.DataFrame({
#     'Actual': pred_gen2['y'],
#     'Predicted': test_prophet['yhat'][:len(pred_gen2)]  # Align with actual data length
# })

# # Displaying the first 10 rows of the comparison table
# print("Comparison of Actual and Predicted Values:")
# print(comparison_df.head(60))

# comparison_df.to_csv('comparison_results_prophet_model.csv', index=False)


# print('Prophet R2 Score: %f' % (r2_score(test['DAILY_YIELD'],test_prophet['yhat'])))
# print('Prophet MAE Score: %f' % (mean_absolute_error(test['DAILY_YIELD'],test_prophet['yhat'])))
# print('Prophet RMSE Score: %f' % (mean_squared_error(test['DAILY_YIELD'],test_prophet['yhat'],squared=False)))



# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
gen_1 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-1/Plant_1_Generation_Data.csv')
gen_1.drop('PLANT_ID', axis=1, inplace=True)

# Format datetime
gen_1['DATE_TIME'] = pd.to_datetime(gen_1['DATE_TIME'], format='%d-%m-%Y %H:%M')

# Step 2: Exploratory Data Analysis (EDA)
# Group by DATE_TIME and sum the DAILY_YIELD
data = gen_1.groupby('DATE_TIME')['DAILY_YIELD'].sum().reset_index()

# Visualize the data
plt.figure(figsize=(12, 6))
plt.plot(data['DATE_TIME'], data['DAILY_YIELD'], label='Daily Yield', color='blue')
plt.title('Daily Solar Energy Generation')
plt.xlabel('Date')
plt.ylabel('Daily Yield (kWh)')
plt.legend()
plt.show()

# Check for missing values
print(data.isnull().sum())

# Step 3: Split the dataset into training, validation, and testing sets
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)
test_size = len(data) - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# Step 4: Prepare data for Prophet
train_data.rename(columns={'DATE_TIME': 'ds', 'DAILY_YIELD': 'y'}, inplace=True)

# Initialize and fit Prophet model
model = Prophet()
model.fit(train_data)

# Step 5: Validate the model
# Create a DataFrame for validation predictions
future_val = val_data['DATE_TIME'].reset_index(drop=True)
future_val = pd.DataFrame({'ds': future_val})

# Predicting validation data
val_forecast = model.predict(future_val)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(train_data['ds'], train_data['y'], label='Train Data', color='blue')
plt.plot(val_forecast['ds'], val_forecast['yhat'], label='Validation Predictions', color='orange')
plt.scatter(val_data['DATE_TIME'], val_data['DAILY_YIELD'], color='red', label='Actual Validation Data', marker='o')
plt.title('Validation Predictions vs Actual Data')
plt.xlabel('Date')
plt.ylabel('Daily Yield (kWh)')
plt.legend()
plt.show()

# Step 6: Evaluate validation performance
val_metrics = {
    'R2 Score': r2_score(val_data['DAILY_YIELD'], val_forecast['yhat'][:val_data.shape[0]]),
    'MAE': mean_absolute_error(val_data['DAILY_YIELD'], val_forecast['yhat'][:val_data.shape[0]]),
    'RMSE': mean_squared_error(val_data['DAILY_YIELD'], val_forecast['yhat'][:val_data.shape[0]], squared=False)
}

print("Validation Metrics:")
for key, value in val_metrics.items():
    print(f"{key}: {value}")

# Step 7: Testing the model
# Prepare test data
test_data.rename(columns={'DATE_TIME': 'ds', 'DAILY_YIELD': 'y'}, inplace=True)
future_test = test_data['ds'].reset_index(drop=True)
future_test = pd.DataFrame({'ds': future_test})

# Predicting test data
test_forecast = model.predict(future_test)

# Plotting test results
plt.figure(figsize=(12, 6))
plt.plot(train_data['ds'], train_data['y'], label='Train Data', color='blue')
plt.plot(val_forecast['ds'], val_forecast['yhat'], label='Validation Predictions', color='orange')
plt.scatter(test_data['ds'], test_data['y'], color='red', label='Actual Test Data', marker='o')
plt.plot(test_forecast['ds'], test_forecast['yhat'], label='Test Predictions', color='green')
plt.title('Test Predictions vs Actual Data')
plt.xlabel('Date')
plt.ylabel('Daily Yield (kWh)')
plt.legend()
plt.show()

# Evaluate test performance
test_metrics = {
    'R2 Score': r2_score(test_data['y'], test_forecast['yhat']),
    'MAE': mean_absolute_error(test_data['y'], test_forecast['yhat']),
    'RMSE': mean_squared_error(test_data['y'], test_forecast['yhat'], squared=False)
}

print("Test Metrics:")
for key, value in test_metrics.items():
    print(f"{key}: {value}")

# Save the comparison results to CSV
# Creating a DataFrame for comparison with aligned actual and predicted values
comparison_results = pd.DataFrame({
    'Date': test_forecast['ds'],
    'Actual': test_data['y'].values,  # Ensure the actual values align with the correct dates
    'Predicted': test_forecast['yhat']
})

comparison_results.to_csv('comparison_results_prophet_model.csv', index=False)

print("Model training and evaluation completed.")


