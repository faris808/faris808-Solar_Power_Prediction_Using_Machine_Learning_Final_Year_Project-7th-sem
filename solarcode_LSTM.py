# Importing various libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Importing the datasets
gen_1 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-1/Plant_1_Generation_Data.csv')
sens_1 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-1/Plant_1_Weather_Sensor_Data.csv')
gen_2 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-2/Plant_2_Generation_Data.csv')
sens_2 = pd.read_csv('C:/Users/Md Faris/Desktop/ML Project/Development-work/Plant-2/Plant_2_Weather_Sensor_Data.csv')


gen_1.info()
sens_1.info()
gen_1.describe()
sens_1.describe()

gen_1.isna().sum()
sens_1.isna().sum()


gen_1['PLANT_ID'].nunique()
sens_1['PLANT_ID'].nunique()
#we dont have any missing values and we have also checked that data is from one plant as expected. 

gen_1['SOURCE_KEY'].value_counts()
print('There are {} different inverters. Number of measurements per inverter range from {} to {}.' .format(gen_1.SOURCE_KEY.nunique(),gen_1.SOURCE_KEY.value_counts().min(), gen_1.SOURCE_KEY.value_counts().max() ))
#There are 22 different inverters.


df_gen1 = gen_1.copy()
df_sens1 = sens_1.copy()
df_gen2 = gen_2.copy()
df_sens2 = sens_2.copy()

#Here we will combine the two datasets. To achieve this, we will standardize the DATE_TIME formats, eliminate redundant columns, and merge the datasets based on the DATE_TIME column. Additionally, we will extract separate date and time columns and assign numerical labels from 1 to 22 to the inverters.

df_gen1['DATE_TIME'] = pd.to_datetime(df_gen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_sens1['DATE_TIME'] = pd.to_datetime(df_sens1['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
df_gen2['DATE_TIME'] = pd.to_datetime(df_gen2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S') 
df_sens2['DATE_TIME'] = pd.to_datetime(df_sens2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

# Dropping the unnecessary columns and merge both the dataframes along DATE_TIME
df_plant1 = pd.merge(df_gen1.drop(columns = ['PLANT_ID']), df_sens1.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

df_plant1.head()

# Adding inverter number column to the dataframe
sensor_keys = df_plant1['SOURCE_KEY'].unique()
sensor_numbers = dict(zip(sensor_keys, range(1, len(sensor_keys) + 1)))

# Add SENSOR_NUM column using map
df_plant1['SENSOR_NUM'] = df_plant1['SOURCE_KEY'].map(sensor_numbers)

# Add SENSOR_NAME column
df_plant1['SENSOR_NAME'] = df_plant1['SENSOR_NUM'].astype(str)

# Convert DATE_TIME to datetime if it's not already
df_plant1['DATE_TIME'] = pd.to_datetime(df_plant1['DATE_TIME'])

# Adding separate date and time columns
df_plant1['DATE'] = df_plant1['DATE_TIME'].dt.date
df_plant1['TIME'] = df_plant1['DATE_TIME'].dt.time

# Add hours and minutes for ML models
df_plant1['HOURS'] = df_plant1['DATE_TIME'].dt.hour
df_plant1['MINUTES'] = df_plant1['DATE_TIME'].dt.minute
df_plant1['MINUTES_PASS'] = df_plant1['HOURS'] * 60 + df_plant1['MINUTES']

# Add date as string column
df_plant1['DATE_STR'] = df_plant1['DATE'].astype(str)

df_plant1.head(50)

numeric_columns = df_plant1.select_dtypes(include=np.number).columns.tolist()
corr_df = df_plant1[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

df = gen_1.copy()
df.head()

# Step 1: Convert 'DATE_TIME' to a datetime object
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], format='%d-%m-%Y %H:%M')

# Step 2: Set 'DATE_TIME' as the index
df.set_index('DATE_TIME', inplace=True)

# output_file = 'final_dataset1.csv'
# df.to_csv(output_file, index=False)

# Step 3: Resample the data into 15-minute intervals and sum the values over each interval
df_resampled = df.resample('15T').sum()


df_resampled['DAILY_YIELD_MW'] = df_resampled['DAILY_YIELD']/1000
print("Our acutal dataset is:")
print(df_resampled.head())

# output_file = 'final_dataset2.csv'
# df_resampled.to_csv(output_file, index=False)

daily_yield = df_resampled['DAILY_YIELD_MW']
print(daily_yield.head(20))
print(daily_yield.shape)


# Making a function to execute above idea

def df_to_X_y(df, window_size=5):  # 5 here means we are taking the last 5 values
    df_as_np = df.to_numpy() #converting the dataframe to numpy arrays
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)

WINDOW_SIZE = 5
X, y = df_to_X_y(daily_yield, WINDOW_SIZE)
X.shape, y.shape


# making a train and test dataset to train the model

X_train, y_train  = X[:2600], y[:2600]
X_val, y_val  = X[2600:2900], y[2600:2900]
X_test, y_test  = X[2900:], y[2900:]

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)


#  Tensorflow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, Dropout

# Defining the model and adding layers into the LSTM model
model1 = Sequential()

# Assuming input shape of (5, 1) for 60 minutes of daily yield data
model1.add(InputLayer(input_shape=(5, 1)))  # 5 time steps, 1 feature

# LSTM layer with 32 units, reduced from 64 to prevent overfitting
model1.add(LSTM(32))

# Adding a Dense layer with 8 units and ReLU activation
model1.add(Dense(8, activation='relu'))

# Adding Dropout for regularization (optional, 0.2 means 20% dropout rate)
model1.add(Dropout(0.2))

# Output layer for a single value prediction with linear activation
model1.add(Dense(1, activation='linear'))

# Compile the model
model1.compile(optimizer='adam', loss='mse')

# Model summary to check the architecture
model1.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

# Define the model checkpoint callback
# Save the best model based on validation loss (lowest RMSE)
cp = ModelCheckpoint(
    filepath='model1/best_model.keras',  # Use .h5 extension for model saving
    monitor='val_root_mean_squared_error',  # Monitor validation RMSE
    save_best_only=True,
    save_weights_only=False,  # Save full model instead of just weights
    verbose=1  # Print out saving messages
)

# Add early stopping to stop training if the model stops improving
es = EarlyStopping(
    monitor='val_root_mean_squared_error',  # Stop if RMSE doesn't improve
    patience=10,  # Wait 10 epochs before stopping
    restore_best_weights=True,  # Restore the best weights after stopping
    verbose=1
)

# Compile the model
# Lower learning rate is set to help the optimizer converge more smoothly
model1.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=1e-4),  # 0.0001 is written as 1e-4 for clarity
    metrics=[RootMeanSquaredError()]  # RMSE metric for model evaluation
)

# Fit the model with better structure, additional callbacks, and dynamic batch size
model1.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),  # Provide validation data for real-time evaluation
    epochs=75,  # Increase epochs for a more robust model, can be adjusted later
    batch_size=32,  # Use a moderate batch size (adjust based on your hardware)
    callbacks=[cp, es],  # Include EarlyStopping for better convergence
    verbose=1  # Show training progress
)


from tensorflow.keras.models import load_model

# Load the best saved model
model1 = load_model('model1/best_model.keras')  # Use the correct file extension and path

# Print model summary to confirm successful loading
model1.summary()


# Make predictions on the training data
train_predictions = model1.predict(X_train).flatten()

# Create a DataFrame to compare predictions and actual values
train_results = pd.DataFrame({
    'Train Predictions': train_predictions,
    'Actual Values': y_train.flatten()  # Ensure y_train is also flattened for consistency
})

# Optionally display the first few rows of the DataFrame to verify
print(train_results.head())

import matplotlib.pyplot as plt

# Set plot size for better visibility
plt.figure(figsize=(10, 6))

# Plot Train Predictions and Actual Values
plt.plot(train_results['Train Predictions'][50:100], label='Train Predictions', color='blue', linestyle='--', marker='o')
plt.plot(train_results['Actual Values'][50:100], label='Actual Values', color='green', linestyle='-', marker='x')

# Add labels, title, and grid
plt.title('Comparison of Train Predictions vs Actual Values', fontsize=16)
plt.xlabel('Time Steps (Sample Index)', fontsize=12)
plt.ylabel('Daily Yield', fontsize=12)
plt.grid(True)

# Add a legend to differentiate the lines
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# Make predictions on the validation data
val_predictions = model1.predict(X_val).flatten()

# Create a DataFrame to compare predictions and actual values
val_results = pd.DataFrame({
    'Val Predictions': val_predictions,
    'Actual Values': y_val.flatten()  # Ensure y_val is flattened for consistency
})

# Display the DataFrame to check predictions vs actuals
print(val_results.head())


import matplotlib.pyplot as plt

# Set plot size for better visibility
plt.figure(figsize=(10, 6))

# Plot Validation Predictions and Actual Values
plt.plot(val_results['Val Predictions'][:50], label='Validation Predictions', color='blue', linestyle='--', marker='o')
plt.plot(val_results['Actual Values'][:50], label='Actual Values', color='orange', linestyle='-', marker='x')

# Add labels, title, and grid
plt.title('Comparison of Validation Predictions vs Actual Values', fontsize=16)
plt.xlabel('Time Steps (Sample Index)', fontsize=12)
plt.ylabel('Daily Yield', fontsize=12)
plt.grid(True)

# Add a legend to differentiate the lines
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# Make predictions on the test data
test_predictions_1 = model1.predict(X_test).flatten()

# Create a DataFrame to compare predictions and actual values
test_results_1 = pd.DataFrame({
    'Test Predictions': test_predictions_1,
    'Actual Values': y_test.flatten()  # Ensure y_test is flattened for consistency
})

# Display the first few rows of the DataFrame
print(test_results_1.head())

import matplotlib.pyplot as plt

# Set plot size for better visibility
plt.figure(figsize=(10, 6))

# Plot Test Predictions and Actual Values
plt.plot(test_results_1['Test Predictions'][:50], label='Test Predictions', color='blue', linestyle='--', marker='o')
plt.plot(test_results_1['Actual Values'][:50], label='Actual Values', color='red', linestyle='-', marker='x')

# Add labels, title, and grid
plt.title('Comparison of Test Predictions vs Actual Values', fontsize=16)
plt.xlabel('Time Steps (Sample Index)', fontsize=12)
plt.ylabel('Daily Yield', fontsize=12)
plt.grid(True)

# Add a legend to differentiate the lines
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

def evaluate_model_performance(y_true, y_pred):
    """
    Evaluates model performance by computing key metrics and generating visualizations.

    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values

    Returns:
    dict: A dictionary containing calculated metrics (MSE, RMSE, MAE, R²)
    """

    # Flattening inputs if they are not 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Calculating key metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Printing metrics
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Residuals calculation
    residuals = y_true - y_pred

    # Plotting residuals distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.title('Residuals Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')

    # Plotting Predictions vs Actual Values
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.title('Predictions vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')  # Ideal line

    plt.tight_layout()
    plt.show()

    # Plotting actual vs predicted over time (for time series or sequential data)
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual Values')
    plt.plot(y_pred, label='Predicted Values', alpha=0.7)
    plt.title('Actual vs Predicted Values Over Time')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    # Returning a dictionary of computed metrics
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

    return metrics

train_metrics_model1 = evaluate_model_performance(y_test, test_predictions_1)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define Model
model2 = Sequential()

# Stacked LSTM Layers with Dropout
model2.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model2.add(Dropout(0.2))
model2.add(LSTM(64, return_sequences=False))
model2.add(Dropout(0.2))

# Dense Layers
model2.add(Dense(32, activation='relu'))
model2.add(Dense(1, activation='linear'))

# Compile the model
model2.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['RootMeanSquaredError'])

# Set callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('model2_best.keras', save_best_only=True, monitor='val_loss')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Fit the model
history = model2.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=100,
                     batch_size=64,
                     callbacks=[early_stopping, checkpoint, lr_scheduler])

# Load the best model weights
model2.load_weights('model2_best.keras')

# Print model summary to confirm successful loading
model2.summary()


# Make predictions on the test data
test_predictions_2 = model2.predict(X_test).flatten()

# Create a DataFrame to compare predictions and actual values
test_results_2 = pd.DataFrame({
    'Test Predictions': test_predictions_2,
    'Actual Values': y_test.flatten()  # Ensure y_test is flattened for consistency
})

# Display the first few rows of the DataFrame
print(test_results_2.head(40))
# test_results_2.to_csv('comparison_results_lstm_model.csv', index=False)


import matplotlib.pyplot as plt

# Set plot size for better visibility
plt.figure(figsize=(10, 6))

# Plot Test Predictions and Actual Values
plt.plot(test_results_2['Test Predictions'][:50], label='Test Predictions', color='blue', linestyle='--', marker='o')
plt.plot(test_results_2['Actual Values'][:50], label='Actual Values', color='red', linestyle='-', marker='x')

# Add labels, title, and grid
plt.title('Comparison of Test Predictions vs Actual Values', fontsize=16)
plt.xlabel('Time Steps (Sample Index)', fontsize=12)
plt.ylabel('Daily Yield', fontsize=12)
plt.grid(True)

# Add a legend to differentiate the lines
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

train_metrics_model2 = evaluate_model_performance(y_test, test_predictions_2)







# Import libraries (already included in your code)
# ...

# Load the best performing model (confirmed to be model2)
model = load_model('model2_best.keras')

# Get the last timestep from the test data (assuming your test data is available)
# Get the last timestep from the test data (assuming your test data is available)
last_timestep = X_test[-1]

# Print the shape of the last timestep for verification
print(f"Shape of the last timestep: {last_timestep.shape}")

# Define the number of days to predict (3 days in this case)
num_days_to_predict = 3

# Function to predict for the next day
def predict_next_day(last_timestep):
  predictions = []
  for _ in range(num_days_to_predict * 48):  # 48 timesteps per day (assuming 15-minute intervals)
    # Reshape the last_timestep to include a batch dimension (of size 1)
    last_timestep = last_timestep.reshape(1, *last_timestep.shape)  # Reshape to add batch dimension
    prediction = model.predict(last_timestep)[0][0]
    predictions.append(prediction)
    last_timestep = np.append(last_timestep[1:], prediction)  # Shift the timestep and append the prediction
  return predictions

# Try-except block to handle potential errors during prediction
try:
  # Predict yield for the next 3 days
  next_3_days_predictions = predict_next_day(last_timestep)

  # Print the predicted yield for each timestep
  print(f"Predicted yield for the next {num_days_to_predict} days:")
  for i, prediction in enumerate(next_3_days_predictions):
    print(f"Timestep {i+1}: {prediction:.4f}")
except Exception as e:
  print(f"Error during prediction: {e}")

# You can further process the predictions here, 
# for example, convert them back to the original units (if needed)
# or plot them to visualize the predicted yield trend.