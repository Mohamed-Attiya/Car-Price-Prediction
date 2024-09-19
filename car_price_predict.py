import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import r2_score , mean_squared_error
import pickle
import streamlit as st

# Ignore warnings
warnings.filterwarnings('ignore')

# Print today's date
date_today = datetime.datetime.now()
print(date_today)
print("============================")

# Read the data
data = pd.read_csv("E:\\Car Price Prediction\\car_price_prediction.csv")

# Show basic data information
print(data.head())
print("============================")
print(data.shape)
print("============================")
print(data.info())
print("============================")
print(data.describe())
print("============================")
print(data.duplicated().sum())
print("============================")

# Remove duplicates
data.drop_duplicates(inplace=True)
print(data.shape)
print("============================")
print(data.isnull().sum())
print("============================")

# Print the number of unique values for each column
for col in data.columns:
    print(col, ":", data[col].nunique())
print("============================")  

# Plot histograms for all numerical columns
data.hist(bins=15, figsize=(15, 10))
plt.show()
print("============================")   

# Count the number of cars for each manufacturer
print(data['Manufacturer'].value_counts().sort_values())
print("============================")   
print(data['Manufacturer'].value_counts().sort_values(ascending=False))
print("============================")   

# Plot the top 10 manufacturers
manfacturer_desc = data['Manufacturer'].value_counts().sort_values(ascending=False)[:10]
print(manfacturer_desc)
manfacturer_desc.plot(figsize=(10, 4))
plt.show()
print("============================")  

# Calculate the average price for each manufacturer
manfacturer_desc_price = [data[data['Manufacturer'] == i]['Price'].mean() for i in list(manfacturer_desc.index)]
print(manfacturer_desc_price)
print("============================")  

# Calculate and plot the correlation matrix
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.3f')
plt.title("Correlation Matrix Heatmap")
plt.show()
print("============================") 

# Display info for object columns
data_object = data.select_dtypes(include=['object'])
print(data_object.info())

# Plot bar charts for the top 10 values in each object column
for col in data_object:
    plt.figure(figsize=(15, 5))
    x = data[col].value_counts()[:10]
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'grey', 'yellow', 'cyan']
    plt.bar(x.index, x.values, color=colors[:len(x)])
    plt.title(f"Top 10 of {col}")
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Drop unnecessary columns 'ID' and 'Doors'
data = data.drop(['ID', 'Doors'], axis=1)   
print("============================")
print(data.head()) 
print("============================")

# Handle 'Levy' column (replace '-' with '0' and convert to numeric)
if 'Levy' in data.columns:
    data['Levy'] = data['Levy'].replace("-", "0")
    data['Levy'] = data['Levy'].astype(int)
    print(data.info())
else:
    print("Column 'Levy' not found in the DataFrame")
print("============================")    
print(data.head())
print("============================")  

# Clean 'Mileage' column (remove 'km' and convert to integer)
if 'Mileage' in data.columns:
    data['Mileage'] = data['Mileage'].str.replace('km', '').str.strip()
    data['Mileage'] = data['Mileage'].astype(int)
    print(data['Mileage'].head())
else:
    print("Column 'Mileage' not found in the DataFrame")
print("============================") 
print(data.head())
print("============================") 

# Clean 'Engine volume' column (remove 'Turbo' and convert to float)
if 'Engine volume' in data.columns:
    data['Engine volume'] = data['Engine volume'].str.replace('Turbo', '').str.strip()
    data['Engine volume'] = data['Engine volume'].astype(float)
    print(data['Engine volume'].head())
else:
    print("Column 'Engine volume' not found in the DataFrame")
print("============================") 
print(data.info())
print("============================")  

# Add a new column 'Age' based on the 'Prod. year' column
data['Age'] = datetime.datetime.now().year - data['Prod. year']
print(data[['Prod. year', 'Age']].head())
print("============================")

# Detect and remove outliers using IQR for each numerical column
numeric_data = data.select_dtypes(exclude='object')  
for col in numeric_data:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = ((numeric_data[col] > high) | (numeric_data[col] < low)).sum()
    total = numeric_data[col].shape[0]
    print(f"Total Outliers in {col}: {outliers} ({round(100 * (outliers)/total, 2)}%)")
    print("============================") 
    if outliers > 0:
        data = data.loc[(data[col] <= high) & (data[col] >= low)]

# Print data after outlier removal
print("Data after outlier removal:")
print(data.head())
print("============================")
print(data.shape)
print("============================")

# Apply LabelEncoder to object (categorical) columns
object_data = data.select_dtypes(include='object')
num_data = data.select_dtypes(exclude='object')
label_encoder = LabelEncoder()

for i in range(0 , object_data.shape[1]):
    object_data.iloc[:,i] = label_encoder.fit_transform(object_data.iloc[: , i])
data = pd.concat([object_data , num_data] , axis = 1)    
print(data) 

# Concatenate numeric and encoded object columns, ensuring numeric columns come first
data = pd.concat([object_data , num_data] , axis = 1)    
print(data) 
# Print final data with numeric columns (including 'Age') at the start
print(data.head())
print("============================")
print(data.info())
print("============================")

# Create Model
# Split the dataset into features (X) and target (y)
x = data.drop('Price' , axis = 1)
y = data['Price']
X_train , X_test, y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42)

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Classifier': RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=2),
    'SVR': SVR()
}

# Initialize lists to store results
R2 = []
RMSE = []

# Train each model and collect evaluation metrics
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    R2.append(r2)
    RMSE.append(rmse)
    print(f"{name} - R2 Score: {r2}, RMSE: {rmse}")

# Create a DataFrame to display the results
results_df = pd.DataFrame({
    'Algorithm': list(models.keys()),
    'R2 Score': R2,
    'Root Mean Squared Error': RMSE
})

# Display the results
print(results_df)

# Plotting the results
fig , ax = plt.subplots(figsize=(12, 6))
ax.plot(results_df.Algorithm, R2, label="R2_Score", marker='o')
ax.plot(results_df.Algorithm, RMSE, label="Root_Mean_Squared_Error", marker='o')
ax.set_title("Model Performance Comparison")
ax.set_xlabel("Algorithm")
ax.set_ylabel("Score")
ax.legend()
plt.show()

# Using My Model to Predict New Data
file_name = 'Cars Predictions.sav'
pickle.dump(models['Random Forest Classifier'] , open(file_name , 'wb'))

# Title for the Streamlit app
st.title("Car Price Prediction")

# Load the saved Random Forest model
file_name = 'Cars Predictions.sav'
loaded_model = pickle.load(open(file_name, 'rb'))

# Create inputs for the user to provide values for the features
st.header("Input Car Features")

# Example of some input fields
manufacturer = st.selectbox('Select Manufacturer', data['Manufacturer'].unique())
mileage = st.number_input('Enter Mileage (in km)', min_value=0)
engine_volume = st.number_input('Enter Engine Volume (L)', min_value=0.0, step=0.1)
age = st.number_input('Enter Car Age (years)', min_value=0)

# Additional input fields as per your dataset's features

# When the 'Predict' button is clicked
if st.button('Predict'):
    # Create a DataFrame with the user inputs
    input_data = pd.DataFrame({
        'Manufacturer': [manufacturer],
        'Mileage': [mileage],
        'Engine volume': [engine_volume],
        'Age': [age]
        # Include other required features here
    })

    # Ensure the input data is in the same format as the training data
    # (e.g., label encoding, handling missing data, etc.)

    # Use the model to make predictions
    prediction = loaded_model.predict(input_data)

    # Display the prediction result
    st.write(f"The predicted price of the car is: ${prediction[0]:,.2f}")