from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fastapi.middleware.cors import CORSMiddleware
# Creating FastAPI instance
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict it to specific origins if needed
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load the dataset
plant_health_data = pd.read_csv('plant_health_dataset.csv')
X = plant_health_data.drop(columns=['PlantHealth', 'SpotArea', 'NumHoles', 'GreenColorIntensity', 'PlantHeight', 'DustPresence', 'LeafEdgeType', 'NumLeaves', 'SoilPHLevel'], axis=1)
Y = plant_health_data['PlantHealth']

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standard_data = scaler.transform(X)
x = standard_data
y = Y

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.005, stratify=y, random_state=2)

# Create and train the Random Forest classifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

# Print train and test accuracy
y_train_pred = classifier.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
#print(f"Train Accuracy: {train_accuracy:.4f}")

y_test_pred = classifier.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
#print(f"Test Accuracy: {test_accuracy:.4f}")

# Define the request body model
class RequestBody(BaseModel):
    LeafHumidity: float
    PlantAge: float
    SunExposureLevel: float

# Endpoint to make predictions
@app.post('/predict')
def predict(data: RequestBody):
    # Making the data in a form suitable for prediction
    test_data = [[data.LeafHumidity, data.PlantAge, data.SunExposureLevel]]
    print(test_data)
    
    # Standardize the input data
    np_array_data = np.asarray(test_data)
    reshaped = np_array_data.reshape(1, -1)
    std = scaler.transform(reshaped)
    
    # Make prediction
    prediction = classifier.predict(std)
    print('predictionnnnnnnnnnnnnnn', prediction)
    return {'PlantHealth': prediction.tolist()}
