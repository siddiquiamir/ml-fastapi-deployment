from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model/model.pkl')

# Create instance of Fast API
app = FastAPI()

# Define the request body for input data (features/independent variable)
class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the prediction endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    data = np.array([[request.sepal_length, request.sepal_width, request.petal_length, request.petal_width]])
    prediction = model.predict(data)
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    return {"prediction": species_map[int(prediction[0])]}

# This will start the server, and you'll be able to access your API at http://127.0.0.1:8000.


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris classification API"}

# Test Your API
# You can test your API using the interactive documentation at http://127.0.0.1:8000/docs