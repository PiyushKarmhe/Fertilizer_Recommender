from fastapi import FastAPI, Request
import os
import pickle
from pydantic import BaseModel

app = FastAPI()

# Defining a Pydantic model
class FeaturesRequest(BaseModel):
    feature1: str
    feature2: str
    feature3: int
    feature4: int
    feature5: int

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Loading Model
file_path = os.path.join(script_dir, "save/rf_pipeline.pkl")
with open(file_path, "rb") as model_file:
    model = pickle.load(model_file)

# Loading Encode
file_path = os.path.join(script_dir, "save/Encode.pkl")
with open(file_path, "rb") as Encode_file:
    Encode = pickle.load(Encode_file)

print("Model : \n",model)
print("Encode : \n",Encode)

def preProcess(data):

    data[0] = Encode["Encoders"]["Soil Type"].transform([data[0]])[0]
    data[1] = Encode["Encoders"]["Crop Type"].transform([data[1]])[0]

    return data

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(request: Request, features_request: FeaturesRequest):

    # features = ["Sandy","Maize",36,0,0]
    features = [features_request.feature1, features_request.feature2,
                features_request.feature3, features_request.feature4,
                features_request.feature5]
    
    features = preProcess(features)

    print(features)

    prediction = model.predict([features])[0]
    prediction = Encode["InvertEncodings"]["Fertilizer Name"][prediction]

    print("Prediction : ",prediction)
    print("\n---Sent---")

    return {"prediction": prediction}