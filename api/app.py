import pickle
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from tempfile import NamedTemporaryFile
import os

# Load the trained model
model = pickle.load(open('omdl.pkl', 'rb'))

app = FastAPI()

@app.post("/predict")
async def handler(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file was provided")

    # Create a temporary file and write the uploaded file contents to it
    with NamedTemporaryFile(delete=False) as temp:
        temp.write(file.file.read())
        temp_file_path = temp.name

    try:
        # Read the uploaded Excel file from the temporary file path
        my_data = pd.read_excel(temp_file_path, engine='openpyxl')

        # Ensure all required columns are present
        required_columns = ['Gender', 'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
                            'Study Satisfaction', 'Job Satisfaction', 'Have you ever had suicidal thoughts ?',
                            'Work/Study Hours', 'Family History of Mental Illness', 'Financial Stress_2.0',
                            'Financial Stress_3.0', 'Financial Stress_4.0', 'Financial Stress_5.0']
        
        if not all(col in my_data.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="Uploaded file does not have the required columns")

        # Use the model to make predictions
        result = model.predict(my_data)
        
        # Convert prediction values to human-readable risk levels
        risk_mapping = {1: "High Risk of Depression", 0: "Low Risk of Depression"}
        my_data['Risk Level'] = [risk_mapping[pred] for pred in result]

        # Convert the DataFrame to a list of dictionaries
        data = my_data.to_dict('records')

        # Return the data as a JSON response
        return JSONResponse(content={'result': data})
    
    finally:
        # Remove the temporary file
        os.unlink(temp_file_path)

@app.get("/", response_class=RedirectResponse)
def redirect_to_docs():
    return "/docs"
