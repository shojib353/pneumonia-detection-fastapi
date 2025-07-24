# app/main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image_from_bytes

app = FastAPI()

model = load_model("bestx_model.keras")
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}
@app.get("/health")
def health():
    return {"status": "ok health"}

@app.post("/result")
async def predict(file: UploadFile = File(...)):
    # return JSONResponse(content={"message": "Prediction endpoint is not implemented yet."})
    image_bytes = await file.read()
    input_array = preprocess_image_from_bytes(image_bytes)

    if input_array is None:
        return JSONResponse(content={"error": "Invalid image"}, status_code=400)

    prediction = model.predict(input_array)
    class_idx = int(prediction[0][0] > 0.5)
    label = "Pneumonia" if class_idx == 1 else "Normal"

    return {
        "prediction": label,
        "confidence": float(prediction[0][0])
    }
