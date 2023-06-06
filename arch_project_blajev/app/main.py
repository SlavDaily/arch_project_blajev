from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn
import pickle
import sklearn

import warnings
warnings.filterwarnings("ignore")


class Numbers(BaseModel):
    X_axis: float
    Y_axis: float



app = FastAPI()
a=1

@app.get("/")
async def root():
    return {int(a)}


@app.post("/predict")
def predict(numbers: Numbers):

    try:

        with open('app/model.pkl', 'rb') as file:
            model = pickle.load(file)

        result1 = int(model.predict(np.array([numbers.X_axis,numbers.Y_axis]).reshape(-1,2)))


        if result1 ==1:
            result = "yellow class"

        else: 
            result = "blue class"



        return {"prediction": result}
    except (FileNotFoundError, ValidationError):
        raise HTTPException(status_code=500, detail="Internal Server Error")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="0.0.0.0")

