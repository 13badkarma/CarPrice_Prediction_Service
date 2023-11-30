from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from contextlib import asynccontextmanager

SCHEMA_PATH = "./data/schema.pickle"
MODEL_PATH = "./data/model.pickle"


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # загрузка ML модели и схемы
    with open(MODEL_PATH, 'rb') as model, open(SCHEMA_PATH, 'rb') as schema:
        carprice_model = pickle.load(model)
        carprice_schema = pickle.load(schema)
    ml_models["carprice"] = carprice_model
    ml_models["schema"] = carprice_schema
    yield
    # очистка
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


# решил чуть упростить, не стал переносить обработку дубликатов и nan
def preprocess_data(df: pd.DataFrame):
    # удаляем лишнии колонки
    df.drop(columns=['torque', 'seats', 'engine', 'mileage', 'selling_price'], inplace=True)
    # так как дизель был самый популярный он 1 все остальное 0
    df['fuel'] = df['fuel'].map(lambda x: 1 if x == 'Diesel' else 0)
    # Manual самая популярная поэтому 1, остальное 0. Вдруг попадется Robot коробка
    df['transmission'] = df['transmission'].map(lambda x: 1 if x == 'Manual' else 0)
    df['max_power'] = df.loc[:, ['max_power']].replace(regex=True, to_replace=r'[^0-9.\\-]', value=r'')
    df['max_power'] = df['max_power'].apply(pd.to_numeric)
    df['name'] = df.name.str.extract(r"^(?P<brand_name>\w{1,})\s.*$").brand_name
    return df.reindex(labels=ml_models['schema'], axis=1, fill_value=0)


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.model_dump()])
    pred = ml_models['carprice'].predict(preprocess_data(df))
    return pred


@app.post("/predict_items")
def upload(file: UploadFile) -> List[dict]:
    df = pd.read_csv(file.file)
    df_ml = df.copy()
    df_ml = preprocess_data(df_ml)
    pred = ml_models['carprice'].predict(df_ml)
    df['predicted_price'] = pd.Series(pred)
    return df.to_dict(orient='records')

# можно раскоментить для запуска из IDE
# uvicorn.run(app, port=8000)
