from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import List, Dict, Optional
from PIL import Image
from huggingface_hub import hf_hub_download

import io
import json
import pickle
import traceback
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================================================
# PERFORMANCE TENSORFLOW
# =========================================================

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


# =========================================================
# LOGS
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("smart-vet")


# =========================================================
# CONFIG
# =========================================================

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_env: str = "dev"

    hf_model_repo: str = "guicon/techchallenge-animal-condition-model"
    hf_model_filename: str = "best_model.pkl"

    hf_data_repo: str = "guicon/techchallenge-animal-condition-dataset"
    hf_data_filename: str = "data.csv"

    hf_image_repo: str = "ramoncg/techchallenge-pet-computer-vision-model"
    hf_image_filename: str = "pet_model.keras"

    allowed_origins: str = "http://localhost:8501,http://localhost:8000"

    @property
    def cors_origins(self):
        return [x.strip() for x in self.allowed_origins.split(",") if x.strip()]


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()


# =========================================================
# SCHEMAS
# =========================================================

class ConditionPredictionRequest(BaseModel):
    animal_name: str = Field(example="dog")
    symptoms1: str = Field(example="fever")
    symptoms2: str = Field(example="vomiting")
    symptoms3: str = Field(example="diarrhea")
    symptoms4: str = Field(example="fatigue")
    symptoms5: str = Field(example="dehydration")


class HealthModels(BaseModel):
    tabular_model: str
    dataset: str
    image_model: str


class HealthTabular(BaseModel):
    animals: List[str]
    symptoms: List[str]
    animal_count: int
    symptom_count: int
    dataset_rows: int


class HealthResponse(BaseModel):
    status: str
    environment: str
    models: HealthModels
    tabular: HealthTabular


class ConditionPredictionResponse(BaseModel):
    prediction_binary: int
    prediction_label: str
    probability_yes: Optional[float]
    input_used: Dict


class PredictionItem(BaseModel):
    label: str
    confidence: float
    dataset: str
    reference_image: str


class ImagePredictionResponse(BaseModel):
    success: bool
    message: str
    dog_breeds: List[PredictionItem] = []
    skin_diseases: List[PredictionItem] = []


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Smart VET API",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# HELPERS
# =========================================================

def normalize_text(value):
    if not value:
        return "unknown"

    value = str(value).strip().lower()

    replacements = {
        "seizuers": "seizures",
        "anorexia": "loss of appetite",
        "poor appetite": "loss of appetite",
        "tiredness": "fatigue"
    }

    return replacements.get(value, value)


def build_features(payload):
    symptoms = [
        normalize_text(payload.symptoms1),
        normalize_text(payload.symptoms2),
        normalize_text(payload.symptoms3),
        normalize_text(payload.symptoms4),
        normalize_text(payload.symptoms5),
    ]

    row = {
        "AnimalName": normalize_text(payload.animal_name),
        "symptoms1": symptoms[0],
        "symptoms2": symptoms[1],
        "symptoms3": symptoms[2],
        "symptoms4": symptoms[3],
        "symptoms5": symptoms[4],
        "unique_symptom_count": len(set(symptoms)),
        "unknown_symptom_count": sum(x == "unknown" for x in symptoms),
        "symptom_text_length": len(" ".join(symptoms)),
    }

    return pd.DataFrame([row])


# =========================================================
# MODEL SERVICE
# =========================================================

class ModelService:
    def __init__(self):
        self.tabular_model = None
        self.tabular_dataset = None
        self.image_model = None
        self.dog_detector = None
        self.labels = None
        self.dataset_map = None

    def preload(self):
        try:
            logger.info("Loading dataset...")
            self.tabular_dataset = self.load_tabular_dataset()
    
            logger.info("Loading tabular model...")
            self.tabular_model = self.load_tabular_model()

            logger.info("Loading dog detector...")
            self.dog_detector = self.load_dog_detector()
    
            logger.info("Loading image model...")
            self.image_model = self.load_image_model()
    
            logger.info("Loading labels...")
            self.labels = self.load_labels()
    
            logger.info("Loading dataset map...")
            self.dataset_map = self.load_dataset_map()
    
            logger.info("Warming image model...")
            dummy = np.zeros((1, 160, 160, 3), dtype=np.float32)
            self.image_model.predict(dummy, verbose=0)
    
            logger.info("Preload complete")
    
        except Exception:
            logger.error(traceback.format_exc())
            raise

    def load_tabular_dataset(self):
        path = hf_hub_download(
            repo_id=settings.hf_data_repo,
            repo_type="dataset",
            filename=settings.hf_data_filename
        )
        df = pd.read_csv(path)

        rename_map = {
            "Animal": "AnimalName",
            "Symptom 1": "symptoms1",
            "Symptom 2": "symptoms2",
            "Symptom 3": "symptoms3",
            "Symptom 4": "symptoms4",
            "Symptom 5": "symptoms5"
        }

        return df.rename(columns=rename_map)

    def load_tabular_model(self):
        path = hf_hub_download(
            repo_id=settings.hf_model_repo,
            filename=settings.hf_model_filename
        )

        with open(path, "rb") as f:
            return pickle.load(f)
        
    def load_dog_detector(self):
        return MobileNetV2(
            weights="imagenet",
            include_top=True
        )

    def load_image_model(self):
        path = hf_hub_download(
            repo_id=settings.hf_image_repo,
            filename=settings.hf_image_filename
        )

        return tf.keras.models.load_model(
            path,
            compile=False,
            safe_mode=False
        )

    def load_labels(self):
        path = hf_hub_download(
            repo_id=settings.hf_image_repo,
            filename="labels.json"
        )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["classes"]

    def load_dataset_map(self):
        path = hf_hub_download(
            repo_id=settings.hf_image_repo,
            filename="dataset_map.json"
        )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def predict_condition(self, payload):
        features = build_features(payload)

        prediction = int(self.tabular_model.predict(features)[0])

        probability = None
        if hasattr(self.tabular_model, "predict_proba"):
            probability = float(
                self.tabular_model.predict_proba(features)[0][1]
            )

        return {
            "prediction_binary": prediction,
            "prediction_label": "Yes - perigoso" if prediction == 1 else "No - nao perigoso",
            "probability_yes": probability,
            "input_used": features.iloc[0].to_dict()
        }
    
    def validate_dog_image(self, image_bytes):
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            img = img.resize((224, 224))

            img_array = tf.keras.utils.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

        preds = self.dog_detector.predict(img_array, verbose=0)
        decoded = decode_predictions(preds, top=5)[0]

        dog_keywords = [
            "dog",
            "puppy",
            "retriever",
            "terrier",
            "bulldog",
            "shepherd",
            "husky",
            "poodle",
            "beagle",
            "rottweiler",
            "doberman",
            "chihuahua"
        ]

        for _, label, score in decoded:
            label_lower = label.lower()

            if any(keyword in label_lower for keyword in dog_keywords):
                return True

        return False

    def predict_image(self, image_bytes):
        if not self.validate_dog_image(image_bytes):
            return {
                "success": False,
                "message": "Apenas imagens de cachorros são permitidas.",
                "dog_breeds": [],
                "skin_diseases": []
            }
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            img = img.resize((160, 160))
    
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
    
        preds = self.image_model.predict(img_array, verbose=0)[0]
    
    
        dog_predictions = []
        skin_predictions = []
    
    
        for idx, score in enumerate(preds):
            label = self.labels[idx]
            dataset = self.dataset_map.get(label, "Unknown")
    
            item = {
                "label": label,
                "confidence": float(score),
                "dataset": dataset,
                "reference_image": f"https://huggingface.co/{settings.hf_image_repo}/resolve/main/references/{label}.jpg"
            }
    
            if dataset in {"Stanford-Dogs", "Oxford-Pets"}:
                dog_predictions.append(item)
    
            elif dataset == "Kaggle-Skin-Disease":
                skin_predictions.append(item)
    
        dog_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        skin_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    
        return {
            "success": True,
            "message": "Imagem processada com sucesso.",
            "dog_breeds": dog_predictions[:5],
            "skin_diseases": skin_predictions[:5]
        }


model_service = ModelService()


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():
    model_service.preload()


# =========================================================
# ROUTES
# =========================================================

@app.get("/health", response_model=HealthResponse)
def health():
    df = model_service.tabular_dataset

    animals = sorted(
        df["AnimalName"].dropna().astype(str).str.lower().unique().tolist()
    )

    symptoms = []

    for col in ["symptoms1", "symptoms2", "symptoms3", "symptoms4", "symptoms5"]:
        symptoms.extend(
            df[col].dropna().astype(str).str.lower().tolist()
        )

    symptoms = sorted(list(set(symptoms)))

    return {
        "status": "online",
        "environment": settings.api_env,
        "models": {
            "tabular_model": settings.hf_model_repo,
            "dataset": settings.hf_data_repo,
            "image_model": settings.hf_image_repo
        },
        "tabular": {
            "animals": animals,
            "symptoms": symptoms,
            "animal_count": len(animals),
            "symptom_count": len(symptoms),
            "dataset_rows": len(df)
        }
    }


@app.post("/predict/condition", response_model=ConditionPredictionResponse)
def predict_condition(payload: ConditionPredictionRequest):
    return model_service.predict_condition(payload)


@app.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        return model_service.predict_image(image_bytes)

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )
