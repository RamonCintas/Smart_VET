from fastapi import Depends, FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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

from security import require_api_key, validate_image_bytes
from clinical_repository import ClinicalRepository, PatientNotFoundError
from langchain_service import LLMInterpreter as LangChainInterpreter

from keras.applications.mobilenet_v2 import decode_predictions
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input

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

@dataclass
class LLMProviderConfig:
    provider: str
    api_key: Optional[str]
    base_url: str
    model: str
    json_mode: bool = True


LLM_PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.6-luna"
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.6-flash"
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "model": "Qwen/Qwen3-4B-Instruct-2507"
    }
}


def normalize_llm_provider(provider):
    if not provider:
        return None

    normalized = str(provider).strip().lower().replace("_", "-")
    aliases = {
        "google": "gemini",
        "google-gemini": "gemini",
        "hf": "huggingface",
        "hugging-face": "huggingface"
    }

    return aliases.get(normalized, normalized)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_env: str = "dev"

    hf_model_repo: str = "guicon/techchallenge-animal-condition-model"
    hf_model_filename: str = "best_model.pkl"

    hf_data_repo: str = "guicon/techchallenge-animal-condition-dataset"
    hf_data_filename: str = "data.csv"

    hf_image_repo: str = "ramoncg/techchallenge-pet-computer-vision-model"
    hf_image_filename: str = "pet_model.keras"

    openai_api_key: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_json_mode: bool = True
    llm_fallback_1_provider: Optional[str] = None
    llm_fallback_1_api_key: Optional[str] = None
    llm_fallback_1_base_url: Optional[str] = None
    llm_fallback_1_model: Optional[str] = None
    llm_fallback_1_json_mode: bool = True
    llm_timeout_seconds: int = 45
    llm_temperature: float = 0.2
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 1.0
    allow_remote_clinical_context: bool = True

    clinical_data_dir: str = str(Path(__file__).resolve().parent / "data")
    clinical_data_reload: bool = True

    allowed_origins: str = "http://localhost:8501,http://localhost:8000"

    api_key: str = ""
    max_upload_bytes: int = 5 * 1024 * 1024

    @property
    def cors_origins(self):
        return [x.strip() for x in self.allowed_origins.split(",") if x.strip()]

    @property
    def llm_enabled(self):
        return bool(self.llm_provider_configs)

    @property
    def llm_provider_configs(self):
        configs = []

        primary_provider = normalize_llm_provider(self.llm_provider)
        if not primary_provider:
            if self.llm_api_key:
                primary_provider = "gemini"
            elif self.openai_api_key:
                primary_provider = "openai"

        primary = self._build_llm_provider_config(
            provider=primary_provider,
            api_key=self.llm_api_key or (
                self.openai_api_key if primary_provider == "openai" else None
            ),
            base_url=self.llm_base_url,
            model=self.llm_model,
            json_mode=self.llm_json_mode
        )

        if primary:
            configs.append(primary)

        fallback_1 = self._build_llm_provider_config(
            provider=self.llm_fallback_1_provider,
            api_key=self.llm_fallback_1_api_key,
            base_url=self.llm_fallback_1_base_url,
            model=self.llm_fallback_1_model,
            json_mode=self.llm_fallback_1_json_mode
        )

        if fallback_1:
            configs.append(fallback_1)

        return configs

    def _build_llm_provider_config(self, provider, api_key, base_url, model, json_mode):
        provider = normalize_llm_provider(provider)

        if not provider:
            return None

        defaults = LLM_PROVIDER_DEFAULTS.get(provider, {})
        resolved_base_url = base_url or defaults.get("base_url")
        resolved_model = model or defaults.get("model")

        if not resolved_base_url or not resolved_model:
            logger.warning("LLM provider ignored due to missing base URL or model: %s", provider)
            return None

        if not api_key:
            logger.warning("LLM provider ignored due to missing API key: %s", provider)
            return None

        return LLMProviderConfig(
            provider=provider,
            api_key=api_key,
            base_url=resolved_base_url,
            model=resolved_model,
            json_mode=json_mode
        )


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()


# =========================================================
# SCHEMAS
# =========================================================

class ConditionPredictionRequest(BaseModel):
    patient_id: Optional[str] = Field(default=None, examples=["VET-001"])
    animal_name: str = Field(examples=["dog"])
    symptoms1: str = Field(examples=["fever"])
    symptoms2: str = Field(examples=["vomiting"])
    symptoms3: str = Field(examples=["diarrhea"])
    symptoms4: str = Field(examples=["fatigue"])
    symptoms5: str = Field(examples=["dehydration"])


class HealthModels(BaseModel):
    tabular_model: str
    dataset: str
    image_model: str


class HealthLLM(BaseModel):
    enabled: bool
    model: str
    provider: str


class HealthTabular(BaseModel):
    animals: List[str]
    symptoms: List[str]
    animal_count: int
    symptom_count: int
    dataset_rows: int


class HealthClinicalData(BaseModel):
    patient_count: int
    medication_count: int
    data_version: str
    loaded_at: str


class HealthResponse(BaseModel):
    status: str
    environment: str
    models: HealthModels
    llm: HealthLLM
    tabular: HealthTabular
    clinical_data: HealthClinicalData


class ClinicalContextMetadata(BaseModel):
    patient_id: str
    patient_name: str
    protocol_version: str
    data_version: str
    matched_medication_ids: List[str]


class PatientSummary(BaseModel):
    id: str
    name: str
    species: str
    breed: str


class DiagnosticInterpretation(BaseModel):
    summary: str
    risk_level: str
    explanation: str
    actionable_insights: List[str]
    limitations: List[str]
    model_used: str
    provider: str
    generated_by: str


class ConditionPredictionResponse(BaseModel):
    prediction_binary: int
    prediction_label: str
    probability_yes: Optional[float]
    input_used: Dict
    interpretation: Optional[DiagnosticInterpretation] = None
    clinical_context: Optional[ClinicalContextMetadata] = None


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
    interpretation: Optional[DiagnosticInterpretation] = None
    clinical_context: Optional[ClinicalContextMetadata] = None


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Smart VET API",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


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
        self.clinical_repository = ClinicalRepository(
            settings.clinical_data_dir,
            reload_on_change=settings.clinical_data_reload
        )
        self.llm_interpreter = LangChainInterpreter(
            settings,
            self.clinical_repository
        )

    def preload(self):
        try:
            logger.info("Loading structured clinical data...")
            self.clinical_repository.load(force=True)

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
        assert self.tabular_model is not None, "Tabular model not loaded"

        self.llm_interpreter.validate_patient(payload.patient_id)

        features = build_features(payload)

        prediction = int(self.tabular_model.predict(features)[0])

        probability = None
        if hasattr(self.tabular_model, "predict_proba"):
            probability = float(
                self.tabular_model.predict_proba(features)[0][1]
            )

        result = {
            "prediction_binary": prediction,
            "prediction_label": "Yes - perigoso" if prediction == 1 else "No - nao perigoso",
            "probability_yes": probability,
            "input_used": features.iloc[0].to_dict()
        }

        interpretation, clinical_context = self.llm_interpreter.interpret_condition(
            result,
            patient_id=payload.patient_id
        )
        result["interpretation"] = interpretation
        result["clinical_context"] = clinical_context

        return result
    
    def validate_dog_image(self, image_bytes):
        assert self.dog_detector is not None, "Dog detector not loaded"

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

    def predict_image(self, image_bytes, patient_id=None):
        assert self.image_model is not None, "Image model not loaded"
        assert self.labels is not None, "Labels not loaded"
        assert self.dataset_map is not None, "Dataset map not loaded"

        self.llm_interpreter.validate_patient(patient_id)

        if not self.validate_dog_image(image_bytes):
            return {
                "success": False,
                "message": "Apenas imagens de cachorros são permitidas.",
                "dog_breeds": [],
                "skin_diseases": [],
                "interpretation": None,
                "clinical_context": None
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
    
        result = {
            "success": True,
            "message": "Imagem processada com sucesso.",
            "dog_breeds": dog_predictions[:5],
            "skin_diseases": skin_predictions[:5]
        }

        interpretation, clinical_context = self.llm_interpreter.interpret_image(
            result,
            patient_id=patient_id
        )
        result["interpretation"] = interpretation
        result["clinical_context"] = clinical_context

        return result


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
    assert df is not None, "Tabular dataset not loaded"

    llm_configs = settings.llm_provider_configs
    clinical_data = model_service.clinical_repository.metadata()

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
        "llm": {
            "enabled": settings.llm_enabled,
            "model": " -> ".join(config.model for config in llm_configs) or "rule-based-fallback",
            "provider": " -> ".join(config.provider for config in llm_configs) or "rules"
        },
        "tabular": {
            "animals": animals,
            "symptoms": symptoms,
            "animal_count": len(animals),
            "symptom_count": len(symptoms),
            "dataset_rows": len(df)
        },
        "clinical_data": clinical_data
    }


@app.get(
    "/patients",
    response_model=List[PatientSummary],
    dependencies=[Depends(require_api_key)]
)
def list_patients():
    return model_service.clinical_repository.list_patients()


@app.post(
    "/predict/condition",
    response_model=ConditionPredictionResponse,
    dependencies=[Depends(require_api_key)]
)
def predict_condition(payload: ConditionPredictionRequest):
    try:
        return model_service.predict_condition(payload)
    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    dependencies=[Depends(require_api_key)]
)
async def predict_image(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(default=None)
):
    try:
        image_bytes = await file.read()
        validate_image_bytes(image_bytes)
        return model_service.predict_image(image_bytes, patient_id=patient_id)

    except PatientNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    except HTTPException:
        raise

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )
