from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Optional
from PIL import Image
from huggingface_hub import hf_hub_download

import io
import json
import pickle
import re
import traceback
import logging
import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from security import require_api_key, validate_image_bytes

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
        "model": "Qwen/Qwen2.5-7B-Instruct"
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


class HealthResponse(BaseModel):
    status: str
    environment: str
    models: HealthModels
    llm: HealthLLM
    tabular: HealthTabular


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


def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class LLMInterpreter:
    def __init__(self, settings):
        self.settings = settings

    def interpret_condition(self, prediction):
        prompt_payload = {
            "tipo_resultado": "modelo_tabular_condicao_clinica",
            "entrada_normalizada": prediction.get("input_used", {}),
            "classe_predita": prediction.get("prediction_label"),
            "predicao_binaria": prediction.get("prediction_binary"),
            "probabilidade_risco": prediction.get("probability_yes")
        }

        return self._interpret(
            prompt_payload=prompt_payload,
            fallback=self._fallback_condition(prediction)
        )

    def interpret_image(self, prediction):
        prompt_payload = {
            "tipo_resultado": "modelo_visao_imagem_canina",
            "racas_provaveis": prediction.get("dog_breeds", [])[:5],
            "condicoes_dermatologicas_provaveis": prediction.get("skin_diseases", [])[:5],
            "mensagem_processamento": prediction.get("message")
        }

        return self._interpret(
            prompt_payload=prompt_payload,
            fallback=self._fallback_image(prediction)
        )

    def _interpret(self, prompt_payload, fallback):
        providers = self.settings.llm_provider_configs

        if not providers:
            return fallback

        for provider_config in providers:
            try:
                interpreted = self._call_provider(prompt_payload, provider_config)

                return self._normalize_interpretation(
                    interpreted,
                    generated_by="llm",
                    provider=provider_config.provider,
                    model_used=provider_config.model
                )

            except Exception:
                logger.warning(
                    "LLM interpretation failed for provider '%s'. Trying next provider.",
                    provider_config.provider
                )
                logger.warning(traceback.format_exc())

        logger.warning("All LLM providers failed. Using rule-based fallback.")
        return fallback

    def _call_provider(self, prompt_payload, provider_config):
        attempts = [provider_config.json_mode]

        if provider_config.json_mode:
            attempts.append(False)

        last_error = None

        for use_json_mode in attempts:
            try:
                content = self._request_chat_completion(
                    prompt_payload=prompt_payload,
                    provider_config=provider_config,
                    use_json_mode=use_json_mode
                )
                return self._parse_json_content(content)

            except Exception as exc:
                last_error = exc

                if use_json_mode:
                    logger.warning(
                        "JSON mode failed for provider '%s'. Retrying without response_format.",
                        provider_config.provider
                    )

        raise last_error or RuntimeError("LLM provider call failed for unknown reason")

    def _request_chat_completion(self, prompt_payload, provider_config, use_json_mode):
        headers = {
            "Content-Type": "application/json"
        }

        if provider_config.api_key:
            headers["Authorization"] = f"Bearer {provider_config.api_key}"

        payload = {
            "model": provider_config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Voce e um assistente de interpretacao para triagem veterinaria. "
                        "Explique resultados de modelos de machine learning em portugues do Brasil, "
                        "com linguagem clara para profissionais de saude veterinaria. "
                        "Nao afirme diagnostico definitivo, nao prescreva tratamento e recomende "
                        "validacao clinica quando houver risco. "
                        "Responda somente JSON valido, sem markdown, com as chaves: summary, risk_level, "
                        "explanation, actionable_insights, limitations."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Interprete os dados abaixo e transforme os numeros em insights acionaveis. "
                        "Use no maximo 4 itens em actionable_insights e 3 em limitations.\n\n"
                        f"{json.dumps(prompt_payload, ensure_ascii=False)}"
                    )
                }
            ]
        }

        if provider_config.provider != "gemini":
            payload["temperature"] = self.settings.llm_temperature

        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(
            f"{provider_config.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.settings.llm_timeout_seconds
        )

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _parse_json_content(self, content):
        if isinstance(content, dict):
            return content

        if not content:
            raise ValueError("Empty LLM response content")

        text = str(content).strip()
        candidates = [text]

        markdown_match = re.search(
            r"```(?:json)?\s*(.*?)```",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        if markdown_match:
            candidates.append(markdown_match.group(1).strip())

        first_brace = text.find("{")
        last_brace = text.rfind("}")

        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            candidates.append(text[first_brace:last_brace + 1])

        last_error = None

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError as exc:
                last_error = exc

        raise ValueError(f"Could not parse LLM response as JSON: {last_error}")

    def _normalize_interpretation(self, interpretation, generated_by, provider, model_used):
        return {
            "summary": str(interpretation.get("summary", "")).strip(),
            "risk_level": str(interpretation.get("risk_level", "indeterminado")).strip(),
            "explanation": str(interpretation.get("explanation", "")).strip(),
            "actionable_insights": self._as_list(interpretation.get("actionable_insights")),
            "limitations": self._as_list(interpretation.get("limitations")),
            "model_used": model_used,
            "provider": provider,
            "generated_by": generated_by
        }

    def _as_list(self, value):
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        if value:
            return [str(value).strip()]

        return []

    def _fallback_condition(self, prediction):
        probability = safe_float(prediction.get("probability_yes"))
        is_critical = prediction.get("prediction_binary") == 1

        if probability is None:
            confidence_text = "sem probabilidade calibrada disponivel"
        else:
            confidence_text = f"probabilidade estimada de risco de {probability * 100:.2f}%"

        risk_level = "alto" if is_critical else "baixo"
        summary = (
            "O modelo tabular indicou possivel estado critico."
            if is_critical
            else "O modelo tabular nao indicou sinais criticos no conjunto informado."
        )

        return {
            "summary": summary,
            "risk_level": risk_level,
            "explanation": (
                f"A classificacao retornada foi '{prediction.get('prediction_label')}', com {confidence_text}. "
                "A leitura deve ser usada como apoio de triagem e nao como diagnostico definitivo."
            ),
            "actionable_insights": [
                "Revisar os sintomas informados e confirmar se representam o quadro atual do animal.",
                "Priorizar avaliacao veterinaria se houver piora, dor intensa, apatia persistente ou desidratacao.",
                "Comparar o resultado com historico clinico, exame fisico e sinais vitais."
            ],
            "limitations": [
                "O modelo usa apenas especie e cinco sintomas normalizados.",
                "A probabilidade depende da distribuicao do dataset usado no treinamento.",
                "A interpretacao automatica nao substitui avaliacao veterinaria."
            ],
            "model_used": "rule-based-fallback",
            "provider": "rules",
            "generated_by": "rules"
        }

    def _fallback_image(self, prediction):
        skin_diseases = prediction.get("skin_diseases", [])
        best_skin = skin_diseases[0] if skin_diseases else None
        best_label = best_skin.get("label") if best_skin else "nao identificado"
        best_confidence = safe_float(best_skin.get("confidence")) if best_skin else None

        if best_label == "healthy":
            risk_level = "baixo"
            summary = "O modelo de imagem apontou pele aparentemente saudavel entre as classes avaliadas."
        elif best_skin:
            risk_level = "moderado"
            summary = f"O modelo de imagem sinalizou possivel condicao dermatologica: {best_label}."
        else:
            risk_level = "indeterminado"
            summary = "O modelo de imagem nao retornou condicao dermatologica suficiente para interpretacao."

        confidence_text = (
            f" com confianca de {best_confidence * 100:.2f}%"
            if best_confidence is not None
            else ""
        )

        return {
            "summary": summary,
            "risk_level": risk_level,
            "explanation": (
                f"A principal classe dermatologica retornada foi '{best_label}'{confidence_text}. "
                "Esse resultado deve orientar a triagem visual e precisa ser confirmado por exame clinico."
            ),
            "actionable_insights": [
                "Verificar qualidade da imagem, iluminacao e foco antes de tomar decisoes.",
                "Comparar a area da pele com sinais clinicos como coceira, descamacao, feridas ou secrecao.",
                "Encaminhar para avaliacao veterinaria se a lesao evoluir ou houver desconforto."
            ],
            "limitations": [
                "O modelo avalia apenas classes presentes no treinamento.",
                "Imagem, angulo e iluminacao podem alterar a confianca.",
                "A interpretacao automatica nao substitui exame dermatologico veterinario."
            ],
            "model_used": "rule-based-fallback",
            "provider": "rules",
            "generated_by": "rules"
        }


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
        self.llm_interpreter = LLMInterpreter(settings)

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
        assert self.tabular_model is not None, "Tabular model not loaded"

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

        result["interpretation"] = self.llm_interpreter.interpret_condition(result)

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

    def predict_image(self, image_bytes):
        assert self.image_model is not None, "Image model not loaded"
        assert self.labels is not None, "Labels not loaded"
        assert self.dataset_map is not None, "Dataset map not loaded"

        if not self.validate_dog_image(image_bytes):
            return {
                "success": False,
                "message": "Apenas imagens de cachorros são permitidas.",
                "dog_breeds": [],
                "skin_diseases": [],
                "interpretation": None
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

        result["interpretation"] = self.llm_interpreter.interpret_image(result)

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
        }
    }


@app.post(
    "/predict/condition",
    response_model=ConditionPredictionResponse,
    dependencies=[Depends(require_api_key)]
)
def predict_condition(payload: ConditionPredictionRequest):
    return model_service.predict_condition(payload)


@app.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    dependencies=[Depends(require_api_key)]
)
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        validate_image_bytes(image_bytes)
        return model_service.predict_image(image_bytes)

    except HTTPException:
        raise

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )
