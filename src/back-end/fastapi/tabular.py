
import json
import pickle

from typing import Any, Dict, List, Optional

import pandas as pd

from fastapi import APIRouter, Depends, HTTPException, status
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

from core import (
    settings,
    llm_client,
    require_api_key,
    logger
)


# =========================================================
# ROUTER
# =========================================================

router = APIRouter(
    tags=["Modelo Tabular"]
)


# =========================================================
# SCHEMAS
# =========================================================

class ConditionPredictionRequest(BaseModel):

    animal_name: str = Field(
        min_length=1,
        max_length=100,
        examples=["dog"]
    )

    symptoms1: str = Field(
        min_length=1,
        max_length=200,
        examples=["fever"]
    )

    symptoms2: str = Field(
        min_length=1,
        max_length=200,
        examples=["vomiting"]
    )

    symptoms3: str = Field(
        min_length=1,
        max_length=200,
        examples=["diarrhea"]
    )

    symptoms4: str = Field(
        min_length=1,
        max_length=200,
        examples=["fatigue"]
    )

    symptoms5: str = Field(
        min_length=1,
        max_length=200,
        examples=["dehydration"]
    )


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

    input_used: Dict[str, Any]

    interpretation: Optional[DiagnosticInterpretation] = None


class TabularHealthResponse(BaseModel):

    status: str

    environment: str

    model: str

    dataset: str

    animals: List[str]

    symptoms: List[str]

    animal_count: int

    symptom_count: int

    dataset_rows: int


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(value: Any) -> str:

    if value is None:

        return "unknown"

    value = str(value).strip().lower()

    if not value or value in {"nan", "none", "null"}:

        return "unknown"

    replacements = {
        "seizuers": "seizures",
        "anorexia": "loss of appetite",
        "poor appetite": "loss of appetite",
        "tiredness": "fatigue"
    }

    return replacements.get(value, value)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def build_features(payload: ConditionPredictionRequest):

    symptoms = [
        normalize_text(payload.symptoms1),
        normalize_text(payload.symptoms2),
        normalize_text(payload.symptoms3),
        normalize_text(payload.symptoms4),
        normalize_text(payload.symptoms5)
    ]

    row = {

        "AnimalName": normalize_text(
            payload.animal_name
        ),

        "symptoms1": symptoms[0],

        "symptoms2": symptoms[1],

        "symptoms3": symptoms[2],

        "symptoms4": symptoms[3],

        "symptoms5": symptoms[4],

        "unique_symptom_count": len(
            set(symptoms)
        ),

        "unknown_symptom_count": sum(
            symptom == "unknown"
            for symptom in symptoms
        ),

        "symptom_text_length": len(
            " ".join(symptoms)
        )
    }

    return pd.DataFrame([row])


# =========================================================
# UTILITIES
# =========================================================

def safe_float(value):

    if value is None:

        return None

    try:

        return float(value)

    except (TypeError, ValueError):

        return None


# =========================================================
# LLM INTERPRETER
# =========================================================

class LLMInterpreter:

    def __init__(self, settings):

        self.settings = settings

    # -----------------------------------------------------
    # CONDITION INTERPRETATION
    # -----------------------------------------------------

    def interpret_condition(self, prediction):

        prompt_payload = {

            "tipo_resultado": (
                "modelo_tabular_condicao_clinica"
            ),

            "entrada_normalizada": (
                prediction.get("input_used", {})
            ),

            "classe_predita": (
                prediction.get("prediction_label")
            ),

            "predicao_binaria": (
                prediction.get("prediction_binary")
            ),

            "probabilidade_risco": (
                prediction.get("probability_yes")
            )
        }

        fallback = self._fallback_condition(
            prediction
        )

        return self._interpret(
            prompt_payload=prompt_payload,
            fallback=fallback
        )

    # -----------------------------------------------------
    # INTERPRETATION GENERATION
    # -----------------------------------------------------

    def _interpret(
        self,
        prompt_payload,
        fallback
    ):

        if not self.settings.llm_enabled:

            return fallback

        messages = [

            {
                "role": "system",

                "content": (
                    "Você é um assistente especializado "
                    "na interpretação de resultados de "
                    "modelos de machine learning aplicados "
                    "à triagem veterinária.\n\n"

                    "Responda sempre em português do Brasil.\n\n"

                    "Não apresente diagnósticos definitivos.\n"
                    "Não prescreva medicamentos ou tratamentos.\n"
                    "Não invente informações clínicas.\n"
                    "Não confunda probabilidade do modelo com "
                    "probabilidade clínica real.\n"
                    "Considere os resultados como apoio à triagem.\n"
                    "Recomende avaliação veterinária quando necessário.\n\n"

                    "Responda exclusivamente em JSON válido, "
                    "com as seguintes chaves:\n"

                    "summary: string\n"
                    "risk_level: string\n"
                    "explanation: string\n"
                    "actionable_insights: array de strings\n"
                    "limitations: array de strings\n\n"

                    "Utilize no máximo 4 actionable_insights "
                    "e 3 limitations."
                )
            },

            {
                "role": "user",

                "content": (
                    "Interprete os resultados abaixo.\n\n"

                    "Explique o significado da classificação, "
                    "a probabilidade estimada e as limitações "
                    "do modelo.\n\n"

                    "Não transforme a classificação do modelo "
                    "em diagnóstico veterinário.\n\n"

                    + json.dumps(
                        prompt_payload,
                        ensure_ascii=False,
                        default=str
                    )
                )
            }
        ]

        try:

            result = llm_client.generate(
                messages=messages,
                json_mode=True
            )

            interpreted = llm_client.parse_json(
                result["content"]
            )

            return self._normalize_interpretation(
                interpretation=interpreted,
                generated_by="llm",
                provider=result["provider"],
                model_used=result["model"]
            )

        except Exception:

            logger.exception(
                "Tabular LLM interpretation failed. "
                "Using rule-based fallback."
            )

            return fallback

    # -----------------------------------------------------
    # NORMALIZE INTERPRETATION
    # -----------------------------------------------------

    def _normalize_interpretation(
        self,
        interpretation,
        generated_by,
        provider,
        model_used
    ):

        return {

            "summary": str(
                interpretation.get("summary", "")
            ).strip(),

            "risk_level": str(
                interpretation.get(
                    "risk_level",
                    "indeterminado"
                )
            ).strip(),

            "explanation": str(
                interpretation.get("explanation", "")
            ).strip(),

            "actionable_insights": self._as_list(
                interpretation.get(
                    "actionable_insights"
                )
            )[:4],

            "limitations": self._as_list(
                interpretation.get(
                    "limitations"
                )
            )[:3],

            "model_used": model_used,

            "provider": provider,

            "generated_by": generated_by
        }

    # -----------------------------------------------------
    # LIST NORMALIZATION
    # -----------------------------------------------------

    @staticmethod
    def _as_list(value):

        if isinstance(value, list):

            return [
                str(item).strip()
                for item in value
                if item is not None and str(item).strip()
            ]

        if value is not None and str(value).strip():

            return [str(value).strip()]

        return []

    # -----------------------------------------------------
    # RULE-BASED FALLBACK
    # -----------------------------------------------------

    def _fallback_condition(self, prediction):

        probability = safe_float(
            prediction.get("probability_yes")
        )

        is_critical = (
            prediction.get("prediction_binary") == 1
        )

        if probability is None:

            confidence_text = (
                "sem probabilidade estimada disponível"
            )

        else:

            confidence_text = (
                f"probabilidade estimada de "
                f"{probability * 100:.2f}%"
            )

        risk_level = (
            "alto"
            if is_critical
            else "baixo"
        )

        summary = (

            "O modelo tabular indicou possível "
            "estado crítico."

            if is_critical

            else

            "O modelo tabular não indicou "
            "estado crítico no conjunto informado."
        )

        return {

            "summary": summary,

            "risk_level": risk_level,

            "explanation": (

                f"A classificação retornada foi "
                f"'{prediction.get('prediction_label')}', "
                f"com {confidence_text}. "

                "O resultado deve ser utilizado como "
                "apoio à triagem e não como diagnóstico "
                "veterinário definitivo."
            ),

            "actionable_insights": [

                "Revisar os sintomas informados e confirmar "
                "se representam o quadro atual do animal.",

                "Priorizar avaliação veterinária diante de "
                "piora clínica, dor intensa, apatia persistente "
                "ou desidratação.",

                "Comparar o resultado com histórico clínico, "
                "exame físico e sinais vitais."
            ],

            "limitations": [

                "O modelo utiliza a espécie e cinco sintomas "
                "normalizados, além dos atributos derivados.",

                "A probabilidade depende da distribuição "
                "dos dados utilizados no treinamento.",

                "A interpretação automatizada não substitui "
                "a avaliação de um médico-veterinário."
            ],

            "model_used": "rule-based-fallback",

            "provider": "rules",

            "generated_by": "rules"
        }


# =========================================================
# TABULAR MODEL SERVICE
# =========================================================

class TabularModelService:

    def __init__(self, settings):

        self.settings = settings

        self.model = None

        self.dataset = None

        self.llm_interpreter = LLMInterpreter(
            settings
        )

    # -----------------------------------------------------
    # PRELOAD
    # -----------------------------------------------------

    def preload(self):

        logger.info(
            "Starting tabular module initialization..."
        )

        try:

            logger.info(
                "Loading tabular dataset from Hugging Face..."
            )

            self.dataset = self.load_dataset()

            logger.info(
                "Tabular dataset loaded successfully. Rows: %s",
                len(self.dataset)
            )

            logger.info(
                "Loading tabular model from Hugging Face..."
            )

            self.model = self.load_model()

            logger.info(
                "Tabular model loaded successfully."
            )

            logger.info(
                "Tabular module initialized successfully."
            )

        except Exception:

            logger.exception(
                "Tabular module initialization failed."
            )

            self.dataset = None

            self.model = None

            raise

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    def load_dataset(self):

        path = hf_hub_download(
            repo_id=self.settings.hf_data_repo,
            repo_type="dataset",
            filename=self.settings.hf_data_filename
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

        df = df.rename(
            columns=rename_map
        )

        required_columns = [

            "AnimalName",

            "symptoms1",

            "symptoms2",

            "symptoms3",

            "symptoms4",

            "symptoms5"
        ]

        missing_columns = [

            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Dataset missing required columns: "
                f"{missing_columns}"
            )

        if df.empty:

            raise ValueError(
                "Tabular dataset is empty."
            )

        return df

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    def load_model(self):

        path = hf_hub_download(
            repo_id=self.settings.hf_model_repo,
            filename=self.settings.hf_model_filename
        )

        with open(path, "rb") as file:

            model = pickle.load(file)

        if not hasattr(model, "predict"):

            raise TypeError(
                "Loaded tabular model does not implement predict()."
            )

        return model

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    def predict_condition(self, payload):

        if self.model is None:

            raise RuntimeError(
                "Tabular model not loaded."
            )

        features = build_features(payload)

        raw_prediction = self.model.predict(
            features
        )[0]

        prediction = int(raw_prediction)

        if prediction not in (0, 1):

            raise ValueError(
                f"Unexpected prediction class: {prediction}"
            )

        probability = None

        if hasattr(self.model, "predict_proba"):

            probabilities = self.model.predict_proba(
                features
            )

            if (
                probabilities is not None
                and len(probabilities) > 0
            ):

                classes = getattr(
                    self.model,
                    "classes_",
                    None
                )

                if classes is not None:

                    class_indices = [
                        index
                        for index, class_value in enumerate(classes)
                        if class_value == 1
                    ]

                    if class_indices:

                        probability = float(
                            probabilities[0][class_indices[0]]
                        )

                elif len(probabilities[0]) == 2:

                    probability = float(
                        probabilities[0][1]
                    )

        result = {

            "prediction_binary": prediction,

            "prediction_label": (

                "Yes - perigoso"

                if prediction == 1

                else

                "No - nao perigoso"
            ),

            "probability_yes": probability,

            "input_used": features.iloc[0].to_dict()
        }

        result["interpretation"] = (
            self.llm_interpreter.interpret_condition(
                result
            )
        )

        return result

    # -----------------------------------------------------
    # HEALTH INFORMATION
    # -----------------------------------------------------

    def health(self):

        if self.dataset is None:

            raise RuntimeError(
                "Tabular dataset not loaded."
            )

        df = self.dataset

        required_columns = [

            "AnimalName",

            "symptoms1",

            "symptoms2",

            "symptoms3",

            "symptoms4",

            "symptoms5"
        ]

        missing_columns = [

            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:

            raise RuntimeError(
                f"Dataset missing columns: {missing_columns}"
            )

        animals = sorted(

            df["AnimalName"]

            .dropna()

            .astype(str)

            .str.strip()

            .str.lower()

            .unique()

            .tolist()
        )

        symptoms = []

        for column in required_columns[1:]:

            symptoms.extend(

                df[column]

                .dropna()

                .astype(str)

                .str.strip()

                .str.lower()

                .tolist()
            )

        symptoms = sorted(
            set(
                symptom
                for symptom in symptoms
                if symptom and symptom not in {
                    "nan",
                    "none",
                    "null"
                }
            )
        )

        return {

            "status": "online",

            "environment": self.settings.api_env,

            "model": self.settings.hf_model_repo,

            "dataset": self.settings.hf_data_repo,

            "animals": animals,

            "symptoms": symptoms,

            "animal_count": len(animals),

            "symptom_count": len(symptoms),

            "dataset_rows": len(df)
        }


# =========================================================
# SERVICE INSTANCE
# =========================================================

tabular_service = TabularModelService(
    settings=settings
)


# =========================================================
# INITIALIZATION
# =========================================================

def initialize_tabular():

    tabular_service.preload()


# =========================================================
# ROUTES
# =========================================================

@router.get(
    "/health/tabular",
    response_model=TabularHealthResponse
)
def tabular_health():

    try:

        return tabular_service.health()

    except Exception:

        logger.exception(
            "Tabular health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tabular service unavailable."
        )


@router.post(
    "/predict/condition",
    response_model=ConditionPredictionResponse,
    dependencies=[
        Depends(require_api_key)
    ]
)
def predict_condition(
    payload: ConditionPredictionRequest
):

    try:

        return tabular_service.predict_condition(
            payload
        )

    except Exception:

        logger.exception(
            "Tabular prediction failed."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao processar a predição tabular."
        )
