
import json
from io import BytesIO
from typing import Any, List, Optional

import numpy as np
import tensorflow as tf

from PIL import (
    Image,
    UnidentifiedImageError
)

from keras.applications import MobileNetV2

from keras.applications.mobilenet_v2 import (
    decode_predictions,
    preprocess_input
)

from huggingface_hub import hf_hub_download

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    status
)

from starlette.concurrency import run_in_threadpool

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
    tags=["Visão Computacional"]
)


# =========================================================
# SCHEMAS
# =========================================================

class DiagnosticInterpretation(BaseModel):

    summary: str

    risk_level: str

    explanation: str

    actionable_insights: List[str]

    limitations: List[str]

    model_used: str

    provider: str

    generated_by: str


class PredictionItem(BaseModel):

    label: str

    confidence: float

    dataset: str

    reference_image: str


class ImagePredictionResponse(BaseModel):

    success: bool

    message: str

    dog_breeds: List[PredictionItem] = Field(
        default_factory=list
    )

    skin_diseases: List[PredictionItem] = Field(
        default_factory=list
    )

    interpretation: Optional[
        DiagnosticInterpretation
    ] = None


# =========================================================
# IMAGE VALIDATION
# =========================================================

def validate_image_bytes(image_bytes: bytes):

    if not image_bytes:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image cannot be empty."
        )

    if len(image_bytes) > settings.max_upload_bytes:

        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds the allowed size."
        )

    try:

        with Image.open(
            BytesIO(image_bytes)
        ) as image:

            if image.format not in {
                "JPEG",
                "PNG",
                "WEBP"
            }:

                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Unsupported image format."
                )

            image.verify()

    except HTTPException:

        raise

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError
    ) as exc:

        logger.warning(
            "Rejected invalid image upload."
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file."
        ) from exc


# =========================================================
# UTILITIES
# =========================================================

def safe_float(value: Any):

    if value is None:

        return None

    try:

        result = float(value)

        if not np.isfinite(result):

            return None

        return result

    except (TypeError, ValueError, OverflowError):

        return None


# =========================================================
# VISION LLM INTERPRETER
# =========================================================

class VisionLLMInterpreter:

    def __init__(self, settings):

        self.settings = settings

    # -----------------------------------------------------
    # IMAGE INTERPRETATION
    # -----------------------------------------------------

    def interpret_image(self, prediction):

        prompt_payload = {

            "tipo_resultado": (
                "modelo_visao_computacional_canina"
            ),

            "racas_provaveis": [

                {
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "dataset": item.get("dataset")
                }

                for item in prediction.get(
                    "dog_breeds",
                    []
                )[:5]
            ],

            "condicoes_dermatologicas_provaveis": [

                {
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "dataset": item.get("dataset")
                }

                for item in prediction.get(
                    "skin_diseases",
                    []
                )[:5]
            ],

            "mensagem_processamento": prediction.get(
                "message"
            )
        }

        return self._interpret(
            prompt_payload=prompt_payload,
            fallback=self._fallback_image(prediction)
        )

    # -----------------------------------------------------
    # INTERPRETATION ORCHESTRATION
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
                    "modelos de visão computacional aplicados "
                    "à triagem veterinária de cães.\n\n"

                    "Responda sempre em português do Brasil.\n\n"

                    "Explique possíveis características "
                    "de raças e condições dermatológicas "
                    "identificadas pelo modelo.\n\n"

                    "REGRAS:\n"

                    "- Não apresente diagnóstico definitivo.\n"
                    "- Não prescreva medicamentos.\n"
                    "- Não recomende tratamentos sem avaliação clínica.\n"
                    "- Não invente informações clínicas.\n"
                    "- Não confunda confiança do modelo com "
                    "probabilidade clínica real.\n"
                    "- Considere as classificações como "
                    "indicadores auxiliares de triagem.\n"
                    "- Recomende avaliação veterinária "
                    "quando houver alterações relevantes.\n\n"

                    "Responda exclusivamente em JSON válido, "
                    "sem markdown, com as seguintes chaves:\n"

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

                    "Explique as principais classificações, "
                    "suas respectivas confianças e as limitações "
                    "da análise visual.\n\n"

                    "Diferencie identificação de raça de "
                    "identificação de possíveis condições "
                    "dermatológicas.\n\n"

                    "Não considere a identificação de uma "
                    "classe como confirmação de doença.\n\n"

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

            if not isinstance(interpreted, dict):

                raise ValueError(
                    "Invalid LLM interpretation structure."
                )

            return self._normalize_interpretation(
                interpretation=interpreted,
                generated_by="llm",
                provider=result["provider"],
                model_used=result["model"]
            )

        except Exception:

            logger.exception(
                "Vision interpretation failed. "
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
                interpretation.get(
                    "summary",
                    "Interpretação visual concluída."
                )
            ).strip(),

            "risk_level": str(
                interpretation.get(
                    "risk_level",
                    "indeterminado"
                )
            ).strip(),

            "explanation": str(
                interpretation.get(
                    "explanation",
                    "Resultado auxiliar de triagem visual."
                )
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

            return [
                str(value).strip()
            ]

        return []

    # -----------------------------------------------------
    # RULE-BASED FALLBACK
    # -----------------------------------------------------

    def _fallback_image(self, prediction):

        skin_diseases = prediction.get(
            "skin_diseases",
            []
        )

        best_skin = (

            skin_diseases[0]

            if skin_diseases

            else None
        )

        best_label = (

            str(best_skin.get("label", "")).strip()

            if best_skin

            else "não identificado"
        )

        best_confidence = (

            safe_float(
                best_skin.get("confidence")
            )

            if best_skin

            else None
        )

        if best_label.lower() == "healthy":

            risk_level = "baixo"

            summary = (
                "O modelo de imagem classificou a "
                "pele como aparentemente saudável "
                "entre as classes avaliadas."
            )

        elif best_skin:

            risk_level = "indeterminado"

            summary = (
                "O modelo de imagem sinalizou uma "
                "possível condição dermatológica: "
                f"{best_label}."
            )

        else:

            risk_level = "indeterminado"

            summary = (
                "O modelo de imagem não retornou "
                "uma condição dermatológica suficiente "
                "para interpretação."
            )

        confidence_text = (

            f" com confiança de "
            f"{best_confidence * 100:.2f}%"

            if best_confidence is not None

            else ""
        )

        return {

            "summary": summary,

            "risk_level": risk_level,

            "explanation": (

                f"A principal classe dermatológica "
                f"retornada foi '{best_label}'"
                f"{confidence_text}. "

                "O resultado deve ser utilizado "
                "como apoio à triagem visual e "
                "precisa ser confirmado por avaliação "
                "clínica veterinária."
            ),

            "actionable_insights": [

                "Verificar a qualidade da imagem, "
                "iluminação, foco e enquadramento.",

                "Observar a presença de coceira, "
                "descamação, feridas, vermelhidão "
                "ou secreção.",

                "Procurar avaliação veterinária "
                "se houver lesões, desconforto "
                "ou evolução dos sinais."
            ],

            "limitations": [

                "O modelo reconhece apenas as classes "
                "presentes no treinamento.",

                "Imagem, ângulo, iluminação e "
                "qualidade podem alterar as classificações.",

                "A classificação automatizada não "
                "substitui exame dermatológico veterinário."
            ],

            "model_used": "rule-based-fallback",

            "provider": "rules",

            "generated_by": "rules"
        }


# =========================================================
# VISION MODEL SERVICE
# =========================================================

class VisionModelService:

    def __init__(self, settings):

        self.settings = settings

        self.image_model = None

        self.dog_detector = None

        self.labels = None

        self.dataset_map = None

        self.llm_interpreter = VisionLLMInterpreter(
            settings
        )

    # -----------------------------------------------------
    # PRELOAD
    # -----------------------------------------------------

    def preload(self):

        logger.info(
            "Starting vision module initialization..."
        )

        try:

            logger.info(
                "Loading dog detector..."
            )

            self.dog_detector = (
                self.load_dog_detector()
            )

            logger.info(
                "Loading vision model..."
            )

            self.image_model = (
                self.load_image_model()
            )

            logger.info(
                "Loading vision labels..."
            )

            self.labels = self.load_labels()

            logger.info(
                "Loading vision dataset map..."
            )

            self.dataset_map = (
                self.load_dataset_map()
            )

            self._validate_artifacts()

            # ---------------------------------------------
            # MODEL WARMUP
            # ---------------------------------------------

            logger.info(
                "Warming vision model..."
            )

            dummy = np.zeros(
                (1, 160, 160, 3),
                dtype=np.float32
            )

            self.image_model.predict(
                dummy,
                verbose=0
            )

            logger.info(
                "Vision module initialized successfully."
            )

        except Exception:

            self.image_model = None

            self.dog_detector = None

            self.labels = None

            self.dataset_map = None

            logger.exception(
                "Vision module initialization failed."
            )

            raise

    # -----------------------------------------------------
    # ARTIFACT VALIDATION
    # -----------------------------------------------------

    def _validate_artifacts(self):

        if self.image_model is None:

            raise RuntimeError(
                "Vision model is unavailable."
            )

        if self.dog_detector is None:

            raise RuntimeError(
                "Dog detector is unavailable."
            )

        if not isinstance(self.labels, list):

            raise ValueError(
                "Vision labels must be a list."
            )

        if not self.labels:

            raise ValueError(
                "Vision labels cannot be empty."
            )

        if not isinstance(self.dataset_map, dict):

            raise ValueError(
                "Vision dataset map must be a dictionary."
            )

        if not all(
            isinstance(label, str) and label.strip()
            for label in self.labels
        ):

            raise ValueError(
                "Vision labels contain invalid entries."
            )

        if len(set(self.labels)) != len(self.labels):

            raise ValueError(
                "Vision labels contain duplicates."
            )

        missing_labels = [

            label

            for label in self.labels

            if label not in self.dataset_map
        ]

        if missing_labels:

            raise ValueError(
                "Dataset map is missing labels: "
                f"{missing_labels[:10]}"
            )

    # -----------------------------------------------------
    # DOG DETECTOR
    # -----------------------------------------------------

    def load_dog_detector(self):

        return MobileNetV2(
            weights="imagenet",
            include_top=True
        )

    # -----------------------------------------------------
    # IMAGE MODEL
    # -----------------------------------------------------

    def load_image_model(self):

        path = hf_hub_download(
            repo_id=self.settings.hf_image_repo,
            filename=self.settings.hf_image_filename
        )

        return tf.keras.models.load_model(
            path,
            compile=False
        )

    # -----------------------------------------------------
    # LABELS
    # -----------------------------------------------------

    def load_labels(self):

        path = hf_hub_download(
            repo_id=self.settings.hf_image_repo,
            filename="labels.json"
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if not isinstance(data, dict):

            raise ValueError(
                "Invalid labels.json structure."
            )

        return data.get("classes")

    # -----------------------------------------------------
    # DATASET MAP
    # -----------------------------------------------------

    def load_dataset_map(self):

        path = hf_hub_download(
            repo_id=self.settings.hf_image_repo,
            filename="dataset_map.json"
        )

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    # -----------------------------------------------------
    # DOG IMAGE VALIDATION
    # -----------------------------------------------------

    def validate_dog_image(self, image_bytes):

        if self.dog_detector is None:

            raise RuntimeError(
                "Dog detector not loaded."
            )

        with Image.open(
            BytesIO(image_bytes)
        ) as image:

            image = image.convert("RGB")

            image = image.resize(
                (224, 224)
            )

            image_array = (
                tf.keras.utils.img_to_array(image)
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            image_array = preprocess_input(
                image_array
            )

        predictions = self.dog_detector.predict(
            image_array,
            verbose=0
        )

        decoded = decode_predictions(
            predictions,
            top=5
        )[0]

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

            if any(
                keyword in label_lower
                for keyword in dog_keywords
            ):

                return True

        return False

    # -----------------------------------------------------
    # IMAGE PREDICTION
    # -----------------------------------------------------

    def predict_image(self, image_bytes):

        if self.image_model is None:

            raise RuntimeError(
                "Vision model not loaded."
            )

        if self.labels is None:

            raise RuntimeError(
                "Vision labels not loaded."
            )

        if self.dataset_map is None:

            raise RuntimeError(
                "Vision dataset map not loaded."
            )

        # ---------------------------------------------
        # DOG VALIDATION
        # ---------------------------------------------

        if not self.validate_dog_image(
            image_bytes
        ):

            return {

                "success": False,

                "message": (
                    "Apenas imagens de cachorros "
                    "são permitidas."
                ),

                "dog_breeds": [],

                "skin_diseases": [],

                "interpretation": None
            }

        # ---------------------------------------------
        # IMAGE PREPROCESSING
        # ---------------------------------------------

        with Image.open(
            BytesIO(image_bytes)
        ) as image:

            image = image.convert("RGB")

            image = image.resize(
                (160, 160)
            )

            image_array = (
                tf.keras.utils.img_to_array(image)
            )

            image_array = (
                tf.keras.applications.efficientnet_v2
                .preprocess_input(image_array)
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

        # ---------------------------------------------
        # MODEL INFERENCE
        # ---------------------------------------------

        predictions = self.image_model.predict(
            image_array,
            verbose=0
        )

        predictions = np.asarray(
            predictions
        )

        if predictions.ndim != 2:

            raise RuntimeError(
                "Unexpected vision model output dimensions."
            )

        if predictions.shape[0] != 1:

            raise RuntimeError(
                "Unexpected vision model batch size."
            )

        predictions = predictions[0]

        if not np.all(
            np.isfinite(predictions)
        ):

            raise RuntimeError(
                "Vision model returned invalid predictions."
            )

        if len(predictions) != len(self.labels):

            raise RuntimeError(
                "Model output size does not match labels count."
            )

        # ---------------------------------------------
        # SEPARATE PREDICTIONS
        # ---------------------------------------------

        dog_predictions = []

        skin_predictions = []

        for index, score in enumerate(predictions):

            label = self.labels[index]

            dataset = self.dataset_map.get(
                label,
                "Unknown"
            )

            item = {

                "label": label,

                "confidence": float(score),

                "dataset": dataset,

                "reference_image": (

                    f"https://huggingface.co/"
                    f"{self.settings.hf_image_repo}"
                    f"/resolve/main/references/{label}.jpg"
                )
            }

            if dataset in {

                "Stanford-Dogs",

                "Oxford-Pets"

            }:

                dog_predictions.append(item)

            elif dataset == "Kaggle-Skin-Disease":

                skin_predictions.append(item)

        # ---------------------------------------------
        # SORT PREDICTIONS
        # ---------------------------------------------

        dog_predictions.sort(
            key=lambda item: item["confidence"],
            reverse=True
        )

        skin_predictions.sort(
            key=lambda item: item["confidence"],
            reverse=True
        )

        # ---------------------------------------------
        # BUILD RESULT
        # ---------------------------------------------

        result = {

            "success": True,

            "message": (
                "Imagem processada com sucesso."
            ),

            "dog_breeds": dog_predictions[:5],

            "skin_diseases": skin_predictions[:5]
        }

        # ---------------------------------------------
        # LLM INTERPRETATION
        # ---------------------------------------------

        result["interpretation"] = (
            self.llm_interpreter.interpret_image(
                result
            )
        )

        return result


# =========================================================
# SERVICE INSTANCE
# =========================================================

vision_service = VisionModelService(
    settings=settings
)


# =========================================================
# INITIALIZATION
# =========================================================

def initialize_vision():

    vision_service.preload()


# =========================================================
# ROUTES
# =========================================================

@router.post(
    "/predict/image",
    response_model=ImagePredictionResponse,
    dependencies=[
        Depends(require_api_key)
    ]
)
async def predict_image(
    file: UploadFile = File(...)
):

    try:

        # ---------------------------------------------
        # READ IMAGE WITH SIZE LIMIT
        # ---------------------------------------------

        image_bytes = await file.read(
            settings.max_upload_bytes + 1
        )

        # ---------------------------------------------
        # VALIDATE IMAGE
        # ---------------------------------------------

        validate_image_bytes(
            image_bytes
        )

        # ---------------------------------------------
        # RUN INFERENCE OUTSIDE EVENT LOOP
        # ---------------------------------------------

        result = await run_in_threadpool(
            vision_service.predict_image,
            image_bytes
        )

        return result

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            "Vision prediction failed."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro ao processar a imagem."
        )

    finally:

        await file.close()
