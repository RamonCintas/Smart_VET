import json
from pathlib import Path
from types import SimpleNamespace

import langchain_service
from clinical_repository import ClinicalRepository
from langchain_service import LLMInterpreter


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "src" / "back-end" / "fastapi" / "data"


def provider_config(json_mode=True):
    return SimpleNamespace(
        provider="custom",
        api_key="test-secret",
        base_url="https://llm.example/v1",
        model="smart-vet-custom",
        json_mode=json_mode,
    )


def settings(allow_remote_clinical_context, providers=None):
    return SimpleNamespace(
        allow_remote_clinical_context=allow_remote_clinical_context,
        llm_provider_configs=providers or [],
        llm_timeout_seconds=10,
        llm_temperature=0.1,
        llm_max_retries=2,
        llm_retry_backoff_seconds=0,
    )


def prediction():
    return {
        "prediction_binary": 1,
        "prediction_label": "Yes - perigoso",
        "probability_yes": 0.87,
        "input_used": {"AnimalName": "dog", "symptoms1": "fever"},
    }


class FakeResponse:
    def __init__(self, content, status_code=200, headers=None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            response = langchain_service.requests.Response()
            response.status_code = self.status_code
            raise langchain_service.requests.HTTPError(response=response)

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": self.content,
                    }
                }
            ]
        }


def test_langchain_uses_structured_patient_context(monkeypatch):
    requests_sent = []
    response_content = json.dumps(
        {
            "summary": "Paciente com risco elevado segundo o modelo.",
            "risk_level": "alto",
            "explanation": "O resultado deve ser confrontado com o prontuário.",
            "actionable_insights": ["Revisar sinais vitais atuais."],
            "limitations": ["Triagem não substitui avaliação clínica."],
        },
        ensure_ascii=False,
    )

    def fake_post(url, headers, json, timeout):
        requests_sent.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return FakeResponse(response_content)

    monkeypatch.setattr(langchain_service.requests, "post", fake_post)
    interpreter = LLMInterpreter(
        settings(True, [provider_config()]),
        ClinicalRepository(DATA_DIR),
    )

    result, context = interpreter.interpret_condition(prediction(), "VET-001")

    assert result["generated_by"] == "llm"
    assert result["provider"] == "custom"
    assert result["model_used"] == "smart-vet-custom"
    assert context["patient_id"] == "VET-001"
    assert len(requests_sent) == 1
    prompt_content = requests_sent[0]["json"]["messages"][1]["content"]
    assert "VET-001" in prompt_content
    assert "contexto_clinico_estruturado" in prompt_content
    assert "test-secret" not in prompt_content


def test_remote_context_can_be_disabled(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("Remote LLM must not receive clinical context")

    monkeypatch.setattr(langchain_service.requests, "post", fail_if_called)
    interpreter = LLMInterpreter(
        settings(False, [provider_config()]),
        ClinicalRepository(DATA_DIR),
    )

    result, context = interpreter.interpret_condition(prediction(), "VET-001")

    assert result["generated_by"] == "rules"
    assert context["patient_id"] == "VET-001"
    assert "prontuário VET-001" in result["explanation"]


def test_invalid_provider_output_falls_back_to_rules(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return FakeResponse("not valid json")

    monkeypatch.setattr(langchain_service.requests, "post", fake_post)
    interpreter = LLMInterpreter(
        settings(True, [provider_config()]),
        ClinicalRepository(DATA_DIR),
    )

    result, context = interpreter.interpret_condition(prediction(), "VET-001")

    assert result["generated_by"] == "rules"
    assert context["patient_id"] == "VET-001"
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_transient_provider_failure_is_retried(monkeypatch):
    calls = []
    response_content = json.dumps(
        {
            "summary": "Resposta após indisponibilidade temporária.",
            "risk_level": "moderado",
            "explanation": "O provedor respondeu na segunda tentativa.",
            "actionable_insights": [],
            "limitations": [],
        },
        ensure_ascii=False,
    )
    responses = [
        FakeResponse("temporarily unavailable", status_code=503),
        FakeResponse(response_content),
    ]

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        return responses.pop(0)

    monkeypatch.setattr(langchain_service.requests, "post", fake_post)
    monkeypatch.setattr(langchain_service.time, "sleep", lambda _: None)
    interpreter = LLMInterpreter(
        settings(True, [provider_config()]),
        ClinicalRepository(DATA_DIR),
    )

    result, context = interpreter.interpret_condition(prediction(), "VET-001")

    assert result["generated_by"] == "llm"
    assert context["patient_id"] == "VET-001"
    assert len(calls) == 2
