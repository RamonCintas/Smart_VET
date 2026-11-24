"""Core do Smart VET: configuração, LLMs cloud, validação e autenticação.

O Gemma V6 NÃO é carregado neste módulo. A inferência do chat geral é feita
pelo Hugging Face Space através do gradio_client, no agente.py. Este módulo
mantém apenas os LLMs cloud usados na rota com paciente e na segunda validação.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import requests
from fastapi import Header, HTTPException, status
from pydantic_settings import BaseSettings, SettingsConfigDict


# =========================================================
# CAMINHOS E LOGGING
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("smart-vet")


# =========================================================
# PROVEDORES LLM CLOUD
# =========================================================
LLM_PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-5.4",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.8-flash",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct-1M",
    },
}


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    json_mode: bool = True


def normalize_llm_provider(provider: Optional[str]) -> Optional[str]:
    if not provider:
        return None
    normalized = str(provider).strip().lower().replace("_", "-")
    aliases = {
        "google": "gemini",
        "google-gemini": "gemini",
        "hf": "huggingface",
        "hugging-face": "huggingface",
    }
    return aliases.get(normalized, normalized)


def _environment_api_key(provider: Optional[str]) -> Optional[str]:
    """Resolve chaves convencionais quando o campo LLM_* estiver vazio."""
    normalized = normalize_llm_provider(provider)
    env_names = {
        "openai": ("OPENAI_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "huggingface": ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
    }
    for name in env_names.get(normalized, ()):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


# =========================================================
# CONFIGURAÇÕES
# =========================================================
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    api_env: str
    api_key: str
    allowed_origins: str

    # Modelo tabular / Hugging Face
    hf_model_repo: str
    hf_model_filename: str
    hf_data_repo: str
    hf_data_filename: str

    # Visão computacional
    hf_image_repo: str 
    hf_image_filename: str 
    max_upload_bytes: int = 5 * 1024 * 1024

    # LLM cloud principal
    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None
    llm_json_mode: bool = True

    # LLM cloud alternativo
    llm_fallback_1_provider: Optional[str] = None
    llm_fallback_1_api_key: Optional[str] = None
    llm_fallback_1_base_url: Optional[str] = None
    llm_fallback_1_model: Optional[str] = None
    llm_fallback_1_json_mode: bool = True

    # Segundo assistente / validador cloud
    llm_validator_provider: Optional[str] = None
    llm_validator_api_key: Optional[str] = None
    llm_validator_base_url: Optional[str] = None
    llm_validator_model: Optional[str] = None
    llm_validator_json_mode: bool = True
    llm_timeout_seconds: int = 180

    # Credenciais Hugging Face (HF_TOKEN é separado do Basic Auth do Space)
    hf_token: Optional[str] = None

    # Hugging Face Space — Gemma V6 remoto
    smartvet_hf_space_id: str = "ramoncg/smart-vet-gemma-fine-tuning"
    smartvet_hf_space_url: Optional[str] = None
    smartvet_hf_api_name: str = "/gemma_finetune"
    smartvet_space_user: Optional[str] = None
    smartvet_space_password: Optional[str] = None
    gradio_client_timeout_seconds: int = 240
    gradio_client_max_workers: int = 2

    # Compatibilidade com configurações antigas. Não habilitam nem carregam
    # modelo local; mantidas apenas para não quebrar leitores de settings antigos.
    smartvet_ft_enabled: bool = False
    smartvet_ft_preload: bool = False

    @property
    def huggingface_token(self) -> Optional[str]:
        return (
            (self.hf_token or "").strip()
            or os.getenv("HF_TOKEN", "").strip()
            or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
            or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
            or None
        )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def llm_provider_configs(self) -> list[LLMProviderConfig]:
        """Configura os provedores cloud principal e fallback."""
        configs: list[LLMProviderConfig] = []
        primary_provider = normalize_llm_provider(self.llm_provider)
        primary_api_key = self.llm_api_key or _environment_api_key(primary_provider)
        primary_base_url = self.llm_base_url
        primary_model = self.llm_model
        primary_json_mode = self.llm_json_mode

        # Permite usar somente LLM_VALIDATOR_* como provedor cloud caso o
        # usuário não tenha declarado um LLM_PROVIDER separado.
        if not primary_provider and self.llm_validator_provider:
            primary_provider = normalize_llm_provider(self.llm_validator_provider)
            primary_api_key = self.llm_validator_api_key
            primary_base_url = self.llm_validator_base_url
            primary_model = self.llm_validator_model
            primary_json_mode = self.llm_validator_json_mode
        elif not primary_provider and primary_api_key:
            primary_provider = "openai"

        if primary_provider and not primary_api_key:
            primary_api_key = _environment_api_key(primary_provider)

        primary = self._build_llm_provider_config(
            provider=primary_provider,
            api_key=primary_api_key,
            base_url=primary_base_url,
            model=primary_model,
            json_mode=primary_json_mode,
        )
        if primary:
            configs.append(primary)

        fallback_provider = normalize_llm_provider(self.llm_fallback_1_provider)
        fallback_key = self.llm_fallback_1_api_key or _environment_api_key(fallback_provider)
        if (
            not fallback_key
            and fallback_provider
            and fallback_provider == normalize_llm_provider(self.llm_validator_provider)
        ):
            fallback_key = self.llm_validator_api_key or _environment_api_key(fallback_provider)
        fallback = self._build_llm_provider_config(
            provider=fallback_provider,
            api_key=fallback_key,
            base_url=self.llm_fallback_1_base_url,
            model=self.llm_fallback_1_model,
            json_mode=self.llm_fallback_1_json_mode,
        )
        if fallback and all(item.provider != fallback.provider for item in configs):
            configs.append(fallback)
        return configs

    @property
    def llm_validator_provider_configs(self) -> list[LLMProviderConfig]:
        """Seleciona o provedor configurado para a segunda validação cloud."""
        active = self.llm_provider_configs
        if not active:
            return []

        explicit = normalize_llm_provider(self.llm_validator_provider)
        if explicit:
            active_match = next((x for x in active if x.provider == explicit), None)
            if active_match:
                validator = self._build_llm_provider_config(
                    provider=explicit,
                    api_key=(
                        self.llm_validator_api_key
                        or _environment_api_key(explicit)
                        or active_match.api_key
                    ),
                    base_url=self.llm_validator_base_url or active_match.base_url,
                    model=self.llm_validator_model or active_match.model,
                    json_mode=self.llm_validator_json_mode,
                )
                return [validator] if validator else []

            validator = self._build_llm_provider_config(
                provider=explicit,
                api_key=self.llm_validator_api_key or _environment_api_key(explicit),
                base_url=self.llm_validator_base_url,
                model=self.llm_validator_model,
                json_mode=self.llm_validator_json_mode,
            )
            if validator:
                return [validator]
            logger.warning(
                "LLM_VALIDATOR_PROVIDER='%s' sem credenciais válidas; usando provedor cloud ativo.",
                explicit,
            )

        primary = active[0]
        alternate = next((x for x in active[1:] if x.provider != primary.provider), None)
        selected = alternate or primary
        if selected.provider == primary.provider:
            logger.warning(
                "Somente um provedor cloud está ativo ('%s'); ele também será usado na validação. ",
                selected.provider,
            )
        return [selected]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_provider_configs)

    def _build_llm_provider_config(
        self,
        provider: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        model: Optional[str],
        json_mode: bool,
    ) -> Optional[LLMProviderConfig]:
        normalized = normalize_llm_provider(provider)
        if not normalized:
            return None
        defaults = LLM_PROVIDER_DEFAULTS.get(normalized, {})
        resolved_base = (base_url or defaults.get("base_url") or "").strip()
        resolved_model = (model or defaults.get("model") or "").strip()
        resolved_key = (api_key or "").strip()
        if not resolved_base or not resolved_model:
            logger.warning("Provedor LLM '%s' ignorado: URL/modelo ausente.", normalized)
            return None
        if not resolved_key:
            logger.warning("Provedor LLM '%s' ignorado: API key ausente.", normalized)
            return None
        return LLMProviderConfig(
            provider=normalized,
            api_key=resolved_key,
            base_url=resolved_base.rstrip("/"),
            model=resolved_model,
            json_mode=bool(json_mode),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


# =========================================================
# DIAGNÓSTICOS
# =========================================================
def llm_diagnostics() -> dict[str, Any]:
    providers = settings.llm_provider_configs
    validators = settings.llm_validator_provider_configs
    return {
        "enabled": settings.llm_enabled,
        "configured_providers": [
            {
                "provider": item.provider,
                "model": item.model,
                "base_url": item.base_url,
                "json_mode": item.json_mode,
                "api_key_configured": bool(item.api_key),
            }
            for item in providers
        ],
        "validator_providers": [
            {
                "provider": item.provider,
                "model": item.model,
                "api_key_configured": bool(item.api_key),
            }
            for item in validators
        ],
        "validator_uses_dedicated_provider": bool(validators),
        "provider_count": len(providers),
        "local_gemma_fallback": {
            "enabled": False,
            "loaded": False,
            "reason": "A inferência Gemma V6 é executada remotamente no Hugging Face Space.",
        },
    }


def finetuned_llm_diagnostics() -> dict[str, Any]:
    """Nome legado mantido para compatibilidade; descreve o Space remoto."""
    return remote_gemma_diagnostics()


def remote_gemma_diagnostics() -> dict[str, Any]:
    space_id = (settings.smartvet_hf_space_id or "").strip()
    space_url = (settings.smartvet_hf_space_url or "").strip()
    api_name = (settings.smartvet_hf_api_name or "/gemma_finetune").strip()
    if api_name and not api_name.startswith("/"):
        api_name = "/" + api_name
    username = (settings.smartvet_space_user or os.getenv("SMARTVET_SPACE_USER", "")).strip()
    password = settings.smartvet_space_password or os.getenv("SMARTVET_SPACE_PASSWORD", "")
    token = settings.huggingface_token
    return {
        "enabled": bool(space_id or space_url),
        "execution_mode": "huggingface_space_remote",
        "space_id": space_id or None,
        "space_url_configured": bool(space_url),
        "api_name": api_name,
        "task_id": "free_qa",
        "username_configured": bool(username),
        "password_configured": bool(password),
        "basic_auth_configured": bool(username and password),
        "hf_token_configured": bool(token),
        "local_model_loaded": False,
        "local_model_download_enabled": False,
        "note": (
            "O FastAPI chama o endpoint /gemma_finetune do Space; "
            "nenhum peso Gemma é baixado ou carregado localmente."
        ),
    }


# =========================================================
# AUTENTICAÇÃO COMPARTILHADA
# =========================================================
def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    configured_key = (settings.api_key or os.getenv("API_KEY", "")).strip()
    environment = (settings.api_env or "dev").strip().lower()
    if not configured_key and environment in {"prod", "production"}:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_KEY não configurada no ambiente de produção.",
        )
    if not configured_key:
        return
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not hmac.compare_digest(str(x_api_key), configured_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key")


def require_reviewer_key(x_reviewer_key: Optional[str] = Header(default=None)) -> None:
    configured_key = os.getenv("SMART_VET_REVIEW_API_KEY", "").strip()
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Revisão humana não configurada: defina SMART_VET_REVIEW_API_KEY no backend.",
        )
    if not x_reviewer_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credencial de revisor necessária.")
    if not hmac.compare_digest(str(x_reviewer_key), configured_key):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Credencial de revisor inválida.")


# =========================================================
# CLIENTE LLM OPENAI-COMPATIBLE — SOMENTE CLOUD
# =========================================================
class LLMClient:
    def __init__(self, client_settings: Any, allow_local_fallback: bool = False):
        self.settings = client_settings
        # Parâmetro legado aceito para não quebrar integrações antigas, mas
        # deliberadamente nunca habilita fallback local.
        self.allow_local_fallback = False
        if allow_local_fallback:
            logger.warning("allow_local_fallback foi ignorado: Gemma local está desativado.")
        self.session = requests.Session()

    def generate(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages deve conter pelo menos uma mensagem.")
        providers = self.settings.llm_provider_configs
        errors: list[str] = []
        if not providers:
            raise RuntimeError("Nenhum LLM cloud configurado (LLM_PROVIDER/API key).")

        for index, provider in enumerate(providers):
            try:
                content = self._call_provider(
                    messages=messages,
                    provider=provider,
                    json_mode=bool(json_mode and provider.json_mode),
                )
                used_fallback = index > 0
                return {
                    "content": content,
                    "provider": provider.provider,
                    "model": provider.model,
                    "fallback_used": used_fallback,
                    "provider_fallback_used": used_fallback,
                    "fallback_reason": (
                        f"O provedor cloud principal falhou; foi utilizado '{provider.provider}'."
                        if used_fallback else None
                    ),
                }
            except Exception as exc:
                errors.append(
                    f"{provider.provider} ({provider.model}): {type(exc).__name__}: {str(exc)[:700]}"
                )
                logger.exception("Falha no LLM cloud '%s'.", provider.provider)
        raise RuntimeError("Todos os provedores cloud configurados falharam: " + " | ".join(errors))

    def _call_provider(
        self,
        messages: list[dict[str, str]],
        provider: LLMProviderConfig,
        json_mode: bool,
    ) -> str:
        """Executa uma chamada OpenAI-compatible no provedor cloud.

        O objeto `response` é sempre criado pelo POST antes de qualquer
        leitura de status ou corpo. Gemini recebe payload sem parâmetros
        opcionais de amostragem para maior compatibilidade; Hugging Face
        Router e OpenAI recebem somente os campos essenciais.
        """
        attempts = [True, False] if json_mode else [False]
        errors: list[str] = []
        timeout = max(5, int(self.settings.llm_timeout_seconds))
        endpoint = f"{provider.base_url.rstrip('/')}/chat/completions"

        for use_json_mode in attempts:
            headers = {
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            payload: dict[str, Any] = {
                "model": provider.model,
                "messages": messages,
            }
            if use_json_mode:
                payload["response_format"] = {"type": "json_object"}

            response: Optional[requests.Response] = None
            try:
                # A resposta é criada aqui, antes de qualquer acesso a .ok,
                # .status_code, .text ou .json(). Isso evita o NameError
                # observado quando versões antigas/incompletas de core.py
                # acessavam response sem executar o POST.
                response = self.session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(10, timeout),
                )
            except requests.RequestException as exc:
                message = (
                    f"Falha HTTP ao consultar {provider.provider}: "
                    f"{type(exc).__name__}: {str(exc)[:500]}"
                )
                errors.append(message)
                logger.warning("%s", message)
                # Falha de conexão não é corrigida removendo response_format.
                break

            if response is None:
                errors.append(
                    f"O provedor {provider.provider} não retornou um objeto Response."
                )
                break

            if not response.ok:
                detail = (response.text or "").strip()[:1200]
                error_message = (
                    f"HTTP {response.status_code} de {provider.provider}: {detail}"
                )
                errors.append(
                    f"JSON mode={use_json_mode}: {error_message}"
                )
                logger.warning(
                    "Provedor '%s' respondeu HTTP %s (JSON mode=%s): %s",
                    provider.provider,
                    response.status_code,
                    use_json_mode,
                    detail[:500],
                )
                # Alguns provedores não implementam response_format. Nesse
                # caso, repetir sem JSON mode é uma alternativa válida.
                if use_json_mode and response.status_code in {400, 422}:
                    continue
                raise RuntimeError(error_message)

            try:
                data = response.json()
            except ValueError as exc:
                errors.append(
                    f"Resposta não JSON de {provider.provider}: {str(exc)[:300]}"
                )
                break

            if not isinstance(data, dict):
                errors.append(
                    f"Resposta inválida de {provider.provider}: esperado objeto JSON."
                )
                break

            choices = data.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                error_info = data.get("error") or data.get("detail") or data
                errors.append(
                    f"Resposta sem choices válidos de {provider.provider}: "
                    f"{str(error_info)[:700]}"
                )
                break

            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, list):
                # Compatibilidade com respostas multimodais/partes de texto.
                content = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type", "text") == "text"
                )
            content = str(content or "").strip()
            if not content:
                errors.append(
                    f"O provedor {provider.provider} retornou conteúdo vazio."
                )
                break

            if use_json_mode:
                try:
                    self.parse_json(content)
                except ValueError as exc:
                    errors.append(
                        f"Saída não respeitou JSON mode em {provider.provider}: "
                        f"{str(exc)[:400]}"
                    )
                    # A segunda tentativa sem response_format permite que o
                    # cliente tente recuperar uma saída textual estruturada.
                    continue

            return content

        raise RuntimeError(
            "Todas as tentativas de chamada falharam: " + " | ".join(errors)
        )

    @staticmethod
    def parse_json(content: Any) -> dict[str, Any]:
        if isinstance(content, dict):
            return content
        if not content:
            raise ValueError("Conteúdo JSON vazio.")
        text = str(content).strip()
        candidates = [text]
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
        if match:
            candidates.append(match.group(1).strip())
        first_brace, last_brace = text.find("{"), text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidates.append(text[first_brace:last_brace + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        raise ValueError("Não foi possível interpretar a resposta como objeto JSON.")


llm_client = LLMClient(settings, allow_local_fallback=False)


class _ValidatorSettings:
    def __init__(self, base_settings: Settings):
        dedicated = base_settings.llm_validator_provider_configs
        self.llm_provider_configs = dedicated or base_settings.llm_provider_configs
        self.llm_timeout_seconds = base_settings.llm_timeout_seconds
        self.uses_dedicated_provider = bool(dedicated)


validator_llm_client = LLMClient(_ValidatorSettings(settings), allow_local_fallback=False)


def validate_answer_with_llm(
    question: str,
    draft_response: str,
    route: Optional[str] = None,
    sources: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Segunda leitura cloud; risco concreto é avaliado pelo middleware principal.

    Uma indisponibilidade do validador não bloqueia automaticamente toda conversa:
    os guardrails determinísticos continuam ativos e o main.py decide retenção
    somente por sinais concretos de risco.
    """
    safe_sources = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        safe_sources.append({
            key: source.get(key)
            for key in (
                "filename", "source", "document", "title", "page",
                "page_number", "chunk_id", "source_type",
            )
            if source.get(key) is not None
        })
    system_prompt = (
        "Você é o segundo assistente de validação clínica do Smart VET. "
        "Analise a resposta proposta para uso por médico-veterinário. "
        "Não faça diagnóstico, não prescreva e não invente evidências. "
        "Verifique linguagem de certeza, doses/posologias, dados pessoais, "
        "afirmações clínicas sem suporte e citações inexistentes. A rota "
        "general_gemma_v6_huggingface_space é pergunta livre (free_qa): nela "
        "não se exige fonte documental/RAG; avalie conhecimento geral com "
        "prudência e não marque ausência de fonte como risco. Na rota "
        "patient_context_llm, afirmações sobre o prontuário precisam de fontes "
        "recuperadas correspondentes. Responda SOMENTE um objeto JSON com as "
        "chaves: status (approved ou needs_human_review), risk_level "
        "(low/medium/high), notes (string curta), issues (lista de strings), "
        "source_support (supported/partial/insufficient/not_applicable). "
        "Use needs_human_review somente diante de um risco concreto relevante, "
        "como prescrição/dose individualizada, diagnóstico definitivo, vazamento "
        "de dados pessoais, contradição clínica importante, citação inventada ou "
        "afirmação sobre prontuário sem evidência. Uma dúvida leve ou a ausência "
        "de documentos na rota geral não exige revisão humana."
    )
    user_payload = {
        "question": str(question or "")[:4000],
        "draft_response": str(draft_response or "")[:12000],
        "route": route,
        "sources": safe_sources,
    }
    generated = validator_llm_client.generate(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        json_mode=True,
    )
    parsed = LLMClient.parse_json(generated.get("content", ""))
    status_value = str(parsed.get("status", "needs_human_review")).lower()
    if status_value not in {"approved", "needs_human_review"}:
        status_value = "needs_human_review"
    risk_level = str(parsed.get("risk_level", "medium")).lower()
    if risk_level not in {"low", "medium", "high"}:
        risk_level = "medium"
    issues = parsed.get("issues", [])
    if not isinstance(issues, list):
        issues = [str(issues)]
    return {
        "status": status_value,
        "risk_level": risk_level,
        "notes": str(parsed.get("notes") or "Validação concluída."),
        "issues": [str(item)[:300] for item in issues[:20]],
        "source_support": str(parsed.get("source_support") or "not_applicable"),
        "validator_provider": generated.get("provider"),
        "validator_model": generated.get("model"),
        "validator_mode": "llm_second_pass",
        "uses_dedicated_provider": bool(
            getattr(validator_llm_client.settings, "uses_dedicated_provider", False)
        ),
    }


# =========================================================
# CLIENTE LOCAL LEGADO — DESATIVADO, SEM IMPORTAR TORCH/TRANSFORMERS
# =========================================================
class HuggingFaceFineTunedClient:
    """Stub de compatibilidade: o Gemma é exclusivamente remoto no Space."""
    def __init__(self, client_settings: Settings):
        self.settings = client_settings

    def status(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "loaded": False,
            "status": "disabled_remote_only",
            "provider": "huggingface_spaces_gradio",
            "local_model_download_enabled": False,
            "message": "O Gemma V6 é executado remotamente no Hugging Face Space.",
        }

    def preload(self) -> dict[str, Any]:
        return self.status()

    def load(self) -> None:
        raise RuntimeError("Carregamento local do Gemma foi desativado. Utilize o Hugging Face Space.")

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("Inferência local do Gemma foi desativada. Utilize o endpoint remoto /gemma_finetune.")


# Compatibilidade temporária para módulos antigos que ainda consultem status().
peteval_llm_client = HuggingFaceFineTunedClient(settings)


__all__ = [
    "BASE_DIR", "ENV_FILE", "logger", "LLMProviderConfig", "Settings", "settings",
    "get_settings", "llm_diagnostics", "finetuned_llm_diagnostics",
    "remote_gemma_diagnostics", "normalize_llm_provider", "require_api_key",
    "require_reviewer_key", "LLMClient", "llm_client", "validator_llm_client",
    "validate_answer_with_llm", "HuggingFaceFineTunedClient", "peteval_llm_client",
]

