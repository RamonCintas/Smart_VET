"""Painel Streamlit de saúde, auditoria e revisão humana do Smart VET.

Este módulo é somente frontend. Consulta a API FastAPI por HTTP; não abre
SQLite, não acessa ChromaDB diretamente e não carrega modelos.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:
    pass

try:
    REQUEST_TIMEOUT = max(5, int(os.getenv("STREAMLIT_API_TIMEOUT", "45")))
except (TypeError, ValueError):
    REQUEST_TIMEOUT = 45


def build_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def display_text(value: Any) -> str:
    if value is None or value == "":
        return "Não informado"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def normalize_options(values: Any, field_name: str) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"O campo '{field_name}' não é uma lista.")
    cleaned = {
        value.strip() for value in values
        if isinstance(value, str) and value.strip() and value.strip().lower() not in {"nan", "none", "null"}
    }
    return sorted(cleaned, key=str.lower)


def validate_json_response(response: requests.Response, endpoint: str) -> Any:
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise ValueError(f"O endpoint {endpoint} não retornou JSON válido.") from exc


def fetch_endpoint(
    api_url: str,
    endpoint: str,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, Any]] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[Any, Optional[str]]:
    try:
        response = requests.get(
            build_url(api_url, endpoint), headers=headers or {},
            params=params or {}, timeout=timeout,
        )
        return validate_json_response(response, endpoint), None
    except requests.exceptions.ConnectionError:
        return {}, "Não foi possível conectar à FastAPI."
    except requests.exceptions.Timeout:
        return {}, f"O endpoint {endpoint} excedeu o tempo limite."
    except requests.exceptions.HTTPError as exc:
        try:
            payload = response.json()
            detail = payload.get("detail", "") if isinstance(payload, dict) else ""
        except ValueError:
            detail = ""
        return {}, f"Erro HTTP no endpoint {endpoint}: {detail or exc}"
    except requests.exceptions.RequestException as exc:
        return {}, f"Erro de comunicação com {endpoint}: {type(exc).__name__}"
    except ValueError as exc:
        return {}, str(exc)


def get_endpoint_data(api_url: str, endpoint: str, api_key: str, params_json: str = "{}") -> tuple[Any, Optional[str]]:
    """Compatibilidade com chamadas legadas do painel."""
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        params = json.loads(params_json or "{}")
        if not isinstance(params, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        return {}, "Parâmetros de consulta inválidos."
    return fetch_endpoint(api_url, endpoint, headers, params)


# ---------------------------------------------------------------------------
# HEALTH / SERVIÇOS
# ---------------------------------------------------------------------------
def get_api_health(api_url: str, headers: dict[str, str]) -> tuple[dict, Optional[str]]:
    data, error = fetch_endpoint(api_url, "/health", headers)
    return as_dict(data), error


def get_tabular_health(api_url: str, headers: dict[str, str]) -> tuple[dict, Optional[str]]:
    data, error = fetch_endpoint(api_url, "/health/tabular", headers)
    return as_dict(data), error


def get_rag_health(api_url: str, headers: dict[str, str]) -> tuple[dict, Optional[str]]:
    data, error = fetch_endpoint(api_url, "/rag/status", headers)
    return as_dict(data), error


def get_readiness(api_url: str, headers: dict[str, str]) -> tuple[dict, Optional[str]]:
    data, error = fetch_endpoint(api_url, "/health/ready", headers)
    return as_dict(data), error


def get_llm_status(api_url: str, headers: dict[str, str]) -> tuple[dict, Optional[str]]:
    data, error = fetch_endpoint(api_url, "/llm/status", headers)
    return as_dict(data), error


def load_tabular_options(api_url: str, headers: dict[str, str]) -> tuple[list[str], list[str], dict, Optional[str]]:
    health, error = get_tabular_health(api_url, headers)
    if error:
        return [], [], {}, error
    try:
        return (
            normalize_options(health.get("animals", []), "animals"),
            normalize_options(health.get("symptoms", []), "symptoms"),
            health,
            None,
        )
    except ValueError as exc:
        return [], [], health, str(exc)


# ---------------------------------------------------------------------------
# AUDITORIA E REVISÃO HUMANA — API FastAPI /audit
# ---------------------------------------------------------------------------
def get_audit_summary(api_url: str, headers: dict[str, str]) -> tuple[dict, Optional[str]]:
    data, error = fetch_endpoint(api_url, "/audit/summary", headers)
    return as_dict(data), error


def get_audit_logs(api_url: str, headers: dict[str, str], limit: int = 50, request_id: Optional[str] = None) -> tuple[dict, Optional[str]]:
    params: dict[str, Any] = {"limit": max(1, min(int(limit), 500))}
    if request_id:
        params["request_id"] = request_id
    data, error = fetch_endpoint(api_url, "/audit/logs", headers, params=params)
    return as_dict(data), error


def _secret_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


def _get_reviewer_key() -> str:
    # Nome canônico compartilhado pelo core.py/painel.py do FastAPI; alias antigo aceito.
    return (
        _secret_value("SMART_VET_REVIEW_API_KEY")
        or _secret_value("SMARTVET_REVIEWER_KEY")
    )


def get_pending_reviews(api_url: str, headers: dict[str, str], reviewer_key: str) -> tuple[list[dict], Optional[str]]:
    request_headers = {**(headers or {}), "X-Reviewer-Key": reviewer_key}
    try:
        response = requests.get(build_url(api_url, "/audit/reviews/pending"), headers=request_headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError("A fila de revisão não retornou uma lista válida.")
        return [as_dict(item) for item in data], None
    except requests.exceptions.HTTPError as exc:
        try:
            payload = response.json()
            detail = payload.get("detail", "") if isinstance(payload, dict) else ""
        except ValueError:
            detail = ""
        return [], f"A API recusou a consulta da fila: {detail or exc}"
    except requests.exceptions.Timeout:
        return [], "A consulta da fila de revisão excedeu o tempo limite."
    except requests.exceptions.RequestException as exc:
        return [], f"Não foi possível consultar a fila: {type(exc).__name__}"
    except ValueError as exc:
        return [], str(exc)


def submit_review_decision(
    api_url: str,
    headers: dict[str, str],
    reviewer_key: str,
    review_id: str,
    action: str,
    payload: dict[str, Any],
) -> Optional[str]:
    if action not in {"approve", "reject"}:
        return "Decisão de revisão inválida."
    request_headers = {**(headers or {}), "X-Reviewer-Key": reviewer_key}
    try:
        response = requests.post(
            build_url(api_url, f"/audit/reviews/{review_id}/{action}"),
            headers=request_headers, json=payload, timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return None
    except requests.exceptions.HTTPError as exc:
        try:
            data = response.json()
            detail = data.get("detail", "") if isinstance(data, dict) else ""
        except ValueError:
            detail = ""
        return f"A API recusou a decisão: {detail or exc}"
    except requests.exceptions.Timeout:
        return "A decisão excedeu o tempo limite. Verifique se foi registrada antes de repetir."
    except requests.exceptions.RequestException as exc:
        return f"Falha ao enviar decisão: {type(exc).__name__}"


def render_human_review_section(api_url: str, headers: dict[str, str]) -> None:
    st.markdown("#### 👨‍⚕️👩‍⚕️ Fila de revisão humana")
    reviewer_key = _get_reviewer_key()
    if not reviewer_key:
        reviewer_key = st.text_input(
            "Chave do revisor autorizado", type="password",
            key="smartvet_dashboard_reviewer_key",
            help="Configure SMART_VET_REVIEW_API_KEY em secrets ou no ambiente do Streamlit.",
        ).strip()
    if not reviewer_key:
        st.info("A fila é protegida. Informe a credencial do revisor para consultar pendências.")
        return

    reviews, error = get_pending_reviews(api_url, headers, reviewer_key)
    if error:
        st.warning(error)
        return
    if not reviews:
        st.success("Não existem respostas aguardando revisão humana.")
        return

    st.caption(f"{len(reviews)} revisão(ões) aguardando decisão.")
    reviewer_id = st.text_input(
        "Identificação profissional / CRMV", key="smartvet_dashboard_reviewer_id",
        placeholder="CRMV-UF-12345",
    ).strip()

    for review in reviews:
        review_id = str(review.get("review_id") or "").strip()
        created = review.get("created_at") or "data não informada"
        with st.expander(f"Revisão {review_id or 'sem identificador'} · {created}"):
            st.write(f"**Solicitação:** `{review.get('request_id') or 'N/A'}`")
            if review.get("question_text"):
                st.markdown("**Pergunta original**")
                st.write(review["question_text"])
            st.write(f"**Rota:** `{review.get('route') or 'Não informada'}`")
            st.write(f"**Modelo:** `{review.get('model') or 'Não informado'}`")
            if review.get("validation"):
                with st.expander("Resultado da validação"):
                    st.json(review["validation"], expanded=False)
            if review.get("sources"):
                with st.expander("Fontes registradas"):
                    st.json(review["sources"], expanded=False)
            draft = st.text_area(
                "Resposta sugerida (revise antes de aprovar)",
                value=review.get("draft_response") or "",
                key=f"smartvet_review_draft_{review_id}", height=180,
            )
            comment = st.text_area("Comentário do revisor", key=f"smartvet_review_comment_{review_id}", height=80)
            approve_col, reject_col = st.columns(2)
            with approve_col:
                approve = st.button(
                    "✅ Aprovar resposta", key=f"smartvet_approve_{review_id}",
                    use_container_width=True, disabled=not bool(reviewer_id and review_id and draft.strip()),
                )
            with reject_col:
                reject = st.button(
                    "⛔ Rejeitar resposta", key=f"smartvet_reject_{review_id}",
                    use_container_width=True, disabled=not bool(reviewer_id and review_id and comment.strip()),
                )
            if approve:
                error = submit_review_decision(
                    api_url, headers, reviewer_key, review_id, "approve",
                    {"reviewer_id": reviewer_id, "comment": comment.strip() or None, "final_response": draft},
                )
                if error:
                    st.error(error)
                else:
                    st.success("Revisão aprovada e registrada.")
                    st.rerun()
            if reject:
                error = submit_review_decision(
                    api_url, headers, reviewer_key, review_id, "reject",
                    {"reviewer_id": reviewer_id, "comment": comment.strip()},
                )
                if error:
                    st.error(error)
                else:
                    st.success("Revisão rejeitada e registrada.")
                    st.rerun()


# ---------------------------------------------------------------------------
# APRESENTAÇÃO DO PAINEL
# ---------------------------------------------------------------------------
def render_status_indicator(label: str, value: Any) -> None:
    text = str(value or "Não informado")
    normalized = text.strip().lower()
    positive = {"online", "ready", "initialized", "configured", "ok", "true", "success", "carregado", "habilitada", "habilitado", "configurado", "configurada", "encontrado"}
    negative = {"offline", "error", "failed", "false", "not_initialized", "not_configured", "unavailable", "não encontrado", "desabilitada", "desabilitado", "não configurado"}
    if normalized in positive:
        st.success(f"🟢 {label}: {text}")
    elif normalized in negative:
        st.error(f"🔴 {label}: {text}")
    else:
        st.info(f"🔵 {label}: {text}")


def safe_metric(label: str, value: Any) -> None:
    st.metric(label, "N/A" if value is None else value)


def display_value(label: str, value: Any) -> None:
    st.write(f"**{label}:** {display_text(value)}")


def render_api_section(health: dict, readiness: dict, api_url: str) -> None:
    st.subheader("🌐 Saúde da aplicação")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_status_indicator("API", health.get("status", "Indisponível"))
    with c2:
        render_status_indicator("Ambiente", health.get("environment", "Não informado"))
    with c3:
        render_status_indicator("Prontidão", readiness.get("status", "Não informado"))
    st.caption(f"FastAPI: {api_url} · versão {health.get('version', 'não informada')}")
    modules = as_dict(health.get("modules"))
    if modules:
        st.markdown("#### Módulos")
        columns = st.columns(max(1, len(modules)))
        for index, (name, module_status) in enumerate(modules.items()):
            with columns[index]:
                render_status_indicator(str(name).replace("_", " ").capitalize(), module_status)
    if readiness.get("status") != "ready":
        st.warning("A API respondeu, mas nem todos os serviços informaram prontidão.")
    if readiness.get("note"):
        st.caption(readiness["note"])

def render_tabular_section(tabular_health: dict, tabular_error: Optional[str]) -> None:
    st.subheader("🧪 Dataset tabular")
    if tabular_error:
        st.warning(tabular_error)
        return
    if not tabular_health:
        st.warning("O endpoint tabular não retornou dados.")
        return
    animals = as_list(tabular_health.get("animals"))
    symptoms = as_list(tabular_health.get("symptoms"))
    
    c1, c2 = st.columns(2)
    with c1:
        with st.expander(f"Animais disponíveis ({len(animals)})"):
            for item in animals:
                st.write(item)
    with c2:
        with st.expander(f"Sintomas disponíveis ({len(symptoms)})"):
            for item in symptoms:
                st.write(item)
    display_value("Dataset", tabular_health.get("dataset"))
    if animals and symptoms:
        st.success("Dataset tabular carregado corretamente.")
    else:
        st.warning("O dataset respondeu, mas não disponibilizou todas as opções esperadas.")

    st.divider()

def render_models_section(health: dict, llm_status: Optional[dict] = None) -> None:
    st.subheader("🏗️ Modelos e provedores de inteligência artificial")
    models = as_dict(health.get("models"))
    llm = as_dict(health.get("llm"))
    llm_status = as_dict(llm_status)

    st.markdown("#### 🩺 Modelo tabular")
    c1, c2 = st.columns(2)
    with c1:
        display_value("Modelo", models.get("tabular_model"))  
    with c2:
        display_value("Arquivo", models.get("tabular_model_file"))

    st.markdown("#### 🔬 Modelo de visão computacional")
    c1, c2 = st.columns(2)
    with c1:
        display_value("Repositório", models.get("image_model"))
    with c2:
        display_value("Arquivo", models.get("image_model_file"))

    st.markdown("#### ☁️ LLM cloud - rota com paciente e validação")

    providers = as_list(
        llm.get("configured_providers")
        or llm_status.get("configured_providers")
    )

    validator = as_dict(llm.get("validator"))

    primary = as_dict(llm.get("primary"))
    if not primary:
        primary = providers[0] if providers else {}

    if providers:
        base_provider_count = len(providers)
    else:
        base_provider_count = int(
            llm.get("provider_count")
            or llm_status.get("provider_count")
            or 0
        )

    active_provider_count = base_provider_count + (1 if validator else 0)

    c1, c2, c3 = st.columns(3)

    with c1:
        safe_metric(
            "Provedor principal",
            primary.get("provider")
            or llm_status.get("primary_provider")
            or "N/A"
        )

    with c2:
        safe_metric(
            "Modelo",
            primary.get("model") or "N/A"
        )

    with c3:
        safe_metric(
            "Provedores ativos",
            active_provider_count
        )

    render_status_indicator(
        "LLM cloud",
        "Configurado"
        if llm.get("enabled") or llm_status.get("status") == "ok"
        else "Não configurado"
    )

    # Provedores configurados
    if providers:
        with st.expander("Provedores LLM configurados"):
            for item in providers:
                provider = as_dict(item)

                st.markdown(
                    f"**{provider.get('provider', 'Não informado')}**"
                )

                display_value("Modelo", provider.get("model"))

                if provider.get("base_url"):
                    display_value("Base URL", provider.get("base_url"))

                if provider.get("api_key_configured") is not None:
                    render_status_indicator(
                        "Credencial",
                        "Configurada"
                        if provider["api_key_configured"]
                        else "Não configurada"
                    )

                st.divider()

    if validator:
        st.markdown("**Segundo validador cloud**")

        display_value("Provedor", validator.get("provider"))
        display_value("Modelo", validator.get("model"))

        render_status_indicator(
            "Provedor dedicado",
            "Configurado"
            if validator.get("uses_dedicated_provider")
            else "Compartilhado"
        )
    st.divider()

    st.markdown("#### 🧠 Gemma Fine-Tuning - Hugging Face Space remoto")
    remote = as_dict(health.get("gemma_v6_remote"))
    if not remote:
        remote = as_dict(as_dict(health.get("fine_tuned_model")))
    if not remote:
        remote = as_dict(llm_status.get("gemma_v6_remote"))
    if not remote:
        st.warning("A API não retornou o diagnóstico do Hugging Face Space.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        render_status_indicator("Configuração", "Ok" if remote.get("ready_to_connect") else "Incompleta")
    with c2:
        render_status_indicator("Usuário Basic Auth", "Configurado" if remote.get("username_configured") else "Ausente")
    with c3:
        render_status_indicator("Senha Basic Auth", "Configurada" if remote.get("password_configured") else "Ausente")
    c1, c2 = st.columns(2)
    with c1:
        display_value("Space", remote.get("space_id") or ("URL configurada" if remote.get("space_url_configured") else None))
        display_value("Endpoint Gradio", remote.get("api_name") or "/gemma_finetune")
    with c2:
        render_status_indicator("HF_TOKEN", "Configurado" if remote.get("hf_token_configured") else "Não configurado (opcional para Space público)")
    st.caption("(Login de teste do Space está no readme do projeto)")
    if remote.get("note"):
        st.caption(remote["note"])
    with st.expander("Diagnóstico técnico do Space"):
        # Não exibir senha, token ou valor de credenciais, apenas indicadores booleanos.
        safe_remote = {k: v for k, v in remote.items() if not any(secret in str(k).lower() for secret in ("password", "token_value", "secret", "api_key_value"))}
        st.json(safe_remote, expanded=False)


def render_rag_section(rag: dict, rag_error: Optional[str]) -> None:
    st.subheader("📀 RAG e ChromaDB")
    if rag_error:
        st.warning(rag_error)
        return
    if not rag:
        st.warning("A API não retornou informações sobre o RAG.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        safe_metric("Chunks indexados", rag.get("indexed_chunks", 0))
    with c2:
        safe_metric("Pacientes", rag.get("patients", 0))
    with c3:
        safe_metric("Dimensão", rag.get("embedding_dimension", "N/A"))

    st.markdown("#### 🔢 Gerenciamento de Embeddings")

    # =========================================================
    # 1. DADOS DOS PROVEDORES
    # =========================================================

    primary_provider = rag.get("embedding_primary_provider") or "N/A"
    primary_model = rag.get("embedding_primary_model") or "N/A"

    fallback_provider = rag.get("embedding_fallback_provider") or "N/A"
    fallback_model = rag.get("embedding_fallback_model") or "N/A"

    active_provider = rag.get("embedding_provider") or "N/A"
    active_model = rag.get("embedding_model") or "N/A"

    primary_key = bool(
        rag.get("embedding_primary_key_configured")
    )

    fallback_key = bool(
        rag.get("embedding_fallback_key_configured")
    )

    # =========================================================
    # 2. MODELO PRINCIPAL E FALLBACK - LINHA SUPERIOR
    # =========================================================

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("🎯 **MODELO PRINCIPAL**")

            st.markdown(f"### {primary_provider.upper()}")

            st.caption("Modelo de embedding")

            st.code(primary_model, language=None)

            render_status_indicator(
                "Chave principal",
                "Configurada" if primary_key else "Não configurada"
            )

    with c2:
        with st.container(border=True):
            st.markdown("⚙️ **MODELO FALLBACK**")

            st.markdown(f"### {fallback_provider.upper()}")

            st.caption("Modelo de embedding reserva")

            st.code(fallback_model, language=None)

            render_status_indicator(
                "Chave fallback",
                "Configurada" if fallback_key else "Não configurada"
            )

    # =========================================================
    # 3. MODELO ATIVO - LINHA INFERIOR
    # =========================================================

    st.markdown(" ")

    with st.container(border=True):

        st.markdown("⚡ **MODELO DE EMBEDDING ATIVO**")

        st.markdown(f"## {active_provider.upper()}")

        st.caption("Modelo atualmente utilizado pelo ChromaDB")

        st.code(active_model, language=None)

        # Identifica a origem do modelo ativo
        if active_provider.lower() == primary_provider.lower():
            st.success(
                "O provedor principal está em utilização."
            )

        elif active_provider.lower() == fallback_provider.lower():
            st.warning(
                "O provedor fallback está em utilização."
            )

        else:
            st.info(
                "O provedor ativo é diferente dos provedores principal e fallback."
            )
    
        

    st.markdown("#### 📜 Banco Vetorial ChromaDB")
    display_value("Coleção", rag.get("collection"))
    display_value("Diretório", rag.get("chroma_directory"))
    display_value("Arquivo SQLite", rag.get("chroma_database_file"))
    if rag.get("chroma_directory_exists") and rag.get("persistence_enabled"):
        st.success("Persistência do ChromaDB configurada.")
    else:
        st.warning("A persistência do ChromaDB não foi confirmada.")
    st.divider()
    st.markdown("#### 🗂️ Prontuários")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_status_indicator("ZIP", "Encontrado" if rag.get("zip_found") else "Não encontrado")
    with c2:
        render_status_indicator("Ingestão automática", "Habilitada" if rag.get("auto_ingest_zip") else "Desabilitada")
    with c3:
        render_status_indicator("Logs", "Habilitados" if rag.get("logs_enabled") else "Desabilitados")
    display_value("Caminho do ZIP", rag.get("zip_path"))
    if not rag.get("indexed_chunks", 0):
        st.warning("O ChromaDB ainda não possui chunks indexados.")
    else:
        st.success("O ChromaDB contém documentos indexados.")

def render_alerts_section(health: dict) -> None:
    st.subheader(
        "🔔 Notificações automáticas da equipe médico-veterinária"
    )

    # =========================================================
    # 1. NORMALIZAÇÃO DA RESPOSTA
    # =========================================================

    payload = health if isinstance(health, dict) else as_dict(health)

    if not payload:
        st.warning(
            "O Streamlit não recebeu os dados de notificações do FastAPI."
        )
        return

    # Procura campos também dentro de estruturas aninhadas
    def find_value(data, field_names, depth=0):
        if depth > 6:
            return None

        if isinstance(data, dict):

            # Primeiro procura na estrutura atual
            for field in field_names:
                value = data.get(field)

                if value is not None and value != "":
                    return value

            # Depois procura nas estruturas internas
            for value in data.values():
                found = find_value(
                    value,
                    field_names,
                    depth + 1
                )

                if found is not None and found != "":
                    return found

        elif isinstance(data, list):
            for item in data:
                found = find_value(
                    item,
                    field_names,
                    depth + 1
                )

                if found is not None and found != "":
                    return found

        return None

    # =========================================================
    # 2. LOCALIZA A ESTRUTURA DE NOTIFICAÇÕES
    # =========================================================

    notifications = as_dict(
        find_value(
            payload,
            ["medical_team_notifications"]
        )
    )

    # =========================================================
    # 3. IDENTIFICA OS DOIS E-MAILS
    # =========================================================

    # Remetente: conta SMTP
    sender_email = find_value(
        payload,
        [
            "medical_team_sender_email",
            "sender_email",
            "email_user",
        ]
    )

    # Destinatário: equipe médica
    recipient_email = (
        notifications.get("recipient_email")
        or notifications.get("medical_team_email")
        or find_value(
            payload,
            [
                "medical_team_email",
                "recipient_email",
            ]
        )
    )

    sender_email = str(sender_email or "").strip()
    recipient_email = str(recipient_email or "").strip()

    # =========================================================
    # 4. STATUS DAS CONFIGURAÇÕES
    # =========================================================

    enabled = notifications.get(
        "enabled",
        find_value(
            payload,
            ["medical_team_notification_enabled"]
        )
    )

    recipient_ready = notifications.get(
        "recipient_configured",
        find_value(
            payload,
            ["medical_team_recipient_configured"]
        )
    )

    smtp_ready = notifications.get(
        "smtp_credentials_configured",
        find_value(
            payload,
            ["smtp_credentials_configured"]
        )
    )

    enabled = bool(enabled)
    recipient_ready = bool(
        recipient_ready or recipient_email
    )
    smtp_ready = bool(smtp_ready)

    # =========================================================
    # 5. STATUS GERAL
    # =========================================================

    if enabled and recipient_ready and smtp_ready:
        st.success(
            "Notificações automáticas configuradas."
        )
    else:
        st.warning(
            "Notificações automáticas ainda não estão "
            "totalmente configuradas."
        )

    # =========================================================
    # 6. INDICADORES
    # =========================================================

    cols = st.columns(4)

    cols[0].metric(
        "Serviço",
        "Habilitado" if enabled else "Desabilitado"
    )

    cols[1].metric(
        "Equipe destinatária",
        "Configurada" if recipient_ready else "Ausente"
    )

    cols[2].metric(
        "Remetente SMTP",
        "Identificado" if sender_email else "Ausente"
    )

    cols[3].metric(
        "Credenciais SMTP",
        "Configuradas" if smtp_ready else "Ausentes"
    )


    # =========================================================
    # EVENTOS QUE DISPARAM AS NOTIFICAÇÕES
    # =========================================================
        
    triggers = (
        notifications.get("trigger")
        or notifications.get("triggers")
        or []
    )
        
    if isinstance(triggers, str):
        triggers = [triggers]
        
    st.markdown(
        "##### 🔔 Eventos que geram notificações"
    )
    
    if triggers:
        for trigger in triggers:
            st.markdown(f"- {trigger}")
    else:
        st.caption(
            "Nenhum evento foi retornado pelo FastAPI."
        )

    # =========================================================
    # 7. E-MAIL DO MÉDICO-VETERINÁRIO REMETENTE
    # =========================================================

    st.markdown(
        "##### 📤 Médico-veterinário responsável pelo envio"
    )

    if sender_email:
        st.success(
            f"**E-mail remetente:** {sender_email}"
        )

        st.caption(
            "Conta autenticada no SMTP utilizada para enviar "
            "as notificações."
        )
    else:
        st.error(
            "O endereço do remetente não foi localizado "
            "nos dados recebidos pelo Streamlit."
        )


    # =========================================================
    #  E-MAIL DA EQUIPE MÉDICA DESTINATÁRIA
    # =========================================================

    st.markdown(
        "##### 📥 Equipe médica responsável pelo recebimento"
    )

    if recipient_email:
        st.success(
            f"**E-mail destinatário:** {recipient_email}"
        )

        st.caption(
            "Endereço que recebe as notificações automáticas "
            "sobre alterações nos prontuários."
        )
    else:
        st.error(
            "O endereço da equipe médica não foi localizado "
            "nos dados recebidos pelo Streamlit."
        )


    

    

def render_audit_section(api_url: str, headers: dict[str, str]) -> None:
    st.subheader("🛡️ Auditoria e rastreabilidade")
    st.caption("Pergunta → rota/modelo → evidências → validação → revisão → resposta.")
    summary, error = get_audit_summary(api_url, headers)
    if error:
        st.warning("Não foi possível consultar /audit/summary: " + error)
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        safe_metric("Eventos registrados", summary.get("total_events", 0))
    with c2:
        safe_metric("Solicitações", summary.get("total_requests", 0))
    with c3:
        safe_metric("Revisões pendentes", summary.get("pending_human_reviews", 0))
    
    st.divider()
    render_human_review_section(api_url, headers)
    st.divider()

    st.markdown("#### 🔍 Rotas utilizadas")
    route_counts = as_dict(summary.get("route_counts"))
    if route_counts:
        st.dataframe([{"Rota": route, "Quantidade": count} for route, count in route_counts.items()], use_container_width=True, hide_index=True)
    else:
        st.info("Ainda não existem decisões de roteamento registradas.")
    st.divider()
    
    st.markdown("#### 📋 Histórico de processamento")
    c1, c2 = st.columns([1, 2])
    with c1:
        limit = st.selectbox("Quantidade de eventos", [5, 10, 20, 50, 100, 200, 500], index=0)
    with c2:
        request_id = st.text_input("Filtrar por ID da solicitação", placeholder="Opcional")
    logs, logs_error = get_audit_logs(api_url, headers, limit, request_id.strip() or None)
    if logs_error:
        st.warning(logs_error)
        return
    events = as_list(logs.get("events"))
    if not events:
        st.info("Não existem eventos para os filtros selecionados.")
        return
    for raw in events:
        event = as_dict(raw)
        stage = event.get("stage", "Etapa não informada")
        status = event.get("status", "Não informado")
        timestamp = event.get("timestamp", "")
        with st.expander(f"{stage} | {status} | {timestamp}"):
            for label, key in (
                ("ID da solicitação", "request_id"), ("Rota", "route"),
                ("Modelo", "model"), ("Justificativa", "reason"),
            ):
                if event.get(key) is not None:
                    display_value(label, event.get(key))
            for label, key in (
                ("Pergunta sanitizada", "question_preview"),
                ("Resposta/estado sanitizado", "answer_preview"),
            ):
                if event.get(key):
                    st.markdown(f"**{label}:**")
                    st.write(event[key])
            for label, key in (("Variáveis/colunas", "columns_used"), ("Fontes", "sources"), ("Detalhes", "detail")):
                if event.get(key):
                    st.markdown(f"**{label}:**")
                    st.json(event[key], expanded=False)
            if event.get("duration_ms") is not None:
                st.caption(f"Duração: {event['duration_ms']} ms")


def render_painel(api_url: str, headers: Optional[dict[str, str]] = None) -> None:
    headers = dict(headers or {})
    st.title("📊 Painel Smart VET")
    st.caption("🐾 Smart VET — Sistema de apoio à rotina veterinária.")
    st.caption("Monitoramento da API, modelos, Hugging Face Space, ChromaDB, auditoria e revisão humana.")
    refresh_col, info_col = st.columns([1, 4])
    with refresh_col:
        if st.button("🔄 Atualizar painel", use_container_width=True, key="smartvet_dashboard_refresh"):
            st.rerun()
    with info_col:
        st.caption("Consultas feitas diretamente aos endpoints FastAPI, sem cache local.")
    with st.spinner("Consultando os serviços do Smart VET..."):
        health, health_error = get_api_health(api_url, headers)
        readiness, readiness_error = get_readiness(api_url, headers)
        tabular, tabular_error = get_tabular_health(api_url, headers)
        rag, rag_error = get_rag_health(api_url, headers)
        llm, llm_error = get_llm_status(api_url, headers)
    if health_error:
        st.error("Não foi possível consultar a saúde geral da API.")
        st.warning(health_error)
        st.code(f"FASTAPI_URL={api_url}")
        st.info("Verifique se o FastAPI está em execução e se API_KEY está correta.")
    else:
        render_api_section(health, readiness, api_url)
        if readiness_error:
            st.warning("Não foi possível consultar /health/ready: " + readiness_error)
        st.divider()
        render_models_section(health, llm if not llm_error else {})
        if llm_error:
            st.caption("O endpoint /llm/status não respondeu: " + llm_error)
        st.divider()
    render_rag_section(rag, rag_error)
    st.divider()
    render_tabular_section(tabular, tabular_error)
    render_alerts_section(health)
    st.divider()
    render_audit_section(api_url, headers)
    st.divider()


__all__ = [
    "load_tabular_options", "render_painel", "fetch_endpoint", "get_endpoint_data",
    "get_api_health", "get_tabular_health", "get_rag_health", "get_readiness",
    "get_llm_status", "get_audit_summary", "get_audit_logs", "render_human_review_section",
]
