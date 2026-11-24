"""Interface Streamlit do agente veterinário Smart VET.

Responsabilidade exclusiva do frontend: interface, estado da conversa e HTTP.
Não inicializa FastAPI, ChromaDB, embeddings, modelos, Gradio Client ou ingestão.

Contrato público mantido para app.py:
    from agente import FASTAPI_URL, api_headers, render_agente
    render_agente(FASTAPI_URL=FASTAPI_URL, api_headers=api_headers())

Roteamento decidido exclusivamente pelo FastAPI:
- Sem paciente: Gemma V6 hospedado no Hugging Face Space (free_qa).
- Com paciente: LLM cloud + RAG ChromaDB filtrado por patient_id.

O histórico é mantido no st.session_state, separado por paciente. As revisões
pendentes são consultadas pelo endpoint HTTP de status e atualizadas na mesma
conversa, sem apagar as mensagens anteriores durante reruns do Streamlit.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Optional

import requests
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:  # dotenv é opcional no frontend
    pass

FASTAPI_URL = (
    os.getenv("FASTAPI_URL")
    or os.getenv("SMARTVET_API_URL")
    or os.getenv("API_BASE_URL")
    or "http://localhost:8000"
).strip().rstrip("/")

try:
    REQUEST_TIMEOUT_SECONDS = max(30, int(os.getenv("REQUEST_TIMEOUT_SECONDS", "360")))
except (TypeError, ValueError):
    REQUEST_TIMEOUT_SECONDS = 360

MEDICAL_DISCLAIMER = (
    "Este conteúdo é apoio informativo à rotina veterinária. Não substitui o exame "
    "clínico nem a decisão do médico-veterinário responsável. Respostas de IA podem conter erros."
)

_STATE_MESSAGES = "smartvet_agent_messages"
_STATE_PATIENT = "smartvet_agent_patient_id"
_STATE_LAST_ERROR = "smartvet_agent_last_error"
_STATE_LAST_RESPONSE = "smartvet_agent_last_response"
_STATE_API_URL = "_smartvet_agent_api_url"
_STATE_GLOBAL_PATIENT = "smartvet_current_patient_id"
_STATE_RECORD_NOTICE = "smartvet_record_change_notice"
_STATE_CONVERSATIONS = "smartvet_agent_conversations"
_STATE_CONVERSATION_KEY = "smartvet_agent_conversation_key"
_STATE_PATIENT_SELECTOR_ID = "smartvet_agent_patient_selector_id"
_STATE_PATIENT_LABELS = "smartvet_agent_patient_labels"


class SmartVetAPIError(RuntimeError):
    """Erro HTTP tratado, sem expor credenciais ou cabeçalhos."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def api_headers() -> dict[str, str]:
    """Retorna os headers da API sem mostrar a chave na interface."""
    key = (
        os.getenv("API_KEY")
        or os.getenv("SMARTVET_API_KEY")
        or os.getenv("X_API_KEY")
        or ""
    ).strip()
    headers = {"Accept": "application/json"}
    if key:
        headers["X-API-Key"] = key

    # Contexto compartilhado com os módulos Streamlit tabular/vision.
    # O backend só associa exames ao prontuário quando recebe este cabeçalho.
    try:
        patient_id = str(st.session_state.get(_STATE_GLOBAL_PATIENT) or "").strip()
    except Exception:
        patient_id = ""
    if patient_id:
        headers["X-SmartVet-Patient-ID"] = patient_id
    return headers


def _headers_for_current_patient(base_headers: Any = None) -> dict[str, str]:
    """Normaliza headers recebidos como dict ou como função factory.

    Algumas versões do app.py passam api_headers() (snapshot); outras passam
    api_headers (função). Aceitamos ambos para evitar TypeError ao tentar fazer
    dict(function). O identificador do paciente é sempre recalculado depois da
    seleção atual no agente.
    """
    if callable(base_headers):
        base_headers = base_headers()

    if base_headers is None:
        base_headers = globals()["api_headers"]()

    if not isinstance(base_headers, Mapping):
        raise SmartVetAPIError(
            "Os headers da FastAPI devem ser um dicionário ou uma função que retorne um dicionário."
        )

    headers = dict(base_headers)

    # Remove qualquer contexto antigo antes de aplicar o paciente atual.
    for header_name in list(headers):
        if str(header_name).lower() == "x-smartvet-patient-id":
            headers.pop(header_name, None)

    try:
        patient_id = str(st.session_state.get(_STATE_GLOBAL_PATIENT) or "").strip()
    except Exception:
        patient_id = ""

    if patient_id:
        headers["X-SmartVet-Patient-ID"] = patient_id
    return headers


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
        if detail is not None:
            return str(detail)[:1000]
    text = (response.text or "").strip()
    return text[:1000] if text else "A API não forneceu detalhes do erro."


def _api_request(
    method: str,
    path: str,
    headers: Optional[dict[str, str]] = None,
    *,
    timeout: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    base_url = str(st.session_state.get(_STATE_API_URL) or FASTAPI_URL).strip().rstrip("/")
    if not base_url:
        raise SmartVetAPIError("FASTAPI_URL não está configurada.")
    url = f"{base_url}/{path.lstrip('/')}"
    request_headers = _headers_for_current_patient(headers)
    try:
        response = requests.request(
            method.upper(),
            url,
            headers=request_headers,
            timeout=timeout or REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.exceptions.Timeout as exc:
        raise SmartVetAPIError(
            "A API excedeu o tempo limite. O Gemma remoto pode estar aguardando a fila "
            "ZeroGPU; verifique o backend e aumente REQUEST_TIMEOUT_SECONDS se necessário."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise SmartVetAPIError(
            f"Não foi possível conectar ao FastAPI em {base_url}. Confira o backend e FASTAPI_URL."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise SmartVetAPIError(f"Falha de comunicação com a API ({type(exc).__name__}).") from exc

    if not response.ok:
        raise SmartVetAPIError(
            f"A API retornou HTTP {response.status_code}: {_error_detail(response)}",
            status_code=response.status_code,
        )
    if response.status_code == 204 or not response.content:
        return None
    if "application/json" in response.headers.get("content-type", "").lower():
        try:
            return response.json()
        except ValueError as exc:
            raise SmartVetAPIError("A API retornou JSON inválido.", response.status_code) from exc
    return response.text


def _api_download(path: str, headers: dict[str, str], timeout: int = 120) -> tuple[bytes, str]:
    """Baixa conteúdo binário sem decodificar ZIP/PDF como texto."""
    base_url = str(st.session_state.get(_STATE_API_URL) or FASTAPI_URL).strip().rstrip("/")
    request_headers = _headers_for_current_patient(headers)
    try:
        response = requests.get(
            f"{base_url}/{path.lstrip('/')}",
            headers=request_headers,
            timeout=timeout,
        )
        if not response.ok:
            raise SmartVetAPIError(
                f"A API retornou HTTP {response.status_code}: {_error_detail(response)}",
                status_code=response.status_code,
            )
        disposition = response.headers.get("content-disposition", "")
        filename = "prontuario.zip"
        if "filename=" in disposition:
            filename = disposition.split("filename=", 1)[1].strip().strip('"')
        return response.content, filename
    except requests.exceptions.Timeout as exc:
        raise SmartVetAPIError("O download do prontuário excedeu o tempo limite.") from exc
    except requests.exceptions.RequestException as exc:
        raise SmartVetAPIError(f"Falha ao baixar o prontuário ({type(exc).__name__}).") from exc


# ---------------------------------------------------------------------------
# HISTÓRICO DE CONVERSA — SESSION STATE, SEPARADO POR PACIENTE
# ---------------------------------------------------------------------------
def _conversation_key(patient_id: Optional[str]) -> str:
    patient_id = str(patient_id or "").strip()
    return f"patient:{patient_id}" if patient_id else "general"


def _copy_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    return [copy.deepcopy(item) for item in messages if isinstance(item, dict)]


def _persist_current_conversation() -> None:
    """Persiste uma cópia do histórico ativo no session_state por chave de paciente."""
    conversations = st.session_state.setdefault(_STATE_CONVERSATIONS, {})
    if not isinstance(conversations, dict):
        conversations = {}
        st.session_state[_STATE_CONVERSATIONS] = conversations

    key = st.session_state.get(_STATE_CONVERSATION_KEY)
    if not key:
        key = _conversation_key(st.session_state.get(_STATE_PATIENT))
        st.session_state[_STATE_CONVERSATION_KEY] = key

    conversations[key] = _copy_messages(st.session_state.get(_STATE_MESSAGES, []))


def _activate_conversation(patient_id: Optional[str]) -> bool:
    """Ativa o histórico do paciente; retorna True quando houve troca de conversa."""
    conversations = st.session_state.setdefault(_STATE_CONVERSATIONS, {})
    if not isinstance(conversations, dict):
        conversations = {}
        st.session_state[_STATE_CONVERSATIONS] = conversations

    new_key = _conversation_key(patient_id)
    current_key = st.session_state.get(_STATE_CONVERSATION_KEY)

    # Migra o histórico de sessões criadas antes desta versão, sem descartá-lo.
    if not current_key:
        current_key = _conversation_key(st.session_state.get(_STATE_PATIENT))
        conversations.setdefault(
            current_key,
            _copy_messages(st.session_state.get(_STATE_MESSAGES, [])),
        )
        st.session_state[_STATE_CONVERSATION_KEY] = current_key

    if current_key == new_key:
        st.session_state[_STATE_PATIENT] = patient_id
        st.session_state[_STATE_GLOBAL_PATIENT] = patient_id
        conversations.setdefault(new_key, _copy_messages(st.session_state.get(_STATE_MESSAGES, [])))
        return False

    # Salva a conversa anterior antes de carregar a nova.
    _persist_current_conversation()
    st.session_state[_STATE_MESSAGES] = _copy_messages(conversations.get(new_key, []))
    st.session_state[_STATE_CONVERSATION_KEY] = new_key
    st.session_state[_STATE_PATIENT] = patient_id
    st.session_state[_STATE_GLOBAL_PATIENT] = patient_id
    st.session_state[_STATE_LAST_ERROR] = None
    st.session_state[_STATE_LAST_RESPONSE] = None
    return True


def _on_patient_selection_changed() -> None:
    """Atualiza o contexto antes do rerun provocado pelo selectbox."""
    selected_patient_id = st.session_state.get(_STATE_PATIENT_SELECTOR_ID)
    selected_patient_id = str(selected_patient_id).strip() if selected_patient_id else None
    _activate_conversation(selected_patient_id)


def _initialize_state() -> None:
    """Inicializa o estado sem deixar o widget de paciente perder a seleção.

    O ID do paciente é mantido em uma chave própria, independente do texto
    exibido no selectbox. Se a lista /patients falhar temporariamente ou o
    widget for reconstruído durante um rerun, o paciente ativo é preservado.
    """
    st.session_state.setdefault(_STATE_MESSAGES, [])
    st.session_state.setdefault(_STATE_PATIENT, None)
    st.session_state.setdefault(_STATE_LAST_ERROR, None)
    st.session_state.setdefault(_STATE_LAST_RESPONSE, None)
    st.session_state.setdefault(_STATE_GLOBAL_PATIENT, None)

    # Recupera o contexto anterior se uma das duas chaves ainda existir.
    saved_patient_id = (
        st.session_state.get(_STATE_PATIENT)
        or st.session_state.get(_STATE_GLOBAL_PATIENT)
    )
    if saved_patient_id:
        saved_patient_id = str(saved_patient_id).strip() or None
        st.session_state[_STATE_PATIENT] = saved_patient_id
        st.session_state[_STATE_GLOBAL_PATIENT] = saved_patient_id

    # A chave nova armazena o ID, não o rótulo "Nome — ID" que pode mudar.
    # Só inicializa quando ausente; não sobrescreve a escolha explícita do usuário.
    if _STATE_PATIENT_SELECTOR_ID not in st.session_state:
        st.session_state[_STATE_PATIENT_SELECTOR_ID] = saved_patient_id

    conversations = st.session_state.setdefault(_STATE_CONVERSATIONS, {})
    if not isinstance(conversations, dict):
        conversations = {}
        st.session_state[_STATE_CONVERSATIONS] = conversations

    labels = st.session_state.setdefault(_STATE_PATIENT_LABELS, {})
    if not isinstance(labels, dict):
        st.session_state[_STATE_PATIENT_LABELS] = {}

    current_key = st.session_state.get(_STATE_CONVERSATION_KEY)
    if not current_key:
        current_key = _conversation_key(saved_patient_id)
        st.session_state[_STATE_CONVERSATION_KEY] = current_key

    # Migra o histórico legado sem substituir uma conversa já salva.
    conversations.setdefault(
        current_key,
        _copy_messages(st.session_state.get(_STATE_MESSAGES, [])),
    )
    st.session_state[_STATE_MESSAGES] = _copy_messages(
        conversations.get(current_key, st.session_state.get(_STATE_MESSAGES, []))
    )


def _clear_conversation() -> None:
    """Limpa somente a conversa ativa; não apaga as conversas de outros pacientes."""
    st.session_state[_STATE_MESSAGES] = []
    st.session_state[_STATE_LAST_ERROR] = None
    st.session_state[_STATE_LAST_RESPONSE] = None
    _persist_current_conversation()


# ---------------------------------------------------------------------------
# PACIENTES / ERROS
# ---------------------------------------------------------------------------
def _get_patients(headers: dict[str, str]) -> list[dict[str, Any]]:
    result = _api_request("GET", "/patients", headers, timeout=45)
    if isinstance(result, dict):
        raw = result.get("patients", [])
    elif isinstance(result, list):
        raw = result
    else:
        raw = []
    return [item for item in raw if isinstance(item, dict)]


def _patient_option(patient: dict[str, Any]) -> str:
    name = str(patient.get("patient_name") or "Paciente").strip()
    patient_id = str(patient.get("patient_id") or "").strip()
    return f"{name} — {patient_id}" if patient_id else name


def _render_api_error(error: Exception) -> None:
    st.error(str(error))
    base_url = st.session_state.get(_STATE_API_URL) or FASTAPI_URL
    st.caption(f"FastAPI configurado: {base_url}")
    if isinstance(error, SmartVetAPIError) and error.status_code in {401, 403}:
        st.info("Confira se API_KEY no Streamlit corresponde à API_KEY configurada no FastAPI.")
    elif isinstance(error, SmartVetAPIError) and error.status_code == 503:
        st.info("Confira as credenciais ou a disponibilidade dos serviços configurados no backend.")


# ---------------------------------------------------------------------------
# EXPLICABILIDADE
# ---------------------------------------------------------------------------
def _display_sources(sources: Any) -> None:
    if not isinstance(sources, list) or not sources:
        st.caption("Nenhuma fonte documental do ChromaDB foi recuperada nesta resposta.")
        return
    st.markdown("**Fontes recuperadas do prontuário**")
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue
        number = source.get("source_number") or index
        filename = source.get("filename") or "Documento sem nome"
        page = source.get("page")
        chunk_id = source.get("chunk_id")
        distance = source.get("distance")
        line = f"[FONTE-{number}] {filename}"
        if page is not None:
            line += f" — página {page}"
        if chunk_id:
            line += f" — chunk {str(chunk_id)[:16]}"
        st.write(line)
        if distance is not None:
            try:
                st.caption(f"Distância vetorial: {float(distance):.4f}")
            except (TypeError, ValueError):
                pass


def _display_decision_trace(trace: Any) -> None:
    if not isinstance(trace, dict) or not trace:
        st.caption("O backend não retornou a trilha de decisão.")
        return
    route = trace.get("selected_route") or trace.get("route") or "Não informada"
    reason = trace.get("route_reason") or trace.get("reason") or "Justificativa não informada."
    st.write(f"**Rota selecionada:** `{route}`")
    st.write(f"**Motivo do roteamento:** {reason}")
    for key, label in (
        ("fine_tuning_reason", "Modelo fine-tuned"),
        ("rag_reason", "Recuperação RAG"),
    ):
        if trace.get(key):
            st.write(f"**{label}:** {trace[key]}")
    if isinstance(trace.get("signals"), dict) and trace["signals"]:
        with st.expander("Sinais técnicos considerados"):
            st.json(trace["signals"])


def _display_processing_details(message: dict[str, Any]) -> None:
    with st.expander("Explicabilidade, validação e auditoria"):
        _display_decision_trace(message.get("decision_trace"))
        _display_sources(message.get("sources"))
        validation = message.get("validation")
        if isinstance(validation, dict) and validation:
            st.markdown("**Validação da resposta**")
            st.json(validation, expanded=False)
        events = message.get("processing_log") or message.get("audit_events") or []
        if isinstance(events, list) and events:
            st.markdown("**Etapas do processamento**")
            for event in events:
                if not isinstance(event, dict):
                    continue
                stage = event.get("stage") or event.get("evento") or "Etapa"
                status = event.get("status") or event.get("nivel") or ""
                detail = event.get("detail") or event.get("reason") or ""
                st.write(f"- **{stage}** — {status} {detail}".strip())
        request_id = message.get("request_id")
        if request_id:
            st.caption(f"Identificador técnico: {request_id}")


# ---------------------------------------------------------------------------
# SINCRONIZAÇÃO DAS REVISÕES HUMANAS
# ---------------------------------------------------------------------------
def _sync_pending_review_statuses() -> None:
    """Consulta revisões pendentes e atualiza somente a mensagem correspondente."""
    messages = st.session_state.get(_STATE_MESSAGES, [])
    pending_count = 0
    changed = False

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or not message.get("requires_human_review"):
            continue

        review_id = str(message.get("review_id") or "").strip()
        if not review_id:
            pending_count += 1
            message["review_sync_error"] = "O backend não retornou o identificador da revisão."
            continue

        try:
            status_data = _api_request(
                "GET",
                f"/audit/reviews/{review_id}/status",
                _headers_for_current_patient(),
                timeout=12,
            )
        except SmartVetAPIError as exc:
            if exc.status_code == 404:
                message["review_sync_error"] = "Revisão não encontrada no backend."
            else:
                message["review_sync_error"] = str(exc)
            pending_count += 1
            continue

        if not isinstance(status_data, dict):
            pending_count += 1
            continue

        status_value = str(status_data.get("status") or "pending").strip().lower()
        if status_value == "approved":
            approved_text = str(
                status_data.get("final_response")
                or message.get("draft_response")
                or message.get("content")
                or ""
            ).strip()
            if approved_text:
                message["content"] = approved_text
            message["requires_human_review"] = False
            message["review_status"] = "approved"
            message["reviewed_at"] = status_data.get("reviewed_at")
            message.pop("draft_response", None)
            message.pop("review_sync_error", None)
            messages[message_index] = message
            changed = True

        elif status_value == "rejected":
            message["content"] = (
                "Esta resposta foi rejeitada pelo revisor e não foi liberada para uso clínico. "
                "Envie uma nova solicitação após revisar as informações."
            )
            message["requires_human_review"] = False
            message["review_status"] = "rejected"
            message["reviewed_at"] = status_data.get("reviewed_at")
            message.pop("draft_response", None)
            message.pop("review_sync_error", None)
            messages[message_index] = message
            changed = True

        else:
            pending_count += 1

    if changed:
        st.session_state[_STATE_MESSAGES] = messages
        _persist_current_conversation()
        st.rerun()

    if pending_count:
        st.caption(
            f"{pending_count} revisão(ões) aguardando decisão. "
            "O status é consultado automaticamente a cada 5 segundos."
        )


if hasattr(st, "fragment"):
    _review_status_fragment = st.fragment(run_every="5s")(_sync_pending_review_statuses)
else:  # compatibilidade com versões antigas do Streamlit
    _review_status_fragment = _sync_pending_review_statuses


# ---------------------------------------------------------------------------
# RENDERIZAÇÃO DO CHAT
# ---------------------------------------------------------------------------
def _render_chat_history() -> None:
    messages = st.session_state.get(_STATE_MESSAGES, [])
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            continue
        with st.chat_message(message["role"]):
            st.markdown(str(message.get("content") or ""))
            if message["role"] == "assistant":
                if message.get("requires_human_review"):
                    st.warning(
                        "Resposta retida: aguarda aprovação do médico-veterinário revisor. "
                        "O conteúdo não foi liberado para uso clínico."
                    )
                    if message.get("review_id"):
                        st.caption(f"Identificador da revisão: {message['review_id']}")
                    if message.get("review_sync_error"):
                        st.caption(f"Sincronização da revisão: {message['review_sync_error']}")
                elif message.get("review_status") == "approved":
                    st.success("Resposta aprovada e liberada pelo médico-veterinário revisor.")
                elif message.get("review_status") == "rejected":
                    st.error("Resposta rejeitada; não liberada para uso clínico.")
                _display_processing_details(message)

    # Um único fragmento monitora todas as revisões pendentes da conversa ativa.
    has_pending_reviews = any(
        isinstance(item, dict) and item.get("requires_human_review")
        for item in messages
    )
    if has_pending_reviews:
        if st.button(
            "🔄 Atualizar status das revisões agora",
            key="smartvet_refresh_review_status",
            width="content",
        ):
            _sync_pending_review_statuses()
            st.rerun()
        _review_status_fragment()


def _submit_chat_message(prompt: str, patient_id: Optional[str], headers: dict[str, str]) -> None:
    if len(prompt) > 4000:
        st.warning("A mensagem excede o limite de 4.000 caracteres.")
        return

    # Não envia ao LLM uma resposta que ainda está retida para revisão humana.
    history = [
        {"role": item.get("role"), "content": item.get("content", "")}
        for item in st.session_state.get(_STATE_MESSAGES, [])
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and not item.get("error")
        and not item.get("requires_human_review")
        and item.get("review_status") != "rejected"
    ][-8:]

    st.session_state[_STATE_MESSAGES].append({"role": "user", "content": prompt})
    _persist_current_conversation()

    payload: dict[str, Any] = {
        "message": prompt,
        "history": history,
        "context": {},
        "enable_logs": True,
    }
    if patient_id:
        payload["patient_id"] = patient_id
        payload["context"] = {"patient_id": patient_id}

    try:
        request_headers = _headers_for_current_patient(headers)
        with st.spinner("O Smart VET está processando sua solicitação..."):
            result = _api_request(
                "POST",
                "/chat",
                request_headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        if not isinstance(result, dict):
            raise SmartVetAPIError("O endpoint /chat retornou formato inesperado.")

        response_text = str(result.get("response") or result.get("answer") or "").strip()
        if not response_text:
            raise SmartVetAPIError("O endpoint /chat não retornou resposta textual.")

        requires_review = bool(result.get("requires_human_review"))
        assistant_message = {
            "role": "assistant",
            "content": response_text,
            "draft_response": (
                str(result.get("draft_response") or response_text).strip()
                if requires_review else None
            ),
            "sources": result.get("sources") or [],
            "decision_trace": result.get("decision_trace") or {},
            "validation": result.get("validation") or {},
            "processing_log": result.get("processing_log") or result.get("audit_events") or [],
            "request_id": result.get("request_id"),
            "requires_human_review": requires_review,
            "review_status": "pending" if requires_review else None,
            "review_id": result.get("review_id"),
            "route": result.get("route"),
            "model": result.get("model"),
            "provider": result.get("provider"),
        }
        if not requires_review:
            assistant_message.pop("draft_response", None)

        st.session_state[_STATE_MESSAGES].append(assistant_message)
        st.session_state[_STATE_LAST_RESPONSE] = assistant_message
        st.session_state[_STATE_LAST_ERROR] = None
        _persist_current_conversation()

    except Exception as exc:
        st.session_state[_STATE_LAST_ERROR] = str(exc)
        st.session_state[_STATE_MESSAGES].append({
            "role": "assistant",
            "content": (
                "Não foi possível processar sua mensagem. Confira a conexão com a API e tente novamente."
            ),
            "error": True,
        })
        _persist_current_conversation()


# ---------------------------------------------------------------------------
# PRONTUÁRIO / DOCUMENTOS
# ---------------------------------------------------------------------------
def _render_document_tools(patient_id: Optional[str], headers: dict[str, str]) -> None:
    with st.expander("Prontuário, documentos e exames", expanded=bool(patient_id)):
        st.caption(
            "O paciente selecionado no atendimento é aplicado automaticamente. "
            "Não é necessário digitar o identificador novamente. PDFs, resultados "
            "de exames e textos extraídos são armazenados no backend e indexados no ChromaDB."
        )
        if not patient_id:
            st.warning("Selecione um paciente no atendimento antes de enviar PDFs ou baixar um prontuário.")
            return

        record_notice = st.session_state.pop(_STATE_RECORD_NOTICE, None)
        if isinstance(record_notice, dict):
            level = record_notice.get("level", "info")
            notice_text = str(record_notice.get("text") or "Prontuário atualizado.")
            if level == "success":
                st.success(notice_text)
            elif level == "warning":
                st.warning(notice_text)
            else:
                st.info(notice_text)

        current_headers = _headers_for_current_patient(headers)
        try:
            record = _api_request("GET", f"/patients/{patient_id}/record", current_headers, timeout=45)
        except Exception as exc:
            record = None
            st.caption(f"Não foi possível consultar o resumo do prontuário: {exc}")

        if isinstance(record, dict):
            cols = st.columns(3)
            cols[0].metric("Documentos", record.get("document_count", 0))
            cols[1].metric("Exames registrados", record.get("exam_count", 0))
            cols[2].metric("Chunks", record.get("chunk_count", 0))
            st.caption(f"Última atualização: {record.get('last_updated') or 'Não informada'}")
            with st.expander("Documentos e exames registrados", expanded=False):
                for item in record.get("documents", []) or []:
                    if isinstance(item, dict):
                        st.write(
                            f"📄 {item.get('filename') or 'Documento'} · "
                            f"{item.get('chunk_count', 0)} chunks · "
                            f"original {'arquivado' if item.get('original_pdf_archived') else 'não disponível; texto extraído preservado'}"
                        )
                for item in record.get("examinations", []) or []:
                    if isinstance(item, dict):
                        st.write(
                            f"🧪 {item.get('title') or item.get('exam_type') or 'Exame'} · "
                            f"{item.get('created_at') or 'data não informada'}"
                        )
                        attachments = item.get("attachments") or []
                        if attachments:
                            names = [
                                str(att.get("filename") or "imagem")
                                for att in attachments
                                if isinstance(att, dict)
                            ]
                            if names:
                                st.caption("Imagens arquivadas: " + ", ".join(names))

        download_col, refresh_col = st.columns(2)
        with download_col:
            try:
                archive_bytes, archive_name = _api_download(
                    f"/patients/{patient_id}/record/download", current_headers, timeout=120
                )
                st.download_button(
                    "⬇️ Baixar prontuário completo (.zip)",
                    data=archive_bytes,
                    file_name=archive_name,
                    mime="application/zip",
                    width="stretch",
                    key=f"smartvet_download_record_{patient_id}",
                )
            except Exception as exc:
                st.caption(f"Download do prontuário indisponível: {exc}")

        with refresh_col:
            if st.button(
                "Atualizar prontuário",
                key=f"smartvet_refresh_record_{patient_id}",
                width="stretch",
            ):
                st.rerun()

        st.divider()
        st.markdown("**Adicionar documentos ao paciente selecionado**")
        uploaded_pdfs = st.file_uploader(
            "Selecione um ou mais PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"smartvet_pdf_uploader_{patient_id}",
        )
        if st.button(
            "Enviar e indexar PDFs neste prontuário",
            key=f"smartvet_upload_documents_button_{patient_id}",
            disabled=not uploaded_pdfs,
            width="stretch",
        ):
            files = [
                ("files", (item.name, item.getvalue(), "application/pdf"))
                for item in (uploaded_pdfs or [])
            ]
            data = {"patient_id": patient_id}
            try:
                with st.spinner("Indexando PDFs e atualizando o prontuário..."):
                    response = _api_request(
                        "POST",
                        "/documents/upload",
                        current_headers,
                        files=files,
                        data=data,
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                st.success("Upload processado. O prontuário foi atualizado quando novos documentos foram indexados.")
                notifications = response.get("notifications", []) if isinstance(response, dict) else []
                sent = any(
                    isinstance(item, dict) and item.get("status") == "sent"
                    for item in notifications
                )
                if sent:
                    st.success("A equipe médico-veterinária foi notificada automaticamente sobre a alteração do prontuário.")
                elif notifications:
                    st.warning("O prontuário foi atualizado, mas a notificação automática não foi enviada. Confira a configuração SMTP e MEDICAL_TEAM_EMAIL.")
                else:
                    st.info("Nenhum e-mail foi enviado: o documento pode já estar indexado (sem alteração nova) ou nenhum PDF novo foi adicionado.")
                st.session_state[_STATE_RECORD_NOTICE] = {
                    "level": "success" if sent else "warning" if notifications else "info",
                    "text": (
                        "PDF(s) arquivado(s), indexado(s) e prontuário atualizado. A equipe foi notificada."
                        if sent else
                        "Prontuário conferido/atualizado. A notificação pode não ter sido enviada; confira MEDICAL_TEAM_EMAIL e SMTP."
                        if notifications else
                        "Upload processado. Não houve novo documento indexado, portanto nenhum e-mail foi disparado."
                    ),
                }
                st.rerun()
            except Exception as exc:
                _render_api_error(exc)

    


# ---------------------------------------------------------------------------
# ENTRADA PÚBLICA DO MÓDULO
# ---------------------------------------------------------------------------
def render_agente(
    FASTAPI_URL: str = FASTAPI_URL,
    api_headers: Any = None,
) -> None:
    """Renderiza o chat e as ferramentas HTTP do agente Smart VET."""
    _initialize_state()
    base_url = str(FASTAPI_URL or "").strip().rstrip("/")
    st.session_state[_STATE_API_URL] = base_url

    # Aceita um snapshot de headers (compatibilidade) ou uma função factory.
    # A factory é preferível porque lê o paciente atualmente selecionado.
    provided_headers = api_headers() if callable(api_headers) else api_headers
    if provided_headers is None:
        provided_headers = globals()["api_headers"]()
    headers = _headers_for_current_patient(provided_headers)

    st.subheader("🌐 Agente veterinário Smart VET")
    st.caption("Interface conversacional conectada ao FastAPI. Modelos, RAG e documentos permanecem no backend.")
    st.caption(f"FastAPI configurado: `{base_url or 'não configurado'}`")

    # Busca a lista de pacientes. Uma falha transitória NÃO deve trocar o
    # atendimento atual para "geral" nem remover o paciente do selectbox.
    try:
        patients = _get_patients(headers)
        patient_error = None
    except Exception as exc:
        patients = []
        patient_error = exc

    patient_by_id: dict[str, dict[str, Any]] = {}
    labels = st.session_state.setdefault(_STATE_PATIENT_LABELS, {})
    for patient in patients:
        patient_id = str(patient.get("patient_id") or "").strip()
        if not patient_id:
            continue
        patient_by_id[patient_id] = patient
        labels[patient_id] = str(
            patient.get("patient_name") or "Paciente"
        ).strip()

    # Preserva o ID salvo mesmo quando /patients falha ou o backend o omite
    # temporariamente da lista. Só o usuário pode mudar explicitamente para geral.
    saved_patient_id = (
        st.session_state.get(_STATE_PATIENT)
        or st.session_state.get(_STATE_GLOBAL_PATIENT)
        or st.session_state.get(_STATE_PATIENT_SELECTOR_ID)
    )
    saved_patient_id = str(saved_patient_id).strip() if saved_patient_id else None
    if saved_patient_id and saved_patient_id not in patient_by_id:
        patient_by_id[saved_patient_id] = {
            "patient_id": saved_patient_id,
            "patient_name": labels.get(saved_patient_id) or "Paciente selecionado",
        }

    # Opções são IDs estáveis; os nomes são apenas rótulos de apresentação.
    patient_ids = sorted(patient_by_id.keys(), key=str.casefold)
    options: list[Optional[str]] = [None, *patient_ids]

    # Garante que o valor guardado no widget continue pertencendo às opções.
    widget_patient_id = st.session_state.get(_STATE_PATIENT_SELECTOR_ID)
    if widget_patient_id and widget_patient_id not in patient_ids:
        patient_by_id[str(widget_patient_id)] = {
            "patient_id": str(widget_patient_id),
            "patient_name": labels.get(str(widget_patient_id)) or "Paciente selecionado",
        }
        patient_ids.append(str(widget_patient_id))
        patient_ids = sorted(set(patient_ids), key=str.casefold)
        options = [None, *patient_ids]

    def format_patient_selection(patient_id: Optional[str]) -> str:
        if not patient_id:
            return "Atendimento geral — sem paciente"
        patient = patient_by_id.get(str(patient_id), {})
        name = str(patient.get("patient_name") or "Paciente selecionado").strip()
        return f"{name} — {patient_id}"

    col_patient, col_action = st.columns([4, 1])
    with col_patient:
        selected_patient_id = st.selectbox(
            "Paciente selecionado",
            options=options,
            key=_STATE_PATIENT_SELECTOR_ID,
            format_func=format_patient_selection,
            on_change=_on_patient_selection_changed,
            help=(
                "Sem paciente: Gemma V6 remoto no Hugging Face Space (free_qa). "
                "Com paciente: LLM cloud + RAG filtrado por patient_id."
            ),
        )

    selected_patient_id = (
        str(selected_patient_id).strip() if selected_patient_id else None
    )
    if _activate_conversation(selected_patient_id):
        # Atualiza também as outras abas, que usam o contexto compartilhado.
        st.rerun()

    st.session_state[_STATE_GLOBAL_PATIENT] = selected_patient_id
    headers = _headers_for_current_patient(headers)

    with col_action:
        st.write("")
        if st.button(
            "Nova conversa",
            key="smartvet_new_conversation",
            width="stretch",
        ):
            _clear_conversation()
            st.rerun()

    if patient_error:
        if selected_patient_id:
            st.warning(
                "Não foi possível atualizar a lista de pacientes. "
                f"O atendimento selecionado ({selected_patient_id}) foi preservado nesta sessão."
            )
        else:
            st.warning("Não foi possível carregar a lista de pacientes.")
        with st.expander("Detalhes da conexão"):
            _render_api_error(patient_error)

    if selected_patient_id:
        selected_patient = next(
            (p for p in patients if str(p.get("patient_id")) == selected_patient_id),
            {},
        )
        st.info(
            f"Atendimento com paciente: **{selected_patient.get('patient_name') or 'Paciente selecionado'}**. "
            "O FastAPI usará LLM cloud e recuperação RAG limitada a este paciente."
        )
    else:
        st.info(
            "Atendimento geral: o FastAPI encaminhará a conversa ao Gemma Fine-Tuning hospedado no Hugging Face Space"
        )

    st.divider()
    _render_chat_history()
    prompt = st.chat_input("Digite sua pergunta para o Smart VET...", key="smartvet_agent_chat_input")
    if prompt and prompt.strip():
        _submit_chat_message(prompt.strip(), selected_patient_id, headers)
        st.rerun()

    if st.session_state.get(_STATE_LAST_ERROR):
        st.error(st.session_state[_STATE_LAST_ERROR])
    st.caption(MEDICAL_DISCLAIMER)
    st.divider()
    _render_document_tools(selected_patient_id, headers)


__all__ = ["FASTAPI_URL", "api_headers", "render_agente", "SmartVetAPIError"]
