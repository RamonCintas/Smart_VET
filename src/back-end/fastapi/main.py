"""Aplicação FastAPI do Smart VET.

O Gemma V6 é executado exclusivamente no Hugging Face Space pelo agente.py.
Este main.py não baixa pesos, não inicializa torch/transformers/PEFT e não
executa preload local do Gemma. Mantém startup dos módulos tabular, visão,
Chroma/RAG, auditoria e a governança de segunda passagem cloud.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.responses import Response

from core import (
    llm_diagnostics,
    logger,
    remote_gemma_diagnostics,
    require_api_key,
    settings,
    validate_answer_with_llm,
)


# ---------------------------------------------------------------------------
# RESOLUÇÃO EXPLÍCITA DO painel.py LOCAL
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PAINEL_FILE = BASE_DIR / "painel.py"

if not PAINEL_FILE.is_file():
    raise ImportError(
        "Não encontrei painel.py na mesma pasta de main.py. "
        f"Caminho esperado: {PAINEL_FILE}"
    )

_spec_painel = importlib.util.spec_from_file_location("painel", PAINEL_FILE)
if _spec_painel is None or _spec_painel.loader is None:
    raise ImportError(f"Não foi possível carregar o módulo: {PAINEL_FILE}")

_painel_module = importlib.util.module_from_spec(_spec_painel)
sys.modules["painel"] = _painel_module
try:
    _spec_painel.loader.exec_module(_painel_module)
except Exception:
    sys.modules.pop("painel", None)
    raise

for _required_symbol in (
    "router",
    "initialize_audit_db",
    "log_event",
    "create_human_review",
    "append_medical_disclaimer",
    "redact_structure",
    "redact_text",
):
    if not hasattr(_painel_module, _required_symbol):
        raise ImportError(
            f"painel.py não exporta a função/objeto obrigatório: {_required_symbol}"
        )

audit_router = _painel_module.router
initialize_audit_db = _painel_module.initialize_audit_db
log_event = _painel_module.log_event
create_human_review = _painel_module.create_human_review
append_medical_disclaimer = _painel_module.append_medical_disclaimer
redact_structure = _painel_module.redact_structure
redact_text = _painel_module.redact_text

if hasattr(_painel_module, "validate_generated_answer"):
    validate_generated_answer = _painel_module.validate_generated_answer
else:
    # Compatibilidade com versões antigas de painel.py.
    _LOCAL_PII_PATTERNS = {
        "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.I),
        "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
        "TELEFONE": re.compile(r"(?:\+?55\s?)?\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b"),
        "CEP": re.compile(r"\b\d{5}-?\d{3}\b"),
    }
    _LOCAL_PRESCRIPTIVE_PATTERNS = (
        re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:mg/kg|mcg/kg|mg|mcg|µg|ml|mL|UI|g)\b", re.I),
        re.compile(r"\b(?:prescrevo|receito|administre|administrar|inicie|aplique|suspenda|prescreva)\b", re.I),
        re.compile(r"\b(?:vo|iv|ev|im|sc)\b", re.I),
    )

    def validate_generated_answer(
        question: str,
        draft_response: str,
        route: str | None = None,
        sources: list[dict] | None = None,
        training_sources: list[dict] | None = None,
    ) -> dict[str, Any]:
        raw = str(draft_response or "")
        pii_types = [name for name, pattern in _LOCAL_PII_PATTERNS.items() if pattern.search(raw)]
        corrected = redact_text(raw)
        prescriptive = any(pattern.search(corrected) for pattern in _LOCAL_PRESCRIPTIVE_PATTERNS)
        document_route = any(
            term in str(route or "").lower()
            for term in ("rag", "document", "retriev", "patient_context")
        )
        missing_sources = document_route and not bool(sources)
        needs_review = bool(pii_types or prescriptive or missing_sources)
        notes = []
        if pii_types:
            notes.append("Dados pessoais foram identificados e mascarados.")
        if prescriptive:
            notes.append("A resposta contém linguagem prescritiva ou posologia.")
        if missing_sources:
            notes.append("Rota de contexto de paciente sem fontes documentais rastreáveis.")
        return {
            "status": "needs_human_review" if needs_review else "approved",
            "corrected_response": corrected,
            "pii_detected": bool(pii_types),
            "pii_types": pii_types,
            "prescriptive_content_detected": prescriptive,
            "risk_level": "high" if pii_types or prescriptive else ("medium" if needs_review else "low"),
            "notes": " ".join(notes) or "Nenhum gatilho determinístico identificado.",
            "validator_model": "smartvet-local-compatibility-guardrails",
            "validator_provider": "local",
            "validator_mode": "rules_fallback",
        }


# ---------------------------------------------------------------------------
# MÓDULOS DA API
# ---------------------------------------------------------------------------
from agente import (  # noqa: E402
    get_hf_space_status,
    get_rag_service,
    initialize_agent_services,
    notify_medical_team_record_change,
    router as chat_router,
)
from tabular import router as tabular_router, initialize_tabular  # noqa: E402
from vision import router as vision_router, initialize_vision  # noqa: E402


# ===========================================================================
# LIFECYCLE: AUDITORIA + TABULAR + VISION + CHROMA/RAG
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Smart VET API...")
    app.state.tabular_initialized = False
    app.state.vision_initialized = False
    app.state.rag_initialized = False
    app.state.rag_initialization_error = None
    app.state.gemma_space_status = get_hf_space_status()

    try:
        await run_in_threadpool(initialize_audit_db)
        logger.info("Audit database initialized.")

        try:
            await run_in_threadpool(initialize_tabular)
            app.state.tabular_initialized = True
            logger.info("Tabular module initialized.")
        except Exception:
            logger.exception("Tabular module initialization failed.")

        try:
            await run_in_threadpool(initialize_vision)
            app.state.vision_initialized = True
            logger.info("Vision module initialized.")
        except Exception:
            logger.exception("Vision module initialization failed.")

        # Inicializa ChromaDB e ingere o ZIP conforme RAG_AUTO_INGEST_ZIP.
        try:
            rag_result = await run_in_threadpool(initialize_agent_services)
            app.state.rag_initialized = True
            logger.info(
                "RAG/Chroma inicializado | result=%s",
                json.dumps(rag_result, ensure_ascii=False, default=str)[:1000],
            )
        except Exception as exc:
            app.state.rag_initialization_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            logger.exception("RAG/Chroma initialization failed.")

        # NÃO inicializar cliente Gradio nem disparar inferência no startup.
        # O primeiro acesso ao chat sem paciente conecta ao Space sob demanda.
        space_status = app.state.gemma_space_status
        if space_status.get("ready_to_connect"):
            logger.info(
                "Gemma V6 remoto configurado | space=%s | endpoint=%s | local_model_loaded=false",
                space_status.get("space_id"),
                space_status.get("api_name"),
            )
        else:
            logger.warning(
                "Gemma V6 remoto incompletamente configurado. Verifique ID do Space "
                "e SMARTVET_SPACE_USER/PASSWORD no .env do FastAPI."
            )

        logger.info(
            "Startup finalizado | tabular=%s | vision=%s | rag=%s | gemma_space_ready=%s",
            app.state.tabular_initialized,
            app.state.vision_initialized,
            app.state.rag_initialized,
            space_status.get("ready_to_connect", False),
        )
        yield
    finally:
        logger.info("Shutting down Smart VET API...")


app = FastAPI(
    title="Smart VET API",
    version="3.4.0",
    description=(
        "API veterinária com Gemma Fine-Tuning remoto no Hugging Face Space para conversa "
        "geral, LLM cloud + ChromaDB/RAG para paciente selecionado, validador "
        "cloud independente, guardrails e auditoria."
    ),
    lifespan=lifespan,
)


# ===========================================================================
# CORS E CABEÇALHOS DE SEGURANÇA
# ===========================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Reviewer-Key", "X-SmartVet-Patient-ID"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


# ===========================================================================
# GOVERNANÇA DO CHAT: ROTA -> REGRAS -> LLM VALIDADOR -> REVISÃO HUMANA
# ===========================================================================
@app.middleware("http")
async def persist_tabular_vision_exam_results(request: Request, call_next):
    """Anexa resultados de tabular/visão e imagens recebidas ao prontuário.

    O Streamlit envia X-SmartVet-Patient-ID por meio de api_headers(). As imagens
    multipart são copiadas e o ponteiro dos UploadFile é restaurado antes de o
    endpoint original executar; a lógica dos modelos permanece nos módulos próprios.
    """
    captured_attachments: list[dict[str, Any]] = []
    path_hint = request.url.path.lower()
    content_type_in = request.headers.get("content-type", "").lower()
    if (
        request.method.upper() in {"POST", "PUT"}
        and "multipart/form-data" in content_type_in
        and "/documents/upload" not in path_hint
        and "/chat" not in path_hint
    ):
        try:
            total_captured = 0
            max_total = int(os.getenv("EXAM_ATTACHMENTS_TOTAL_MAX_BYTES", str(20 * 1024 * 1024)))
            # Cache o body antes de analisar multipart para que o endpoint FastAPI
            # ainda consiga consumir os mesmos bytes normalmente.
            content_length = int(request.headers.get("content-length", "0") or 0)
            if content_length and content_length > max_total + (1024 * 1024):
                raise ValueError("Upload excede o limite de arquivamento de anexos.")
            await request.body()
            form = await request.form()
            for field_name, value in form.multi_items():
                if not isinstance(value, StarletteUploadFile):
                    continue
                name = str(value.filename or field_name or "imagem")
                mime = str(value.content_type or "application/octet-stream")
                if not mime.lower().startswith("image/"):
                    continue
                data_bytes = await value.read()
                await value.seek(0)
                if data_bytes and total_captured + len(data_bytes) <= max_total:
                    captured_attachments.append({
                        "filename": name,
                        "content_type": mime,
                        "content": data_bytes,
                    })
                    total_captured += len(data_bytes)
        except Exception:
            # O processamento da predição não deve falhar só porque a cópia
            # arquivística do anexo não pôde ser preparada.
            logger.exception("Não foi possível capturar anexos de imagem para o prontuário.")

    response = await call_next(request)
    if request.method.upper() not in {"POST", "PUT"} or response.status_code >= 400:
        return response

    route = request.scope.get("route")
    endpoint = getattr(route, "endpoint", None)
    route_hint = " ".join((
        str(request.url.path),
        str(getattr(route, "path", "") or ""),
        str(getattr(route, "name", "") or ""),
        str(getattr(endpoint, "__module__", "") or ""),
        str(getattr(endpoint, "__name__", "") or ""),
    )).lower()
    if any(term in route_hint for term in ("/health", "status", "options", "labels", "metadata")):
        return response

    if any(term in route_hint for term in ("vision", "dermat", "skin", "dog_breed", "image")):
        exam_type = "vision"
    elif any(term in route_hint for term in ("tabular", "triage", "clinical", "animal_condition", "symptom")):
        exam_type = "tabular"
    else:
        return response

    patient_id = (request.headers.get("X-SmartVet-Patient-ID") or "").strip()
    if not patient_id:
        log_event(
            None,
            stage="exam_result_not_attached",
            status="missing_patient_context",
            route=exam_type,
            reason="Resultado de exame não associado: X-SmartVet-Patient-ID ausente.",
            detail={"path": request.url.path},
        )
        return response

    content_type = response.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        return response

    chunks = [chunk async for chunk in response.body_iterator]
    body = b"".join(chunks)
    try:
        data = json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = None
    if not isinstance(data, (dict, list)):
        headers = dict(response.headers)
        headers.pop("content-length", None)
        headers.pop("transfer-encoding", None)
        return Response(content=body, status_code=response.status_code, headers=headers, media_type=content_type.split(";")[0], background=response.background)

    # Guarda o retorno da predição, sem tentar arquivar imagem/binário enviado.
    safe_result = redact_structure(data)
    result_text = json.dumps(safe_result, ensure_ascii=False, default=str)
    result_text = result_text[:24000]
    if not result_text or result_text in {"{}", "[]", "null"}:
        headers = dict(response.headers)
        headers.pop("content-length", None)
        headers.pop("transfer-encoding", None)
        return Response(content=body, status_code=response.status_code, headers=headers, media_type=content_type.split(";")[0], background=response.background)

    title = f"Resultado de exame {exam_type}"
    if isinstance(data, dict):
        title = str(
            data.get("title") or data.get("prediction_label") or data.get("prediction")
            or data.get("class_name") or data.get("message") or title
        )[:180]

    try:
        rag = get_rag_service()
        record = await run_in_threadpool(
            rag.record_exam_result,
            patient_id,
            exam_type,
            title,
            safe_result,
            None,
            request.url.path,
        )
        archived_attachments = []
        attachment_error = None
        if captured_attachments:
            try:
                archived_attachments = await run_in_threadpool(
                    rag.archive_exam_attachments,
                    patient_id,
                    record.get("record_id"),
                    captured_attachments,
                )
            except Exception as attach_exc:
                attachment_error = type(attach_exc).__name__
                logger.exception("Resultado do exame foi salvo, mas um anexo não pôde ser arquivado.")
        notification = await run_in_threadpool(
            notify_medical_team_record_change,
            patient_id,
            None,
            f"novo_exame_{exam_type}",
            title,
            {
                "record_id": record.get("record_id"),
                "endpoint": request.url.path,
                "attachments": [item.get("filename") for item in archived_attachments],
            },
        )
        data = data if isinstance(data, dict) else {"result": data}
        data["patient_record_update"] = {
            "status": "recorded",
            "record_id": record.get("record_id"),
            "patient_id": patient_id,
            "exam_type": exam_type,
            "attachments_archived": archived_attachments,
            "attachment_archive_error": attachment_error,
            "notification": notification,
        }
        log_event(
            None,
            stage="patient_exam_recorded",
            status="recorded",
            route=exam_type,
            model="tabular_or_vision_backend",
            reason="Resultado de exame associado ao paciente selecionado e indexado no ChromaDB.",
            detail={"record_id": record.get("record_id"), "notification_status": notification.get("status")},
        )
    except Exception as exc:
        logger.exception("Não foi possível anexar o resultado de exame ao prontuário.")
        data = data if isinstance(data, dict) else {"result": data}
        data["patient_record_update"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": "O exame foi concluído, mas não foi possível anexá-lo ao prontuário.",
        }

    payload = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    headers = dict(response.headers)
    for header in ("content-length", "transfer-encoding", "content-encoding"):
        headers.pop(header, None)
    return Response(content=payload, status_code=response.status_code, headers=headers, media_type="application/json", background=response.background)


@app.middleware("http")
async def govern_chat_response(request: Request, call_next):
    is_chat_request = request.url.path.rstrip("/").endswith("/chat") and request.method.upper() == "POST"
    if not is_chat_request:
        return await call_next(request)

    started = time.perf_counter()
    request_id = str(uuid.uuid4())
    request_body = await request.body()

    async def receive_again():
        return {"type": "http.request", "body": request_body, "more_body": False}

    request._receive = receive_again
    request_data: dict[str, Any] = {}
    try:
        parsed = json.loads(request_body.decode("utf-8")) if request_body else {}
        if isinstance(parsed, dict):
            request_data = parsed
    except (UnicodeDecodeError, json.JSONDecodeError):
        request_data = {}

    question = str(request_data.get("message") or request_data.get("prompt") or "")
    safe_question = redact_text(question)
    thread_id = request_data.get("thread_id") or request_data.get("conversation_id")

    context = request_data.get("context")
    context = context if isinstance(context, dict) else {}
    nested_patient = context.get("patient") or context.get("selected_patient") or context.get("animal")
    nested_patient_id = None
    if isinstance(nested_patient, dict):
        nested_patient_id = (
            nested_patient.get("patient_id")
            or nested_patient.get("id")
            or nested_patient.get("animal_id")
        )
    selected_patient_id = request_data.get("patient_id") or context.get("patient_id") or nested_patient_id
    patient_selected = bool(selected_patient_id is not None and str(selected_patient_id).strip())

    log_event(
        request_id=request_id,
        stage="question_received",
        status="received",
        question_preview=safe_question,
        detail={
            "patient_context_attached": patient_selected,
            "expected_chat_route": "patient_context_llm" if patient_selected else "general_gemma_v6_huggingface_space",
        },
    )

    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        log_event(
            request_id=request_id,
            stage="processing_finished",
            status=f"http_{response.status_code}",
            question_preview=safe_question,
            duration_ms=(time.perf_counter() - started) * 1000,
            detail={"note": "Resposta não JSON; sem segunda validação automática."},
        )
        return response

    response_chunks = [chunk async for chunk in response.body_iterator]
    response_body = b"".join(response_chunks)
    try:
        response_data = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        response_data = None

    if not isinstance(response_data, dict) or response.status_code >= 400:
        log_event(
            request_id=request_id,
            stage="processing_finished",
            status=f"http_{response.status_code}",
            question_preview=safe_question,
            duration_ms=(time.perf_counter() - started) * 1000,
            detail={"note": "Endpoint /chat retornou erro ou payload fora do padrão."},
        )
        headers = dict(response.headers)
        for header in ("content-length", "transfer-encoding", "content-encoding"):
            headers.pop(header, None)
        return Response(content=response_body, status_code=response.status_code, headers=headers)

    trace = response_data.get("decision_trace") or response_data.get("routing_trace") or {}
    if not isinstance(trace, dict):
        trace = {}
    route = response_data.get("route") or trace.get("selected_route") or trace.get("route")
    model = response_data.get("model") or trace.get("model")
    provider = response_data.get("provider")
    route_reason = trace.get("route_reason") or trace.get("reason") or response_data.get("route_reason")
    sources = response_data.get("sources") or []
    training_sources = response_data.get("training_sources") or response_data.get("fine_tuning_sources") or []

    explanation = (
        response_data.get("explanation")
        or response_data.get("feature_explanation")
        or response_data.get("tabular_explanation")
        or {}
    )
    columns_used = None
    if isinstance(explanation, dict):
        columns_used = explanation.get("features") or explanation.get("columns") or explanation.get("contributions")
    if columns_used is None:
        columns_used = trace.get("columns_used") or trace.get("features")

    audit_source_refs = []
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            audit_source_refs.append({
                key: source.get(key)
                for key in (
                    "filename", "source", "document", "title", "page", "page_number",
                    "chunk_id", "id", "patient_id", "distance", "source_type", "url", "source_number",
                )
                if source.get(key) is not None
            })

    log_event(
        request_id=request_id,
        stage="route_selected",
        status="reported" if route else "not_reported",
        route=str(route) if route else None,
        model=str(model) if model else None,
        reason=str(route_reason) if route_reason else None,
        question_preview=safe_question,
        columns_used=columns_used,
        sources=audit_source_refs,
        detail={"decision_trace": trace, "provider": provider},
    )

    draft = response_data.get("response") or response_data.get("draft_response") or response_data.get("answer")
    processing_events: list[dict[str, Any]] = [{
        "stage": "Roteamento",
        "status": "reportado" if route else "não informado",
        "detail": route_reason or "O endpoint não retornou justificativa da rota.",
        "provider": provider,
        "model": model,
    }]

    if isinstance(draft, str) and draft.strip():
        safe_draft = redact_text(draft)
        rules_validation = await run_in_threadpool(
            validate_generated_answer,
            question=safe_question,
            draft_response=safe_draft,
            route=str(route) if route else None,
            sources=sources if isinstance(sources, list) else [],
            training_sources=training_sources if isinstance(training_sources, list) else [],
        )

        # O Gemma V6 remoto gera a resposta geral; este segundo passe é uma
        # validação cloud separada. Ele não carrega nem substitui o Gemma.
        try:
            llm_validation = await run_in_threadpool(
                validate_answer_with_llm,
                question=safe_question,
                draft_response=safe_draft,
                route=str(route) if route else None,
                sources=audit_source_refs,
            )
            llm_validator_ok = llm_validation.get("status") == "approved"
        except Exception as exc:
            llm_validation = {
                "status": "unavailable",
                "risk_level": "unknown",
                "notes": "O segundo assistente cloud está indisponível. Os guardrails determinísticos continuam ativos.",
                "validator_error_type": type(exc).__name__,
                "validator_mode": "unavailable_with_deterministic_guardrails",
            }
            llm_validator_ok = False
            logger.exception("Falha no LLM validador de segunda passagem.")

        validation = dict(rules_validation)
        validation["llm_validator"] = llm_validation
        existing_human_gate = bool(
            response_data.get("requires_human_review")
            or response_data.get("human_review")
            or response_data.get("review_request")
        )

        # Revisão humana é exceção baseada em gatilhos concretos. A resposta
        # "needs_human_review" do LLM, isoladamente, não retém toda conversa.
        deterministic_hold_reasons = []
        if validation.get("pii_detected"):
            deterministic_hold_reasons.append("pii_na_resposta")
        if validation.get("prescriptive_content_detected"):
            deterministic_hold_reasons.append("conteudo_prescritivo_ou_posologia")
        if validation.get("missing_sources"):
            deterministic_hold_reasons.append("rota_de_prontuario_sem_fontes")
        input_reasons = validation.get("input_guardrail_reasons") or []
        if isinstance(input_reasons, list):
            deterministic_hold_reasons.extend(str(item) for item in input_reasons if item)

        llm_issues = llm_validation.get("issues") or []
        if not isinstance(llm_issues, list):
            llm_issues = [str(llm_issues)]
        critical_issue_terms = (
            "prescri", "posolog", "dose individual", "diagnóstico definitivo",
            "diagnostico definitivo", "dados pessoais", "pii", "vazamento",
            "citação inventada", "citacao inventada", "sem evidência do prontuário",
            "sem evidencia do prontuario", "unsupported patient record", "unsafe treatment",
        )
        critical_llm_issue = any(
            any(term in str(issue).lower() for term in critical_issue_terms)
            for issue in llm_issues
        )
        llm_requests_review = (
            str(llm_validation.get("status", "")).lower() == "needs_human_review"
            and (
                str(llm_validation.get("risk_level", "")).lower() == "high"
                or critical_llm_issue
            )
        )
        requires_review = bool(
            existing_human_gate
            or deterministic_hold_reasons
            or llm_requests_review
        )
        validation["human_review_reasons"] = list(dict.fromkeys(
            deterministic_hold_reasons + (["llm_validator_high_risk"] if llm_requests_review else [])
        ))
        if not llm_validator_ok and llm_validation.get("status") == "unavailable":
            validation["validation_warning"] = "Validador cloud indisponível; guardrails determinísticos foram aplicados."

        if requires_review:
            review_data = create_human_review(
                request_id=request_id,
                draft_response=validation.get("corrected_response") or safe_draft,
                route=str(route) if route else None,
                model=str(model) if model else None,
                sources=redact_structure(audit_source_refs),
                validation=validation,
                thread_id=str(thread_id) if thread_id else None,
                question=safe_question,
            )
            public_message = (
                "A resposta foi processada, mas está aguardando revisão humana "
                "antes de ser liberada. O rascunho está disponível para avaliação."
            )
            response_data["response"] = public_message
            response_data["draft_response"] = redact_text(validation.get("corrected_response") or safe_draft)
            response_data["requires_human_review"] = True
            response_data["review_id"] = review_data["review_id"]
            response_data["thread_id"] = review_data.get("thread_id") or thread_id
            response_data["human_review"] = {
                **review_data,
                "reason": validation.get("notes") or "A resposta requer confirmação humana.",
                "draft_response": response_data["draft_response"],
            }
            validation["status"] = "needs_human_review"
        else:
            final_answer = validation.get("corrected_response") or safe_draft
            response_data["response"] = append_medical_disclaimer(final_answer)
            response_data.pop("draft_response", None)
            response_data["requires_human_review"] = False
            validation["status"] = "approved"

        response_data["validation"] = {
            key: value for key, value in validation.items() if key != "corrected_response"
        }
        response_data["request_id"] = request_id
        processing_events.extend([
            {
                "stage": "Validação determinística",
                "status": rules_validation.get("status", "não informado"),
                "detail": rules_validation.get("notes", ""),
                "model": rules_validation.get("validator_model"),
            },
            {
                "stage": "Segundo assistente cloud",
                "status": llm_validation.get("status", "não informado"),
                "detail": llm_validation.get("notes", ""),
                "provider": llm_validation.get("validator_provider"),
                "model": llm_validation.get("validator_model"),
            },
        ])

        log_event(
            request_id=request_id,
            stage="second_pass_validation",
            status=validation.get("status", "unknown"),
            route=str(route) if route else None,
            model=llm_validation.get("validator_model") or model,
            reason=llm_validation.get("notes") or rules_validation.get("notes"),
            question_preview=safe_question,
            answer_preview=response_data.get("response"),
            columns_used=columns_used,
            sources=redact_structure(audit_source_refs),
            detail={
                "rules_risk_level": rules_validation.get("risk_level"),
                "llm_risk_level": llm_validation.get("risk_level"),
                "validator_provider": llm_validation.get("validator_provider"),
                "validator_mode": llm_validation.get("validator_mode"),
            },
        )
    else:
        response_data["request_id"] = request_id
        response_data["validation"] = {
            "status": "not_run",
            "notes": "O endpoint /chat não retornou resposta textual para validar.",
        }
        processing_events.append({
            "stage": "Validação por segundo assistente",
            "status": "não executada",
            "detail": "O endpoint /chat não retornou resposta textual.",
        })

    processing_events.append({
        "stage": "Resposta",
        "status": "aguardando revisão humana" if response_data.get("requires_human_review") else "liberada",
        "detail": "Resposta retida até aprovação humana." if response_data.get("requires_human_review") else "Resposta aprovada nos dois passes de validação.",
    })
    existing_events = response_data.get("processing_log") or response_data.get("audit_events") or []
    if not isinstance(existing_events, list):
        existing_events = []
    response_data["processing_log"] = existing_events + processing_events

    if not isinstance(response_data.get("decision_trace"), dict):
        response_data["decision_trace"] = {
            "selected_route": route or "not_reported",
            "route_reason": route_reason,
            "trace_complete": bool(route and route_reason),
            "fine_tuning_reason": trace.get("fine_tuning_reason"),
            "rag_reason": trace.get("rag_reason"),
            "columns_used": columns_used,
            "signals": trace.get("signals"),
            "alternatives": trace.get("alternatives"),
        }

    response_data = redact_structure(response_data)
    safe_payload = json.dumps(response_data, ensure_ascii=False, default=str).encode("utf-8")
    held = bool(response_data.get("requires_human_review"))
    log_event(
        request_id=request_id,
        stage="response_held_for_human_review" if held else "response_released",
        status="held" if held else "released",
        route=str(route) if route else None,
        model=str(model) if model else None,
        question_preview=safe_question,
        answer_preview=response_data.get("response"),
        duration_ms=(time.perf_counter() - started) * 1000,
        detail={"review_id": response_data.get("review_id")},
    )
    headers = dict(response.headers)
    for header in ("content-length", "transfer-encoding", "content-encoding"):
        headers.pop(header, None)
    return Response(content=safe_payload, status_code=response.status_code, headers=headers, media_type="application/json")


# ===========================================================================
# ROUTERS
# ===========================================================================
app.include_router(audit_router, dependencies=[Depends(require_api_key)])
app.include_router(chat_router)
app.include_router(tabular_router)
app.include_router(vision_router)


# ===========================================================================
# ROOT E HEALTH
# ===========================================================================
@app.get("/")
def root():
    return {
        "application": "Smart VET API",
        "version": "3.4.0",
        "status": "online",
        "modules": ["chat", "tabular", "vision", "painel_auditoria", "chroma_rag", "gemma_v6_huggingface_space"],
    }


def get_llm_health() -> dict[str, Any]:
    providers = settings.llm_provider_configs
    validators = settings.llm_validator_provider_configs
    return {
        **llm_diagnostics(),
        "primary": {
            "configured": bool(providers),
            "provider": providers[0].provider if providers else None,
            "model": providers[0].model if providers else None,
        },
        "validator": {
            "configured": bool(validators) or bool(providers),
            "provider": validators[0].provider if validators else (providers[0].provider if providers else None),
            "model": validators[0].model if validators else (providers[0].model if providers else None),
            "uses_dedicated_provider": bool(validators),
        },
        "local_gemma": {
            "enabled": False,
            "loaded": False,
            "download_enabled": False,
        },
    }


def get_models_health() -> dict[str, Any]:
    return {
        "tabular_model": settings.hf_model_repo,
        "dataset": settings.hf_data_repo,
        "tabular_model_file": settings.hf_model_filename,
        "image_model": settings.hf_image_repo,
        "image_model_file": settings.hf_image_filename,
    }


@app.get("/health")
def health():
    try:
        rag_status = get_rag_service().status()
    except Exception as exc:
        rag_status = {"status": "unavailable", "error_type": type(exc).__name__}
    space_status = get_hf_space_status()
    return {
        "status": "online",
        "environment": settings.api_env,
        "version": "3.4.0",
        "modules": {
            "chat": "registered",
            "tabular": "initialized" if getattr(app.state, "tabular_initialized", False) else "not_initialized",
            "vision": "initialized" if getattr(app.state, "vision_initialized", False) else "not_initialized",
            "rag": "initialized" if getattr(app.state, "rag_initialized", False) else "not_initialized",
        },
        "models": get_models_health(),
        "llm": get_llm_health(),
        # Campo legado mantido para clientes que ainda consomem fine_tuned_model.
        "fine_tuned_model": {
            **space_status,
            "status": "remote_configured" if space_status.get("ready_to_connect") else "remote_configuration_incomplete",
        },
        "gemma_v6_remote": space_status,
        "medical_team_notifications": {
            "enabled": bool(rag_status.get("medical_team_notification_enabled")),
            "recipient_configured": bool(rag_status.get("medical_team_recipient_configured")),
            "smtp_credentials_configured": bool(rag_status.get("smtp_credentials_configured")),
            "trigger": ["novo_pdf_indexado", "novo_exame_registrado"],
        },
        "rag_persistence": rag_status,
        "governance": {
            "pii_redaction": "enabled_for_chat_responses_and_audit",
            "deterministic_guardrails": "enabled",
            "llm_second_pass_validation": "enabled_when_validator_provider_configured",
            "human_review_interrupt": "persistent_api_review_queue",
            "audit_log": "sqlite",
        },
    }


@app.get("/health/ready")
def readiness():
    tabular_ready = getattr(app.state, "tabular_initialized", False)
    vision_ready = getattr(app.state, "vision_initialized", False)
    space_status = get_hf_space_status()
    space_ready = bool(space_status.get("ready_to_connect"))
    # Não testa o endpoint executando geração; isso consumiria GPU/ZeroGPU.
    ready = tabular_ready and vision_ready and space_ready
    return {
        "status": "ready" if ready else "degraded_or_initializing",
        "tabular_initialized": tabular_ready,
        "vision_initialized": vision_ready,
        "gemma_v6_space_configured": space_ready,
        "gemma_v6_space": space_status,
        "local_gemma_required": False,
        "local_gemma_loaded": False,
        "rag_initialized": getattr(app.state, "rag_initialized", False),
        "rag_initialization_error": getattr(app.state, "rag_initialization_error", None),
        "note": "https://github.com/RamonCintas/Smart_VET",
    }
