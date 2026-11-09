"""Logging detalhado para rastreamento e auditoria.

Cada evento vira uma linha JSON (JSONL) com:
  * ``request_id`` para reconstruir toda a jornada de uma pergunta;
  * dados pessoais mascarados e IDs pseudonimizados (LGPD);
  * ``hash`` encadeado ao ``hash_anterior`` (SHA-256): se alguém editar ou
    apagar uma linha, ``verificar_integridade`` detecta a quebra da cadeia.

Também inclui um callback do LangChain que registra automaticamente chamadas
à LLM, ao retriever e às tools.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .pii import mascarar_pii, pseudonimizar

_HASH_INICIAL = "0" * 64
# IDs que nunca vão em claro para o log: viram pseudônimo (HMAC).
_CAMPOS_ID = {"usuario_id", "paciente_id", "tutor_id", "veterinario_id"}


def novo_request_id() -> str:
    return uuid.uuid4().hex


def _sanitizar(valor: Any, chave: str | None = None) -> Any:
    if isinstance(valor, dict):
        return {k: _sanitizar(v, k) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_sanitizar(v, chave) for v in valor]
    if isinstance(valor, str):
        return mascarar_pii(valor)  # todo texto gravado em log passa pelo mascaramento
    return valor


class AuditLogger:
    """Grava eventos de auditoria em JSONL com cadeia de hashes."""

    def __init__(self, caminho: str | Path = "logs/auditoria.jsonl", nivel_console: int = logging.INFO,
                 versao_modelo: str | None = None):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.versao_modelo = versao_modelo
        self._lock = threading.Lock()
        self._ultimo_hash = self._ler_ultimo_hash()
        self._console = logging.getLogger("smart-vet.auditoria")
        # Se a aplicação já configurou o logging (ex.: basicConfig no main.py), usa o handler dela.
        if not self._console.handlers and not logging.getLogger().handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self._console.addHandler(h)
        self._console.setLevel(nivel_console)

    def _ler_ultimo_hash(self) -> str:
        if not self.caminho.exists():
            return _HASH_INICIAL
        ultimo = _HASH_INICIAL
        with self.caminho.open(encoding="utf-8") as f:
            for linha in f:
                if linha.strip():
                    ultimo = json.loads(linha)["hash"]
        return ultimo

    def registrar(self, evento: str, request_id: str | None = None, nivel: str = "INFO", **dados: Any) -> dict:
        """Registra um evento. IDs em ``_CAMPOS_ID`` são pseudonimizados; textos, mascarados."""
        for campo in _CAMPOS_ID & dados.keys():
            dados[campo] = pseudonimizar(dados[campo])
        registro = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "evento": evento,
            "nivel": nivel,
            "request_id": request_id,
            "versao_modelo": self.versao_modelo,
            "dados": _sanitizar(dados),
        }
        with self._lock:
            registro["hash_anterior"] = self._ultimo_hash
            corpo = json.dumps(registro, ensure_ascii=False, sort_keys=True, default=str)
            registro["hash"] = hashlib.sha256(corpo.encode()).hexdigest()
            with self.caminho.open("a", encoding="utf-8") as f:
                f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
            self._ultimo_hash = registro["hash"]
        self._console.log(getattr(logging, nivel, logging.INFO),
                          "%s req=%s %s", evento, (request_id or "-")[:8], _resumo(registro["dados"]))
        return registro

    # ------------------------------------------------------------ consulta
    def eventos(self, request_id: str | None = None) -> list[dict]:
        if not self.caminho.exists():
            return []
        with self.caminho.open(encoding="utf-8") as f:
            regs = [json.loads(l) for l in f if l.strip()]
        return [r for r in regs if request_id is None or r["request_id"] == request_id]

    def verificar_integridade(self) -> tuple[bool, int | None]:
        """Recalcula a cadeia de hashes. Retorna (ok, linha_da_primeira_falha)."""
        anterior = _HASH_INICIAL
        for i, reg in enumerate(self.eventos(), start=1):
            hash_gravado = reg.pop("hash")
            if reg.get("hash_anterior") != anterior:
                return False, i
            corpo = json.dumps(reg, ensure_ascii=False, sort_keys=True, default=str)
            if hashlib.sha256(corpo.encode()).hexdigest() != hash_gravado:
                return False, i
            anterior = hash_gravado
        return True, None


def _resumo(dados: dict, limite: int = 140) -> str:
    texto = json.dumps(dados, ensure_ascii=False, default=str)
    return texto if len(texto) <= limite else texto[:limite] + "…"


# ---------------------------------------------------------------- LangChain
try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # LangChain é opcional para o núcleo do módulo
    BaseCallbackHandler = object  # type: ignore[assignment,misc]


class AuditCallbackHandler(BaseCallbackHandler):  # type: ignore[misc,valid-type]
    """Callback que registra no AuditLogger tudo o que a LLM/retriever/tools fazem.

    Uso: ``chain.invoke(x, config={"callbacks": [AuditCallbackHandler(logger, request_id)]})``
    """

    def __init__(self, auditoria: AuditLogger, request_id: str):
        self.auditoria = auditoria
        self.request_id = request_id
        self._inicio: dict[Any, float] = {}

    # LLM -------------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._inicio[run_id] = time.perf_counter()
        prompt = "\n".join(f"{m.type}: {m.content}" for lote in messages for m in lote)
        self.auditoria.registrar("llm_inicio", self.request_id, modelo=_nome(serialized, kwargs),
                                 prompt=prompt[:4000])

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._inicio[run_id] = time.perf_counter()
        self.auditoria.registrar("llm_inicio", self.request_id, modelo=_nome(serialized, kwargs),
                                 prompt="\n".join(prompts)[:4000])

    def on_llm_end(self, response, *, run_id, **kwargs):
        dur = time.perf_counter() - self._inicio.pop(run_id, time.perf_counter())
        texto = response.generations[0][0].text if response.generations and response.generations[0] else ""
        uso = (response.llm_output or {}).get("token_usage")
        self.auditoria.registrar("llm_fim", self.request_id, saida=texto[:4000],
                                 latencia_ms=round(dur * 1000, 1), uso_tokens=uso)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.auditoria.registrar("llm_erro", self.request_id, nivel="ERROR", erro=repr(error))

    # Retriever ------------------------------------------------------
    def on_retriever_end(self, documents, *, run_id, **kwargs):
        self.auditoria.registrar("retriever_fim", self.request_id,
                                 documentos=[d.metadata.get("id") or d.metadata.get("source") for d in documents])

    # Tools ----------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.auditoria.registrar("tool_inicio", self.request_id, tool=(serialized or {}).get("name"),
                                 entrada=mascarar_pii(str(input_str))[:2000])

    def on_tool_end(self, output, *, run_id, **kwargs):
        self.auditoria.registrar("tool_fim", self.request_id, saida=str(output)[:2000])

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.auditoria.registrar("tool_erro", self.request_id, nivel="ERROR", erro=repr(error))


def _nome(serialized: dict | None, kwargs: dict) -> str | None:
    meta = kwargs.get("metadata") or {}
    return meta.get("ls_model_name") or (serialized or {}).get("name")
