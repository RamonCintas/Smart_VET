"""SmartVET – camada de segurança e validação do assistente veterinário (Tech Challenge – Fase 3)."""
from .auditoria import AuditCallbackHandler, AuditLogger, novo_request_id
from .explicabilidade import (Fonte, RelatorioExplicabilidade, analisar_resposta, anexar_fontes,
                              fontes_de_documentos, formatar_contexto)
from .guardrails import Decisao, ResultadoGuardrail, validar_entrada, validar_saida
from .pii import mascarar_pii, pseudonimizar
from .politicas import SYSTEM_PROMPT_SEGURANCA, ConfigSeguranca
from .revisao_humana import FilaRevisaoHumana, StatusRevisao

__all__ = [
    "AuditCallbackHandler", "AuditLogger", "novo_request_id",
    "Fonte", "RelatorioExplicabilidade", "analisar_resposta", "anexar_fontes",
    "fontes_de_documentos", "formatar_contexto",
    "Decisao", "ResultadoGuardrail", "validar_entrada", "validar_saida",
    "mascarar_pii", "pseudonimizar",
    "SYSTEM_PROMPT_SEGURANCA", "ConfigSeguranca",
    "FilaRevisaoHumana", "StatusRevisao",
    "construir_grafo_seguro", "criar_nos",
]


def __getattr__(nome):  # importa LangGraph só quando necessário
    if nome in ("construir_grafo_seguro", "criar_nos"):
        from . import grafo_seguro
        return getattr(grafo_seguro, nome)
    raise AttributeError(nome)
