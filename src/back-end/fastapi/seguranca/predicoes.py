"""Camada de segurança aplicada às rotas /predict/* do Smart VET.

As rotas de predição não usam o grafo do assistente (não há pergunta livre nem
retriever), mas a interpretação gerada pelo ``LLMInterpreter`` é texto de LLM
e passa pelas mesmas proteções:

* guardrail de saída: conteúdo prescritivo (dose, posologia, "administre"...)
  vai para a fila de revisão humana e a interpretação é marcada como retida;
* PII mascarada nos campos de texto da interpretação;
* explicabilidade: lista das fontes que geraram o resultado (modelo, dataset,
  versão no Hugging Face e LLM/regras que escreveram a interpretação);
* auditoria: todos os eventos com o mesmo ``request_id`` na cadeia de hashes.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .auditoria import AuditLogger, novo_request_id
from .guardrails import Decisao, validar_saida
from .pii import mascarar_pii
from .politicas import AVISO_VALIDACAO_HUMANA
from .revisao_humana import FilaRevisaoHumana

_CAMPOS_TEXTO = ("summary", "explanation")
_CAMPOS_LISTA = ("actionable_insights", "limitations")


def resumo_imagem(image_bytes: bytes) -> dict:
    """Identifica a imagem no log sem gravar o conteúdo dela."""
    return {"tamanho_bytes": len(image_bytes), "sha256": hashlib.sha256(image_bytes).hexdigest()}


def fonte_interpretacao(interpretacao: dict | None) -> list[dict]:
    if not interpretacao:
        return []
    return [{
        "id": interpretacao.get("model_used"),
        "titulo": "Interpretação textual do resultado",
        "tipo": "llm" if interpretacao.get("generated_by") == "llm" else "regras",
        "provedor": interpretacao.get("provider"),
    }]


def _texto_interpretacao(interpretacao: dict) -> str:
    partes = [interpretacao.get(c) or "" for c in _CAMPOS_TEXTO]
    for campo in _CAMPOS_LISTA:
        partes.extend(interpretacao.get(campo) or [])
    return "\n".join(p for p in partes if p)


def _resumo_resultado(resultado: dict) -> dict:
    resumo = {k: resultado[k] for k in ("prediction_label", "probability_yes", "success", "message")
              if k in resultado}
    for chave in ("dog_breeds", "skin_diseases"):
        if resultado.get(chave):
            resumo[chave] = [{"label": i["label"], "confidence": round(i["confidence"], 4)}
                             for i in resultado[chave][:3]]
    return resumo


class SegurancaPredicoes:
    """Envolve as predições com auditoria, guardrail de saída e revisão humana."""

    def __init__(self, auditoria: AuditLogger, fila: FilaRevisaoHumana):
        self.auditoria = auditoria
        self.fila = fila

    def iniciar(self, endpoint: str, entrada: dict) -> tuple[str, float]:
        request_id = novo_request_id()
        self.auditoria.registrar("requisicao_recebida", request_id, endpoint=endpoint, entrada=entrada)
        return request_id, time.perf_counter()

    def erro(self, request_id: str, exc: Exception) -> None:
        self.auditoria.registrar("predicao_erro", request_id, nivel="ERROR", erro=f"{type(exc).__name__}: {exc}")

    def concluir(self, request_id: str, inicio: float, entrada: dict, resultado: dict,
                 fontes: list[dict]) -> dict:
        """Valida a interpretação, registra a auditoria e anexa ``safety`` ao resultado."""
        interpretacao = resultado.get("interpretation")
        fontes = fontes + fonte_interpretacao(interpretacao)
        self.auditoria.registrar("predicao_concluida", request_id, resultado=_resumo_resultado(resultado),
                                 fontes=fontes)

        safety: dict[str, Any] = {"request_id": request_id, "status": "RESPONDIDA", "motivos": [],
                                  "alertas": [], "revisao_id": None, "aviso": AVISO_VALIDACAO_HUMANA,
                                  "fontes": fontes}
        if interpretacao:
            self._validar_interpretacao(request_id, entrada, interpretacao, safety)

        self.auditoria.registrar("resposta_entregue", request_id, status=safety["status"],
                                 revisao_id=safety["revisao_id"],
                                 duracao_ms=round((time.perf_counter() - inicio) * 1000, 1))
        resultado["safety"] = safety
        return resultado

    def _validar_interpretacao(self, request_id: str, entrada: dict, interpretacao: dict,
                               safety: dict) -> None:
        r = validar_saida(_texto_interpretacao(interpretacao))
        for campo in _CAMPOS_TEXTO:
            interpretacao[campo] = mascarar_pii(interpretacao.get(campo) or "")
        for campo in _CAMPOS_LISTA:
            interpretacao[campo] = [mascarar_pii(i) for i in interpretacao.get(campo) or []]

        precisa_revisao = r.decisao == Decisao.REVISAO_HUMANA
        self.auditoria.registrar("guardrail_saida", request_id, decisao=r.decisao.value, motivos=r.motivos,
                                 alertas=r.alertas, encaminhar_revisao=precisa_revisao)
        safety.update(motivos=r.motivos, alertas=r.alertas)

        if precisa_revisao:
            item = self.fila.enfileirar(request_id, json.dumps(entrada, ensure_ascii=False, default=str),
                                        _texto_interpretacao(interpretacao), r.motivos + r.alertas,
                                        [str(f["id"]) for f in safety["fontes"]])
            interpretacao["summary"] = (f"🕒 Interpretação retida para VALIDAÇÃO HUMANA (revisão #{item.id}). "
                                        f"{interpretacao['summary']}")
            safety.update(status="AGUARDANDO_VALIDACAO", revisao_id=item.id)
