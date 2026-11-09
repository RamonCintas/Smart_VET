"""Rotas FastAPI da camada de segurança (assistente, revisão humana e auditoria).

Integração no ``main.py`` existente:

    from fastapi import Depends
    from security import require_api_key
    from seguranca import AuditLogger, FilaRevisaoHumana, construir_grafo_seguro
    from seguranca.api import criar_router

    auditoria = AuditLogger("logs/auditoria.jsonl", versao_modelo="smartvet-llm-ft-v1")
    fila = FilaRevisaoHumana("logs/fila_revisao.json", auditoria=auditoria)
    grafo = construir_grafo_seguro(llm, retriever, auditoria, fila)   # llm/retriever da Fase 3
    app.include_router(criar_router(auditoria, fila, grafo, dependencias=[Depends(require_api_key)]))
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .auditoria import AuditLogger
from .revisao_humana import FilaRevisaoHumana


class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., min_length=1, max_length=4000)
    usuario_id: str | None = Field(None, description="CRMV de quem pergunta")
    paciente_id: str | None = None
    tutor_id: str | None = None
    contexto_paciente: str | None = Field(None, description="Espécie, raça, peso, idade, histórico")
    nomes_sensiveis: list[str] = Field(default_factory=list, description="Nome do tutor, para mascaramento")


class PerguntaResponse(BaseModel):
    request_id: str
    status: str
    resposta: str
    alertas: list[str] = []
    motivos: list[str] = []
    revisao_id: str | None = None
    explicabilidade: dict[str, Any] | None = None


class DecisaoRevisao(BaseModel):
    revisor_crmv: str = Field(..., min_length=3)
    comentario: str | None = None
    resposta_editada: str | None = None


def criar_router(auditoria: AuditLogger, fila: FilaRevisaoHumana, grafo: Any | None = None,
                 dependencias: list | None = None) -> APIRouter:
    router = APIRouter(prefix="/seguranca", tags=["Segurança e validação"], dependencies=dependencias or [])

    if grafo is not None:
        @router.post("/assistente/perguntar", response_model=PerguntaResponse)
        def perguntar(req: PerguntaRequest):
            estado = grafo.invoke(req.model_dump())
            return PerguntaResponse(**{k: estado.get(k) for k in PerguntaResponse.model_fields if k in estado})

    @router.get("/revisoes/pendentes")
    def listar_pendentes():
        return [vars(i) for i in fila.pendentes()]

    @router.post("/revisoes/{revisao_id}/aprovar")
    def aprovar(revisao_id: str, body: DecisaoRevisao):
        try:
            return vars(fila.aprovar(revisao_id, body.revisor_crmv, body.comentario, body.resposta_editada))
        except KeyError:
            raise HTTPException(404, "Revisão não encontrada")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @router.post("/revisoes/{revisao_id}/rejeitar")
    def rejeitar(revisao_id: str, body: DecisaoRevisao):
        if not body.comentario:
            raise HTTPException(422, "Informe o motivo da rejeição em 'comentario'.")
        try:
            return vars(fila.rejeitar(revisao_id, body.revisor_crmv, body.comentario))
        except KeyError:
            raise HTTPException(404, "Revisão não encontrada")
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @router.get("/auditoria/integridade")
    def integridade():
        ok, linha = auditoria.verificar_integridade()
        return {"integro": ok, "primeira_linha_violada": linha, "total_eventos": len(auditoria.eventos())}

    @router.get("/auditoria/{request_id}")
    def trilha(request_id: str):
        eventos = auditoria.eventos(request_id)
        if not eventos:
            raise HTTPException(404, "Nenhum evento para este request_id")
        return eventos

    return router
