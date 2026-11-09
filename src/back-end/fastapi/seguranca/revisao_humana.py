"""Fila de validação humana (human-in-the-loop).

Respostas com conteúdo prescritivo NÃO são liberadas como conduta: ficam com
status PENDENTE até um médico-veterinário (identificado pelo CRMV) aprovar ou rejeitar.
Cada decisão é registrada no log de auditoria.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from .auditoria import AuditLogger


class StatusRevisao(str, Enum):
    PENDENTE = "PENDENTE"
    APROVADA = "APROVADA"
    REJEITADA = "REJEITADA"


@dataclass
class ItemRevisao:
    request_id: str
    pergunta: str
    resposta_sugerida: str
    motivos: list[str]
    fontes: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: StatusRevisao = StatusRevisao.PENDENTE
    criado_em: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    revisor_crmv: str | None = None
    comentario: str | None = None
    resposta_final: str | None = None
    revisado_em: str | None = None


class FilaRevisaoHumana:
    """Fila simples persistida em JSON. Em produção, troque por uma tabela no banco."""

    def __init__(self, caminho: str | Path = "logs/fila_revisao.json", auditoria: AuditLogger | None = None):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.auditoria = auditoria
        self._lock = threading.Lock()
        self._itens: dict[str, ItemRevisao] = self._carregar()

    # ------------------------------------------------------------ persistência
    def _carregar(self) -> dict[str, ItemRevisao]:
        if not self.caminho.exists():
            return {}
        dados = json.loads(self.caminho.read_text(encoding="utf-8") or "{}")
        return {k: ItemRevisao(**{**v, "status": StatusRevisao(v["status"])}) for k, v in dados.items()}

    def _salvar(self) -> None:
        dados = {k: {**asdict(v), "status": v.status.value} for k, v in self._itens.items()}
        self.caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ operações
    def enfileirar(self, request_id: str, pergunta: str, resposta: str, motivos: list[str],
                   fontes: list[str] | None = None) -> ItemRevisao:
        item = ItemRevisao(request_id, pergunta, resposta, motivos, fontes or [])
        with self._lock:
            self._itens[item.id] = item
            self._salvar()
        if self.auditoria:
            self.auditoria.registrar("revisao_enfileirada", request_id, revisao_id=item.id, motivos=motivos)
        return item

    def _decidir(self, revisao_id: str, status: StatusRevisao, revisor_crmv: str,
                 comentario: str | None, resposta_final: str | None) -> ItemRevisao:
        if not revisor_crmv:
            raise ValueError("A validação exige a identificação (CRMV) do veterinário revisor.")
        with self._lock:
            item = self._itens[revisao_id]
            if item.status != StatusRevisao.PENDENTE:
                raise ValueError(f"Revisão {revisao_id} já foi decidida ({item.status.value}).")
            item.status = status
            item.revisor_crmv = revisor_crmv
            item.comentario = comentario
            item.resposta_final = resposta_final if status == StatusRevisao.APROVADA else None
            item.revisado_em = datetime.now(timezone.utc).isoformat()
            self._salvar()
        if self.auditoria:
            self.auditoria.registrar(
                f"revisao_{status.value.lower()}", item.request_id, revisao_id=revisao_id,
                revisor_crmv=revisor_crmv, comentario=comentario,
                resposta_editada=(resposta_final is not None and resposta_final != item.resposta_sugerida),
            )
        return item

    def aprovar(self, revisao_id: str, revisor_crmv: str, comentario: str | None = None,
                resposta_editada: str | None = None) -> ItemRevisao:
        item = self._itens[revisao_id]
        return self._decidir(revisao_id, StatusRevisao.APROVADA, revisor_crmv, comentario,
                             resposta_editada or item.resposta_sugerida)

    def rejeitar(self, revisao_id: str, revisor_crmv: str, comentario: str) -> ItemRevisao:
        return self._decidir(revisao_id, StatusRevisao.REJEITADA, revisor_crmv, comentario, None)

    def pendentes(self) -> list[ItemRevisao]:
        return [i for i in self._itens.values() if i.status == StatusRevisao.PENDENTE]

    def obter(self, revisao_id: str) -> ItemRevisao:
        return self._itens[revisao_id]
