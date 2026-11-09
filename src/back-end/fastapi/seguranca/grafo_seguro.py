"""Integração das camadas de segurança em um fluxo LangGraph.

    pergunta
       │
  [guard_entrada] ──bloqueado──────────────────────────────┐
       │                                                   │
  [recuperar] ──sem fontes─────────────────────────────────┤
       │                                                   │
  [gerar]  (LLM com SYSTEM_PROMPT_SEGURANCA + [FONTE-n])   │
       │                                                   │
  [explicar]  (valida citações, aderência, anexa fontes)   │
       │                                                   │
  [guard_saida] ──prescritivo/baixa confiança──► [revisao_humana]
       │                                                   │
  [finalizar] ◄────────────────────────────────────────────┘

Os nós são funções comuns (``criar_nos``) para que o grupo possa reaproveitá-los
dentro do grafo principal do projeto, ou usar o grafo pronto (``construir_grafo_seguro``).
"""
from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

from .auditoria import AuditCallbackHandler, AuditLogger, novo_request_id
from .explicabilidade import Fonte, analisar_resposta, anexar_fontes, fontes_de_documentos, formatar_contexto
from .guardrails import Decisao, validar_entrada, validar_saida
from .pii import mascarar_pii
from .politicas import (MENSAGEM_BLOQUEIO, MENSAGEM_EMERGENCIA, MENSAGEM_SEM_FONTE,
                        SYSTEM_PROMPT_SEGURANCA, ConfigSeguranca)
from .revisao_humana import FilaRevisaoHumana


class EstadoSeguro(TypedDict, total=False):
    # entrada
    pergunta: str
    request_id: str
    usuario_id: str | None
    paciente_id: str | None
    tutor_id: str | None
    contexto_paciente: str | None      # espécie, raça, peso, idade, histórico (prontuário)
    nomes_sensiveis: list[str]
    # intermediários
    fontes: list[Fonte]
    resposta_bruta: str
    alertas: list[str]
    motivos: list[str]
    explicabilidade: dict
    # saída
    status: str                         # RESPONDIDA | BLOQUEADA | SEM_FONTE | AGUARDANDO_VALIDACAO
    resposta: str
    revisao_id: str | None
    inicio: float


def criar_nos(llm: Any, retriever: Any, auditoria: AuditLogger, fila: FilaRevisaoHumana,
              config: ConfigSeguranca | None = None) -> dict[str, Callable[[EstadoSeguro], dict]]:
    """Cria os nós do grafo.

    * ``llm``: qualquer chat model LangChain (``.invoke(messages)``), inclusive o modelo fine-tunado do SmartVET.
    * ``retriever``: qualquer retriever LangChain (``.invoke(pergunta) -> list[Document]``).
    """
    config = config or ConfigSeguranca()

    def _cb(estado: EstadoSeguro) -> dict:
        return {"callbacks": [AuditCallbackHandler(auditoria, estado["request_id"])]}

    def guard_entrada(estado: EstadoSeguro) -> dict:
        rid = estado.get("request_id") or novo_request_id()
        r = validar_entrada(estado["pergunta"], config)
        # Minimização de dados (LGPD): a LLM, o retriever e os logs só veem a pergunta sem PII.
        pergunta_segura = mascarar_pii(estado["pergunta"], estado.get("nomes_sensiveis") or [])
        auditoria.registrar("requisicao_recebida", rid, pergunta=pergunta_segura,
                            usuario_id=estado.get("usuario_id"), paciente_id=estado.get("paciente_id"),
                            tutor_id=estado.get("tutor_id"))
        auditoria.registrar("guardrail_entrada", rid, nivel="WARNING" if r.bloqueado else "INFO", **r.to_dict())
        return {"request_id": rid, "pergunta": pergunta_segura, "alertas": r.alertas, "motivos": r.motivos,
                "status": "BLOQUEADA" if r.bloqueado else "EM_ANDAMENTO", "inicio": time.perf_counter()}

    def recuperar(estado: EstadoSeguro) -> dict:
        docs = retriever.invoke(estado["pergunta"], config=_cb(estado))
        fontes = fontes_de_documentos(docs)
        auditoria.registrar("fontes_recuperadas", estado["request_id"],
                            fontes=[{"id": f.id, "titulo": f.titulo, "versao": f.versao,
                                     "score": f.score_recuperacao} for f in fontes])
        status = "SEM_FONTE" if (not fontes and config.bloquear_sem_fonte) else estado["status"]
        return {"fontes": fontes, "status": status}

    def gerar(estado: EstadoSeguro) -> dict:
        contexto = formatar_contexto(estado["fontes"])
        paciente = estado.get("contexto_paciente")
        humano = (f"FONTES:\n{contexto}\n\n"
                  + (f"DADOS DO PACIENTE (prontuário):\n{paciente}\n\n" if paciente else "")
                  + f"PERGUNTA DO VETERINÁRIO:\n{estado['pergunta']}\n\n"
                    "Responda citando [FONTE-n] ao final de cada afirmação.")
        msgs = [("system", SYSTEM_PROMPT_SEGURANCA), ("human", humano)]
        saida = llm.invoke(msgs, config=_cb(estado))
        return {"resposta_bruta": getattr(saida, "content", str(saida))}

    def explicar(estado: EstadoSeguro) -> dict:
        rel = analisar_resposta(estado["resposta_bruta"], estado["fontes"], config)
        auditoria.registrar("explicabilidade", estado["request_id"], **rel.to_dict())
        alertas = list(estado.get("alertas", []))
        if rel.nivel_confianca == "BAIXO":
            alertas.append("BAIXA_FUNDAMENTACAO")
        if rel.citacoes_invalidas:
            alertas.append("CITACAO_INEXISTENTE")
        return {"resposta": anexar_fontes(estado["resposta_bruta"], estado["fontes"], rel),
                "explicabilidade": rel.to_dict(), "alertas": alertas}

    def guard_saida(estado: EstadoSeguro) -> dict:
        r = validar_saida(estado["resposta"], config, estado.get("nomes_sensiveis"))
        alertas = list(estado.get("alertas", [])) + r.alertas
        motivos = list(estado.get("motivos", [])) + r.motivos
        precisa_revisao = r.decisao == Decisao.REVISAO_HUMANA or "BAIXA_FUNDAMENTACAO" in alertas \
            or "CITACAO_INEXISTENTE" in alertas
        auditoria.registrar("guardrail_saida", estado["request_id"], decisao=r.decisao.value,
                            motivos=r.motivos, alertas=r.alertas, encaminhar_revisao=precisa_revisao)
        return {"resposta": r.texto, "alertas": alertas, "motivos": motivos,
                "status": "AGUARDANDO_VALIDACAO" if precisa_revisao else "RESPONDIDA"}

    def revisao_humana(estado: EstadoSeguro) -> dict:
        item = fila.enfileirar(estado["request_id"], estado["pergunta"], estado["resposta"],
                               estado.get("motivos", []) + estado.get("alertas", []),
                               [f.id for f in estado.get("fontes", [])])
        cabecalho = (f"🕒 Resposta retida para VALIDAÇÃO HUMANA (revisão #{item.id}). "
                     "Ela só deve ser usada como conduta após aprovação de um médico-veterinário responsável.\n\n")
        return {"revisao_id": item.id, "resposta": cabecalho + estado["resposta"]}

    def finalizar(estado: EstadoSeguro) -> dict:
        status = estado["status"]
        resposta = {"BLOQUEADA": MENSAGEM_BLOQUEIO, "SEM_FONTE": MENSAGEM_SEM_FONTE}.get(status,
                                                                                       estado.get("resposta", ""))
        if "EMERGENCIA" in estado.get("alertas", []):
            resposta = f"{MENSAGEM_EMERGENCIA}\n\n{resposta}"
            auditoria.registrar("alerta_emergencia", estado["request_id"], nivel="CRITICAL")
        dur = time.perf_counter() - estado.get("inicio", time.perf_counter())
        auditoria.registrar("resposta_entregue", estado["request_id"], status=status, resposta=resposta,
                            alertas=estado.get("alertas", []), revisao_id=estado.get("revisao_id"),
                            duracao_ms=round(dur * 1000, 1))
        return {"resposta": resposta}

    return {"guard_entrada": guard_entrada, "recuperar": recuperar, "gerar": gerar, "explicar": explicar,
            "guard_saida": guard_saida, "revisao_humana": revisao_humana, "finalizar": finalizar}


def construir_grafo_seguro(llm: Any, retriever: Any, auditoria: AuditLogger | None = None,
                           fila: FilaRevisaoHumana | None = None, config: ConfigSeguranca | None = None):
    """Monta e compila o StateGraph com todas as camadas de segurança."""
    from langgraph.graph import END, START, StateGraph

    auditoria = auditoria or AuditLogger()
    fila = fila or FilaRevisaoHumana(auditoria=auditoria)
    nos = criar_nos(llm, retriever, auditoria, fila, config)

    g = StateGraph(EstadoSeguro)
    for nome, fn in nos.items():
        g.add_node(nome, fn)

    g.add_edge(START, "guard_entrada")
    g.add_conditional_edges("guard_entrada", lambda s: "finalizar" if s["status"] == "BLOQUEADA" else "recuperar")
    g.add_conditional_edges("recuperar", lambda s: "finalizar" if s["status"] == "SEM_FONTE" else "gerar")
    g.add_edge("gerar", "explicar")
    g.add_edge("explicar", "guard_saida")
    g.add_conditional_edges("guard_saida",
                            lambda s: "revisao_humana" if s["status"] == "AGUARDANDO_VALIDACAO" else "finalizar")
    g.add_edge("revisao_humana", "finalizar")
    g.add_edge("finalizar", END)
    return g.compile()
