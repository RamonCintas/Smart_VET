"""Explainability: rastrear de onde veio cada informação da resposta.

Fluxo:
1. ``fontes_de_documentos`` converte os documentos do retriever em ``Fonte``.
2. ``formatar_contexto`` numera as fontes como [FONTE-1], [FONTE-2]... no prompt.
3. A LLM é instruída (SYSTEM_PROMPT_SEGURANCA) a citar [FONTE-n] em cada afirmação.
4. ``analisar_resposta`` verifica as citações (existem? são válidas?), mede a
   aderência de cada frase à fonte citada e calcula um nível de confiança.
5. ``anexar_fontes`` adiciona ao final da resposta a lista das fontes usadas.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .politicas import ConfigSeguranca

_CITACAO = re.compile(r"\[FONTE-(\d+)\]", flags=re.IGNORECASE)
_STOPWORDS = set(
    "a o e de da do das dos em no na nos nas um uma para por com sem que se ao aos as os ou "
    "como mais mas pode podem deve devem ser sao esta este essa esse isso pelo pela pelos pelas "
    "the and of to in is for on with".split()
)


@dataclass
class Fonte:
    id: str
    titulo: str
    trecho: str
    tipo: str = "documento"          # protocolo | prontuario | faq | laudo | literatura
    versao: str | None = None         # versão/data do protocolo — importante para auditoria
    score_recuperacao: float | None = None

    def referencia(self) -> str:
        partes = [self.titulo]
        if self.versao:
            partes.append(f"versão {self.versao}")
        partes.append(f"id: {self.id}")
        return f"{' — '.join(partes)} ({self.tipo})"


@dataclass
class AnaliseFrase:
    frase: str
    citacoes: list[int]
    aderencia: float  # 0–1: fração das palavras relevantes da frase presentes na(s) fonte(s) citada(s)


@dataclass
class RelatorioExplicabilidade:
    fontes_citadas: list[str] = field(default_factory=list)
    citacoes_invalidas: list[int] = field(default_factory=list)
    frases_sem_citacao: list[str] = field(default_factory=list)
    frases_baixa_aderencia: list[str] = field(default_factory=list)
    score_grounding: float = 0.0
    nivel_confianca: str = "BAIXO"   # ALTO | MEDIO | BAIXO
    detalhes: list[AnaliseFrase] = field(default_factory=list)

    def to_dict(self, com_detalhes: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not com_detalhes:
            d.pop("detalhes")
        return d


# ---------------------------------------------------------------- utilitários
def _tokens(texto: str) -> set[str]:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return {t for t in re.findall(r"[a-z0-9]+", sem_acento) if len(t) > 2 and t not in _STOPWORDS}


def _frases(texto: str) -> list[str]:
    partes = re.split(r"(?<=[.!?])\s+|\n+", texto)
    return [p.strip() for p in partes if p.strip()]


def fontes_de_documentos(documentos: Iterable[Any]) -> list[Fonte]:
    """Converte ``langchain_core.documents.Document`` (ou dicts) em ``Fonte``.

    Metadados esperados (o que faltar recebe um valor padrão):
    ``id``/``source``, ``titulo``/``title``, ``tipo``, ``versao``, ``score``.
    """
    fontes = []
    for i, doc in enumerate(documentos, start=1):
        if isinstance(doc, dict):
            conteudo, meta = doc.get("page_content") or doc.get("conteudo", ""), doc.get("metadata", doc)
        else:
            conteudo, meta = doc.page_content, doc.metadata or {}
        fontes.append(Fonte(
            id=str(meta.get("id") or meta.get("source") or f"doc-{i}"),
            titulo=str(meta.get("titulo") or meta.get("title") or meta.get("source") or f"Documento {i}"),
            trecho=conteudo,
            tipo=str(meta.get("tipo", "documento")),
            versao=meta.get("versao"),
            score_recuperacao=meta.get("score"),
        ))
    return fontes


def formatar_contexto(fontes: list[Fonte], max_chars_por_fonte: int = 1500) -> str:
    """Monta o bloco de contexto numerado que vai para o prompt."""
    if not fontes:
        return "NENHUMA FONTE DISPONÍVEL."
    blocos = []
    for n, f in enumerate(fontes, start=1):
        blocos.append(f"[FONTE-{n}] {f.referencia()}\n{f.trecho[:max_chars_por_fonte]}")
    return "\n\n".join(blocos)


# ---------------------------------------------------------------- análise
def analisar_resposta(resposta: str, fontes: list[Fonte],
                      config: ConfigSeguranca | None = None) -> RelatorioExplicabilidade:
    config = config or ConfigSeguranca()
    rel = RelatorioExplicabilidade()
    tokens_fonte = [_tokens(f.trecho + " " + f.titulo) for f in fontes]
    citadas: set[int] = set()
    aderencias: list[float] = []

    for frase in _frases(resposta):
        nums = [int(n) for n in _CITACAO.findall(frase)]
        texto_frase = _CITACAO.sub("", frase)
        toks = _tokens(texto_frase)
        if len(toks) < 4:          # frases curtas/títulos/avisos não contam
            continue
        validos = [n for n in nums if 1 <= n <= len(fontes)]
        rel.citacoes_invalidas.extend(n for n in nums if n not in validos)
        if not validos:
            rel.frases_sem_citacao.append(frase)
            aderencias.append(0.0)
            continue
        citadas.update(validos)
        vocab = set().union(*(tokens_fonte[n - 1] for n in validos))
        aderencia = len(toks & vocab) / len(toks)
        aderencias.append(aderencia)
        rel.detalhes.append(AnaliseFrase(frase, validos, round(aderencia, 3)))
        if aderencia < config.limiar_grounding_medio:
            rel.frases_baixa_aderencia.append(frase)

    rel.fontes_citadas = [fontes[n - 1].id for n in sorted(citadas)]
    rel.citacoes_invalidas = sorted(set(rel.citacoes_invalidas))
    rel.score_grounding = round(sum(aderencias) / len(aderencias), 3) if aderencias else 0.0

    if rel.citacoes_invalidas or not citadas:
        rel.nivel_confianca = "BAIXO"
    elif rel.score_grounding >= config.limiar_grounding_alto and not rel.frases_sem_citacao:
        rel.nivel_confianca = "ALTO"
    elif rel.score_grounding >= config.limiar_grounding_medio:
        rel.nivel_confianca = "MEDIO"
    else:
        rel.nivel_confianca = "BAIXO"
    return rel


def anexar_fontes(resposta: str, fontes: list[Fonte], relatorio: RelatorioExplicabilidade) -> str:
    """Acrescenta a seção 'Fontes consultadas' e o nível de confiança à resposta."""
    linhas = [resposta.rstrip(), "", "📚 Fontes consultadas:"]
    usadas = [(n, f) for n, f in enumerate(fontes, start=1) if f.id in relatorio.fontes_citadas]
    if usadas:
        linhas += [f"  [FONTE-{n}] {f.referencia()}" for n, f in usadas]
    else:
        linhas.append("  Nenhuma fonte foi citada na resposta — trate o conteúdo com cautela.")
    linhas.append(f"🔎 Confiança na fundamentação: {relatorio.nivel_confianca} "
                  f"(aderência às fontes: {relatorio.score_grounding:.0%})")
    if relatorio.frases_sem_citacao:
        linhas.append(f"   {len(relatorio.frases_sem_citacao)} trecho(s) sem fonte identificada.")
    return "\n".join(linhas)
