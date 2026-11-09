"""Mascaramento e pseudonimização de dados pessoais (LGPD).

Os dados pessoais no contexto veterinário são, principalmente, os do TUTOR.
Usado antes de gravar qualquer coisa em log e antes de devolver texto ao usuário,
para que CPF, telefone, e-mail, endereço etc. nunca fiquem expostos em claro.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from typing import Iterable

# Ordem importa: padrões mais longos/específicos primeiro.
_PADROES: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("MICROCHIP", re.compile(r"\b\d{15}\b")),  # microchip ISO 11784 (15 dígitos) — identifica o tutor indiretamente
    ("CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")),
    ("TELEFONE", re.compile(r"(?:\+?55\s?)?\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b")),
    ("CEP", re.compile(r"\b\d{5}-\d{3}\b")),
]


def mascarar_pii(texto: str, nomes_sensiveis: Iterable[str] = ()) -> str:
    """Substitui dados pessoais por marcadores como ``[CPF]``.

    ``nomes_sensiveis``: nomes conhecidos do tutor (vindos do cadastro/prontuário) que também devem ser ocultados. Detecção de nomes livres exige
    NER e fica fora do escopo deste módulo.
    """
    if not texto:
        return texto
    for nome in nomes_sensiveis:
        if nome and len(nome.strip()) > 2:
            texto = re.sub(re.escape(nome.strip()), "[NOME]", texto, flags=re.IGNORECASE)
    for rotulo, padrao in _PADROES:
        texto = padrao.sub(f"[{rotulo}]", texto)
    return texto


def contem_pii(texto: str) -> list[str]:
    """Lista os tipos de PII encontrados no texto (sem alterá-lo)."""
    return [rotulo for rotulo, padrao in _PADROES if padrao.search(texto or "")]


def pseudonimizar(identificador: str | None, salt: str | None = None) -> str | None:
    """Gera um pseudônimo estável (HMAC-SHA256) para IDs de tutor/paciente/usuário.

    O mesmo ID sempre gera o mesmo pseudônimo, permitindo auditoria por
    paciente sem guardar o ID real no log. Configure ``AUDIT_SALT`` em produção.
    """
    if identificador is None:
        return None
    chave = (salt or os.getenv("AUDIT_SALT", "troque-este-salt-em-producao")).encode()
    return hmac.new(chave, str(identificador).encode(), hashlib.sha256).hexdigest()[:16]
