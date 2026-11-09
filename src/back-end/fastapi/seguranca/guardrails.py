"""Guardrails de entrada e de saída do assistente SmartVET.

* Entrada: bloqueia pedidos impróprios (falsificação, dose letal, injeção de prompt)
  e sinaliza emergências.
* Saída: detecta conteúdo prescritivo (medicamento + dose, "prescrevo", "administre"...)
  e encaminha para validação humana; remove PII e linguagem de certeza absoluta.

As regras são determinísticas (regex) de propósito: são auditáveis, rápidas e
não dependem de outra LLM. Podem ser combinadas com um classificador LLM depois.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from .pii import contem_pii, mascarar_pii
from .politicas import AVISO_VALIDACAO_HUMANA, ConfigSeguranca


class Decisao(str, Enum):
    PERMITIR = "PERMITIR"
    BLOQUEAR = "BLOQUEAR"
    REVISAO_HUMANA = "REVISAO_HUMANA"


@dataclass
class ResultadoGuardrail:
    decisao: Decisao
    texto: str
    motivos: list[str] = field(default_factory=list)
    alertas: list[str] = field(default_factory=list)

    @property
    def bloqueado(self) -> bool:
        return self.decisao == Decisao.BLOQUEAR

    def to_dict(self) -> dict:
        return {"decisao": self.decisao.value, "motivos": self.motivos, "alertas": self.alertas}


def _normalizar(texto: str) -> str:
    """minúsculas e sem acentos, para casar 'prescrição' e 'prescricao'."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower()


# ---------------------------------------------------------------- entrada
_INJECAO = [
    r"ignor\w* (as |todas as |suas )?(instruc|regras|orientac)",
    r"ignore (all |the )?(previous|above) (instructions|rules)",
    r"desconsider\w* (as |suas )?(instruc|regras)",
    r"voce agora e\b", r"aja como se nao (tivesse|houvesse) (regras|restric)",
    r"modo (desenvolvedor|developer|dan)\b", r"jailbreak",
    r"(mostre|revele|imprima) (o |seu )?(system prompt|prompt do sistema|suas instrucoes)",
]

_PEDIDOS_IMPROPRIOS = {
    "falsificacao_documento": r"(falsific|forj|atestado falso|receita falsa|laudo falso|carteira de vacina\w* falsa|"
                              r"assinar como|usar (o )?crmv de)",
    "dano_intencional": r"(dose letal|dose fatal|como (matar|envenenar)|eutanasia (caseira|em casa sem)|"
                        r"sacrificar (o animal |o cao |o gato )?sem (veterinario|avaliacao))",
    "controlado_sem_receita": r"(conseguir|comprar|obter) \w+ (controlad\w*|cetamina|tramadol|morfina|fenobarbital)"
                              r" sem (receita|prescricao)",
    "dados_de_terceiros": r"(cpf|endereco|telefone|email|e-mail) d[oae] (outro|outra|qualquer) (tutor|cliente|paciente)",
    "prescricao_autonoma": r"(prescreva|emita|gere) (a |uma )?(receita|prescricao) (e |para )?(envie|mande|direto|automatic)",
}

# Sinais de alarme na triagem veterinária.
_EMERGENCIA = [
    r"parada cardi", r"parada cardiorrespirat", r"nao (esta )?respira", r"dispneia|dificuldade (para|de) respirar",
    r"inconscien", r"convuls", r"atropel", r"politrauma", r"intoxica|envenena|comeu (chocolate|raticida|veneno|xilitol)",
    r"torcao gastrica|dilatacao[- ]volvulo|abdomen (distendido|dilatado)",
    r"nao (consegue )?urin|obstrucao (uretral|urinaria)", r"hemorragia|sangramento (intenso|ativo)",
    r"mucosas? (palidas?|cianotic\w*)", r"picada de (cobra|serpente)|acidente ofidico", r"choque|colapso",
]


def validar_entrada(pergunta: str, config: ConfigSeguranca | None = None) -> ResultadoGuardrail:
    """Valida a pergunta do veterinário antes de chegar à LLM."""
    config = config or ConfigSeguranca()
    texto = (pergunta or "").strip()
    norm = _normalizar(texto)
    motivos: list[str] = []
    alertas: list[str] = []

    if not texto:
        return ResultadoGuardrail(Decisao.BLOQUEAR, texto, ["pergunta_vazia"])
    if len(texto) > config.tamanho_max_pergunta:
        return ResultadoGuardrail(Decisao.BLOQUEAR, texto, ["pergunta_muito_longa"])

    if any(re.search(p, norm) for p in _INJECAO):
        motivos.append("tentativa_injecao_prompt")

    for categoria, padrao in _PEDIDOS_IMPROPRIOS.items():
        if re.search(padrao, norm):
            motivos.append(f"pedido_improprio:{categoria}")

    for termo in config.termos_extras_bloqueio:
        if _normalizar(termo) in norm:
            motivos.append(f"termo_bloqueado:{termo}")

    if any(re.search(p, norm) for p in _EMERGENCIA):
        alertas.append("EMERGENCIA")

    if contem_pii(texto):
        alertas.append("PII_NA_PERGUNTA")  # segue, mas a pergunta é mascarada nos logs

    decisao = Decisao.BLOQUEAR if motivos else Decisao.PERMITIR
    return ResultadoGuardrail(decisao, texto, motivos, alertas)


# ---------------------------------------------------------------- saída
_UNIDADES = r"(mg|mcg|µg|g|ml|mL|ui|UI|unidades|gotas|comprimidos?|cp|ampolas?|mg/kg|mcg/kg/min)"
_PRESCRITIVO = {
    "dose_medicamento": rf"\b\d+([.,]\d+)?\s?{_UNIDADES}\b",
    "posologia": r"\b(de \d{1,2} ?em \d{1,2} ?h(oras)?|\d{1,2}/\d{1,2} ?h|[1-4] ?x ?(ao|por) dia|vo|iv|ev|im|sc)\b",
    "verbo_prescritivo": r"\b(prescrevo|receito|administre|administrar imediatamente|inicie|tome|aplique|suspenda|"
                         r"prescreva|deve tomar|deve usar|faca uso de)\b",
}
_CERTEZA_ABSOLUTA = r"\b(com certeza|certamente e|sem duvida|garantid[oa]|100% (seguro|eficaz)|diagnostico definitivo e)\b"


def detectar_conteudo_prescritivo(texto: str) -> list[str]:
    norm = _normalizar(texto)
    # usa o texto original para doses (preserva 'mg', 'UI') e o normalizado para verbos
    achados = []
    for nome, padrao in _PRESCRITIVO.items():
        alvo = texto if nome == "dose_medicamento" else norm
        if re.search(padrao, alvo, flags=re.IGNORECASE):
            achados.append(nome)
    return achados


def validar_saida(
    resposta: str,
    config: ConfigSeguranca | None = None,
    nomes_sensiveis: list[str] | None = None,
) -> ResultadoGuardrail:
    """Valida e ajusta a resposta da LLM antes de chegar ao veterinário."""
    config = config or ConfigSeguranca()
    motivos: list[str] = []
    alertas: list[str] = []
    texto = resposta or ""

    if contem_pii(texto) or nomes_sensiveis:
        mascarado = mascarar_pii(texto, nomes_sensiveis or [])
        if mascarado != texto:
            alertas.append("PII_REMOVIDA_DA_RESPOSTA")
            texto = mascarado

    if re.search(_CERTEZA_ABSOLUTA, _normalizar(texto)):
        alertas.append("LINGUAGEM_DE_CERTEZA_ABSOLUTA")

    prescritivo = detectar_conteudo_prescritivo(texto)
    if prescritivo:
        motivos.extend(f"conteudo_prescritivo:{p}" for p in prescritivo)

    if prescritivo and config.exigir_revisao_para_prescricao:
        decisao = Decisao.REVISAO_HUMANA
    else:
        decisao = Decisao.PERMITIR

    # O aviso vai em TODA resposta clínica, não só nas prescritivas.
    if AVISO_VALIDACAO_HUMANA not in texto:
        texto = f"{texto.rstrip()}\n\n{AVISO_VALIDACAO_HUMANA}"

    return ResultadoGuardrail(decisao, texto, motivos, alertas)
