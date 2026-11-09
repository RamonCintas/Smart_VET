import pytest

from seguranca import AuditLogger, FilaRevisaoHumana
from seguranca.politicas import AVISO_VALIDACAO_HUMANA
from seguranca.predicoes import SegurancaPredicoes, resumo_imagem

FONTES = [{"id": "repo/best_model.pkl", "titulo": "Modelo tabular", "tipo": "modelo", "versao": "abc123"}]


def _interpretacao(**campos):
    base = {"summary": "O modelo tabular indicou possivel estado critico.", "risk_level": "alto",
            "explanation": "A leitura deve ser usada como apoio de triagem.",
            "actionable_insights": ["Priorizar avaliacao veterinaria."], "limitations": ["Usa cinco sintomas."],
            "model_used": "gemini-3.6-flash", "provider": "gemini", "generated_by": "llm"}
    return {**base, **campos}


@pytest.fixture
def seguranca(tmp_path):
    log = AuditLogger(tmp_path / "a.jsonl")
    fila = FilaRevisaoHumana(tmp_path / "f.json", auditoria=log)
    return SegurancaPredicoes(log, fila), log, fila


def _executar(seg, interpretacao):
    entrada = {"animal_name": "dog", "symptoms1": "fever"}
    rid, inicio = seg.iniciar("/predict/condition", entrada)
    resultado = {"prediction_label": "Yes - perigoso", "probability_yes": 0.82, "interpretation": interpretacao}
    return seg.concluir(rid, inicio, entrada, resultado, list(FONTES))


def test_interpretacao_informativa_e_liberada_com_fontes(seguranca):
    seg, log, fila = seguranca
    out = _executar(seg, _interpretacao())
    safety = out["safety"]
    assert safety["status"] == "RESPONDIDA" and safety["revisao_id"] is None
    assert safety["aviso"] == AVISO_VALIDACAO_HUMANA
    assert [f["id"] for f in safety["fontes"]] == ["repo/best_model.pkl", "gemini-3.6-flash"]
    assert not fila.pendentes()
    eventos = [e["evento"] for e in log.eventos(safety["request_id"])]
    assert eventos == ["requisicao_recebida", "predicao_concluida", "guardrail_saida", "resposta_entregue"]
    assert log.verificar_integridade()[0]


def test_interpretacao_prescritiva_vai_para_revisao(seguranca):
    seg, _, fila = seguranca
    out = _executar(seg, _interpretacao(actionable_insights=["Administre dipirona 25 mg/kg VO de 8/8h."]))
    safety = out["safety"]
    assert safety["status"] == "AGUARDANDO_VALIDACAO"
    assert "conteudo_prescritivo:dose_medicamento" in safety["motivos"]
    assert fila.pendentes()[0].id == safety["revisao_id"]
    assert out["interpretation"]["summary"].startswith("🕒 Interpretação retida para VALIDAÇÃO HUMANA")


def test_pii_mascarada_na_interpretacao_e_no_log(seguranca):
    seg, log, _ = seguranca
    out = _executar(seg, _interpretacao(explanation="Contate o tutor no (11) 98765-4321 ou joao@mail.com."))
    assert "98765" not in out["interpretation"]["explanation"]
    assert "PII_REMOVIDA_DA_RESPOSTA" in out["safety"]["alertas"]
    assert "98765" not in log.caminho.read_text(encoding="utf-8")


def test_paciente_id_pseudonimizado_no_log(seguranca):
    seg, log, _ = seguranca
    rid, _ = seg.iniciar("/predict/condition", {"animal_name": "dog"}, paciente_id="VET-001")
    registro = log.eventos(rid)[0]
    assert registro["dados"]["paciente_id"] and "VET-001" not in log.caminho.read_text(encoding="utf-8")


def test_sem_interpretacao_e_erro_auditado(seguranca):
    seg, log, _ = seguranca
    entrada = {"filename": "gato.jpg", **resumo_imagem(b"bytes")}
    rid, inicio = seg.iniciar("/predict/image", entrada)
    out = seg.concluir(rid, inicio, entrada, {"success": False, "interpretation": None}, list(FONTES))
    assert out["safety"]["status"] == "RESPONDIDA"
    seg.erro(rid, ValueError("falha"))
    assert log.eventos(rid)[-1]["evento"] == "predicao_erro"
