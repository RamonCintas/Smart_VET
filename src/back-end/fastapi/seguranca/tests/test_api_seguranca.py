import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langgraph")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from seguranca import AuditLogger, FilaRevisaoHumana, construir_grafo_seguro  # noqa: E402
from seguranca.api import criar_router  # noqa: E402
from seguranca.demo import PROTOCOLOS, RetrieverPalavrasChave  # noqa: E402


@pytest.fixture
def cliente(tmp_path):
    log = AuditLogger(tmp_path / "a.jsonl")
    fila = FilaRevisaoHumana(tmp_path / "f.json", auditoria=log)
    llm = FakeListChatModel(responses=["Administre cefalexina 22 mg/kg VO de 12/12h pelo protocolo de "
                                       "piodermite superficial canina [FONTE-1]."])
    grafo = construir_grafo_seguro(llm, RetrieverPalavrasChave(documentos=PROTOCOLOS), log, fila)
    app = FastAPI()
    app.include_router(criar_router(log, fila, grafo))
    return TestClient(app)


def test_fluxo_completo_pela_api(cliente):
    r = cliente.post("/seguranca/assistente/perguntar",
                     json={"pergunta": "Qual antibiótico iniciar no protocolo de piodermite superficial canina?",
                           "usuario_id": "CRMV-SP-1"})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["status"] == "AGUARDANDO_VALIDACAO" and corpo["revisao_id"]

    pendentes = cliente.get("/seguranca/revisoes/pendentes").json()
    assert [p["id"] for p in pendentes] == [corpo["revisao_id"]]

    r = cliente.post(f"/seguranca/revisoes/{corpo['revisao_id']}/aprovar", json={"revisor_crmv": "CRMV-SP-9"})
    assert r.status_code == 200 and r.json()["status"] == "APROVADA"
    assert cliente.post(f"/seguranca/revisoes/{corpo['revisao_id']}/aprovar",
                        json={"revisor_crmv": "CRMV-SP-9"}).status_code == 409

    trilha = cliente.get(f"/seguranca/auditoria/{corpo['request_id']}").json()
    assert {"requisicao_recebida", "revisao_enfileirada", "revisao_aprovada"} <= {e["evento"] for e in trilha}
    assert cliente.get("/seguranca/auditoria/integridade").json()["integro"] is True


def test_rejeicao_exige_motivo(cliente):
    assert cliente.post("/seguranca/revisoes/x/rejeitar", json={"revisor_crmv": "CRMV-1"}).status_code == 422
    assert cliente.post("/seguranca/revisoes/x/aprovar", json={"revisor_crmv": "CRMV-1"}).status_code == 404
