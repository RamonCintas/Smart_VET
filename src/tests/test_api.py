import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

# ============================================================
# CONFIGURAÇÃO
# ============================================================


TESTS_DIR = Path(__file__).resolve().parent
ENV_FILE = TESTS_DIR / ".env"

load_dotenv(ENV_FILE)

TEST_API_URL = os.getenv("TEST_API_URL")
TEST_API_KEY = os.getenv("TEST_API_KEY")
TEST_API_TIMEOUT = float(
    os.getenv("TEST_API_TIMEOUT", "120")
)

TEST_IMAGE_PATH = TESTS_DIR / "image-test.jpg"



# ============================================================
# VALIDAÇÃO DA CONFIGURAÇÃO
# ============================================================

if not TEST_API_URL:
    raise RuntimeError(
        "\n"
        "============================================================\n"
        "ERRO: TEST_API_URL nao configurada\n"
        "============================================================\n"
        f"Arquivo .env procurado:\n{ENV_FILE}\n"
        "\n"
        "Adicione ao .env:\n"
        "TEST_API_URL=http://localhost:8000\n"
        "============================================================\n"
    )


if not TEST_API_KEY:
    raise RuntimeError(
        "\n"
        "============================================================\n"
        "ERRO: TEST_API_KEY nao configurada\n"
        "============================================================\n"
        f"Arquivo .env procurado:\n{ENV_FILE}\n"
        "\n"
        "Adicione ao .env:\n"
        "TEST_API_KEY=12092002\n"
        "============================================================\n"
    )


TEST_API_URL = TEST_API_URL.rstrip("/")


# ============================================================
# CLIENTE HTTP REAL
# ============================================================

@pytest.fixture
def api_client():
    """
    Cliente HTTP REAL.

    NAO utiliza FastAPI TestClient.

    As requisicoes sao enviadas para TEST_API_URL.
    """

    print()
    print("=" * 70)
    print("SMART VET - API INTEGRATION TEST")
    print("=" * 70)
    print(f"ENV FILE : {ENV_FILE}")
    print(f"API URL  : {TEST_API_URL}")
    print(f"TIMEOUT  : {TEST_API_TIMEOUT}s")
    print("=" * 70)
    print()

    with httpx.Client(
        base_url=TEST_API_URL,
        timeout=TEST_API_TIMEOUT,
        follow_redirects=True,
    ) as client:
        yield client


# ============================================================
# DADOS DE TESTE
# ============================================================

def auth_headers() -> dict[str, str]:
    """
    Headers utilizados nas requisicoes autenticadas.
    """

    return {
        "X-API-Key": TEST_API_KEY,
    }


def condition_payload() -> dict[str, str]:
    """
    Dados utilizados no diagnostico por sintomas.
    """

    return {
        "animal_name": "dog",
        "symptoms1": "fever",
        "symptoms2": "vomiting",
        "symptoms3": "diarrhea",
        "symptoms4": "fatigue",
        "symptoms5": "dehydration",
    }


def load_test_image() -> bytes:
    """
    Carrega a imagem REAL localizada em:

        tests/image-test.jpg

    Essa imagem e enviada para o endpoint real
    /predict/image.
    """

    print()
    print("=" * 70)
    print("CARREGANDO IMAGEM REAL DE TESTE")
    print("=" * 70)
    print(f"Arquivo: {TEST_IMAGE_PATH}")

    if not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(
            "\n"
            "Imagem de teste nao encontrada.\n"
            f"Esperado em:\n{TEST_IMAGE_PATH}\n"
            "\n"
            "Coloque a imagem em:\n"
            "back-end\\fastapi\\tests\\image-test.jpg\n"
        )

    if not TEST_IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"O caminho nao e um arquivo:\n{TEST_IMAGE_PATH}"
        )

    image_bytes = TEST_IMAGE_PATH.read_bytes()

    if not image_bytes:
        raise ValueError(
            f"A imagem esta vazia:\n{TEST_IMAGE_PATH}"
        )

    print(f"Tamanho: {len(image_bytes)} bytes")
    print("[OK] Imagem real carregada.")

    return image_bytes


# ============================================================
# TESTE 1
# HEALTH CHECK
# ============================================================

def test_health_is_public_and_returns_security_headers(
    api_client,
):
    """
    GET /health

    Testa a API REAL.
    """

    print()
    print("=" * 70)
    print("[TEST 1] HEALTH CHECK")
    print("=" * 70)

    url = f"{TEST_API_URL}/health"

    print(f"Request: GET {url}")

    response = api_client.get("/health")

    print(f"HTTP Status: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 200, (
        "\n"
        "Health check falhou.\n"
        f"URL: {url}\n"
        f"Status: {response.status_code}\n"
        f"Response: {response.text}\n"
    )

    data = response.json()

    print(f"JSON: {data}")

    assert isinstance(data, dict)
    assert data.get("status") == "online"

    assert response.headers.get(
        "x-content-type-options"
    ) == "nosniff"

    assert response.headers.get(
        "x-frame-options"
    ) == "DENY"

    print("[OK] Health check passou.")


# ============================================================
# TESTE 2
# API KEY AUSENTE
# ============================================================

def test_condition_prediction_requires_api_key(
    api_client,
):
    """
    Verifica se /predict/condition rejeita requisicoes
    sem API key.
    """

    print()
    print("=" * 70)
    print("[TEST 2] API KEY AUSENTE")
    print("=" * 70)

    url = f"{TEST_API_URL}/predict/condition"

    print(f"Request: POST {url}")
    print("API Key: NAO enviada")

    response = api_client.post(
        "/predict/condition",
        json=condition_payload(),
    )

    print(f"HTTP Status: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 401, (
        "\n"
        "A API nao rejeitou requisicao sem API key.\n"
        f"URL: {url}\n"
        f"Status: {response.status_code}\n"
        f"Response: {response.text}\n"
    )

    print("[OK] API key obrigatoria.")


# ============================================================
# TESTE 3
# API KEY INCORRETA
# ============================================================

def test_condition_prediction_rejects_wrong_api_key(
    api_client,
):
    """
    Verifica se uma API key incorreta e rejeitada.
    """

    print()
    print("=" * 70)
    print("[TEST 3] API KEY INCORRETA")
    print("=" * 70)

    url = f"{TEST_API_URL}/predict/condition"

    print(f"Request: POST {url}")
    print("API Key: wrong-key")

    response = api_client.post(
        "/predict/condition",
        json=condition_payload(),
        headers={
            "X-API-Key": "wrong-key",
        },
    )

    print(f"HTTP Status: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 403, (
        "\n"
        "A API nao rejeitou API key incorreta.\n"
        f"URL: {url}\n"
        f"Status: {response.status_code}\n"
        f"Response: {response.text}\n"
    )

    print("[OK] API key incorreta rejeitada.")


# ============================================================
# TESTE 4
# DIAGNÓSTICO POR SINTOMAS
# ============================================================

def test_condition_diagnosis(api_client):
    """
    TESTE REAL DE DIAGNOSTICO POR SINTOMAS.

    Envia um payload real para:

        POST /predict/condition

    O modelo real da API e executado.
    """

    print()
    print("=" * 70)
    print("[TEST 4] DIAGNOSTICO POR SINTOMAS")
    print("=" * 70)

    payload = condition_payload()

    url = f"{TEST_API_URL}/predict/condition"

    print(f"Request: POST {url}")
    print()
    print("Payload:")
    print(payload)
    print()
    print("Enviando para API REAL...")

    response = api_client.post(
        "/predict/condition",
        json=payload,
        headers=auth_headers(),
    )

    print()
    print(f"HTTP Status: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 200, (
        "\n"
        "DIAGNOSTICO POR SINTOMAS FALHOU.\n"
        f"URL: {url}\n"
        f"Status: {response.status_code}\n"
        f"Response: {response.text}\n"
    )

    data = response.json()

    print()
    print("JSON retornado:")
    print(data)

    assert isinstance(data, dict)

    assert "prediction_label" in data, (
        "Resposta nao possui prediction_label."
    )

    assert data["prediction_label"] is not None

    print()
    print("[OK] DIAGNOSTICO POR SINTOMAS PASSOU.")


# ============================================================
# TESTE 5
# IMAGEM INVÁLIDA
# ============================================================

def test_invalid_image_is_rejected(
    api_client,
):
    """
    Envia dados que nao representam uma imagem.

    A API deve rejeitar com HTTP 400.
    """

    print()
    print("=" * 70)
    print("[TEST 5] IMAGEM INVALIDA")
    print("=" * 70)

    url = f"{TEST_API_URL}/predict/image"

    invalid_file = b"THIS IS NOT A REAL IMAGE"

    print(f"Request: POST {url}")
    print(
        f"Arquivo invalido: "
        f"{len(invalid_file)} bytes"
    )

    response = api_client.post(
        "/predict/image",
        files={
            "file": (
                "payload.jpg",
                invalid_file,
                "image/jpeg",
            )
        },
        headers=auth_headers(),
    )

    print(f"HTTP Status: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 400, (
        "\n"
        "A API nao rejeitou imagem invalida.\n"
        f"URL: {url}\n"
        f"Status: {response.status_code}\n"
        f"Response: {response.text}\n"
    )

    data = response.json()

    assert data.get("detail") == "Invalid image file"

    print("[OK] Imagem invalida corretamente rejeitada.")


# ============================================================
# TESTE 6
# DIAGNÓSTICO POR IMAGEM REAL
# ============================================================

def test_image_diagnosis(
    api_client,
):
    """
    TESTE REAL DE DIAGNOSTICO POR IMAGEM.

    Usa:

        tests/image-test.jpg

    Fluxo:

        imagem fisica
             |
             v
        bytes reais
             |
             v
        multipart/form-data
             |
             v
        POST /predict/image
             |
             v
        FastAPI REAL
             |
             v
        modelo REAL
             |
             v
        resposta JSON

    NAO utiliza:

        TestClient
        monkeypatch
        mock
        fake_result
        imagem artificial
    """

    print()
    print("=" * 70)
    print("[TEST 6] DIAGNOSTICO POR IMAGEM REAL")
    print("=" * 70)

    url = f"{TEST_API_URL}/predict/image"

    # --------------------------------------------------------
    # CARREGAR IMAGEM FISICA
    # --------------------------------------------------------

    image_bytes = load_test_image()

    # --------------------------------------------------------
    # ENVIAR IMAGEM
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ENVIANDO IMAGEM PARA API")
    print("=" * 70)

    print(f"Request: POST {url}")
    print("Arquivo: image-test.jpg")
    print("Content-Type: image/jpeg")
    print(f"Tamanho: {len(image_bytes)} bytes")
    print()
    print("Executando modelo REAL...")
    print()

    response = api_client.post(
        "/predict/image",
        files={
            "file": (
                "image-test.jpg",
                image_bytes,
                "image/jpeg",
            )
        },
        headers=auth_headers(),
    )

    # --------------------------------------------------------
    # RESULTADO HTTP
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTADO DO DIAGNOSTICO POR IMAGEM")
    print("=" * 70)

    print(f"HTTP Status: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 200, (
        "\n"
        "DIAGNOSTICO POR IMAGEM FALHOU.\n"
        f"URL: {url}\n"
        f"Imagem: {TEST_IMAGE_PATH}\n"
        f"Status: {response.status_code}\n"
        f"Response: {response.text}\n"
    )

    # --------------------------------------------------------
    # VALIDAR JSON
    # --------------------------------------------------------

    data = response.json()

    print()
    print("JSON retornado:")
    print(data)

    assert isinstance(data, dict), (
        "A API nao retornou um objeto JSON."
    )

    # --------------------------------------------------------
    # VALIDAR CONTRATO DA API
    # --------------------------------------------------------

    assert "success" in data, (
        "Resposta nao possui o campo 'success'.\n"
        f"Response: {data}"
    )

    assert data["success"] is True, (
        "A API informou que o processamento falhou.\n"
        f"Response: {data}"
    )

    assert "dog_breeds" in data, (
        "Resposta nao possui 'dog_breeds'.\n"
        f"Response: {data}"
    )

    assert "skin_diseases" in data, (
        "Resposta nao possui 'skin_diseases'.\n"
        f"Response: {data}"
    )

    # --------------------------------------------------------
    # VALIDAR TIPOS
    # --------------------------------------------------------

    assert isinstance(
        data["dog_breeds"],
        list,
    ), (
        "dog_breeds deveria ser uma lista.\n"
        f"Response: {data}"
    )

    assert isinstance(
        data["skin_diseases"],
        list,
    ), (
        "skin_diseases deveria ser uma lista.\n"
        f"Response: {data}"
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # NAO exigimos que dog_breeds ou skin_diseases
    # tenham elementos.
    #
    # A imagem pode ser corretamente processada e,
    # dependendo do modelo, retornar:
    #
    # dog_breeds: [...]
    # skin_diseases: []
    #
    # ou:
    #
    # dog_breeds: []
    # skin_diseases: [...]
    #
    # ou ambas vazias.
    #
    # O objetivo deste teste e verificar:
    #
    # 1. imagem real foi enviada;
    # 2. FastAPI recebeu;
    # 3. imagem foi processada;
    # 4. modelo executou;
    # 5. API respondeu;
    # 6. contrato JSON foi respeitado.
    # --------------------------------------------------------

    print()
    print("Diagnostico recebido:")
    print(
        f"  Raças de cachorro : {data['dog_breeds']}"
    )
    print(
        f"  Doenças de pele   : {data['skin_diseases']}"
    )

    if data.get("interpretation") is not None:
        print(
            f"  Interpretation     : "
            f"{data['interpretation']}"
        )

    print()
    print("[OK] DIAGNOSTICO POR IMAGEM PASSOU.")
    print(
        "[OK] A imagem real foi processada pelo modelo."
    )


# ============================================================
# TESTE 7
# CONFIGURAÇÃO
# ============================================================

def test_api_configuration():
    """
    Valida a configuracao usada pelos testes.
    """

    print()
    print("=" * 70)
    print("SMART VET - TEST TARGET")
    print("=" * 70)

    print(f"ENV FILE      : {ENV_FILE}")
    print(f"TEST_API_URL  : {TEST_API_URL}")
    print(
        "TEST_API_KEY  : "
        f"{'CONFIGURADA' if TEST_API_KEY else 'AUSENTE'}"
    )
    print(f"TIMEOUT       : {TEST_API_TIMEOUT}s")
    print(f"TEST IMAGE    : {TEST_IMAGE_PATH}")

    print("=" * 70)

    assert TEST_API_URL.startswith(
        ("http://", "https://")
    )

    assert TEST_API_KEY

    assert TEST_API_TIMEOUT > 0

    assert TEST_IMAGE_PATH.exists(), (
        f"Imagem de teste nao encontrada:\n"
        f"{TEST_IMAGE_PATH}"
    )

    assert TEST_IMAGE_PATH.is_file()

    assert TEST_IMAGE_PATH.stat().st_size > 0
