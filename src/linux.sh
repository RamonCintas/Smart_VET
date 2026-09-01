#!/usr/bin/env bash

set -u

echo "=================================="
echo "Smart VET - Local Setup"
echo "=================================="

# ============================================================
# GARANTE QUE O SCRIPT RODE A PARTIR DA PRÓPRIA PASTA
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ============================================================
# CHECK PYTHON
# ============================================================

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python não encontrado."
    echo "Instale Python 3.13.14"
    exit 1
fi

PYTHON="python3"

# ============================================================
# CHECK BACKEND .ENV
# ============================================================

if [ ! -f "back-end/fastapi/.env" ]; then
    echo "[ERROR] Backend .env não encontrado."
    echo
    echo "SOLUÇÃO:"
    echo "cp back-end/fastapi/envExample.txt back-end/fastapi/.env"
    exit 1
fi

# ============================================================
# CHECK FRONTEND .ENV
# ============================================================

if [ ! -f "front-end/streamlit/.env" ]; then
    echo "[ERROR] Frontend .env não encontrado."
    echo
    echo "SOLUÇÃO:"
    echo "cp front-end/streamlit/envExample.txt front-end/streamlit/.env"
    exit 1
fi

# ============================================================
# BACKEND SETUP
# ============================================================

echo
echo "=================================="
echo "BACKEND SETUP"
echo "=================================="

cd "$SCRIPT_DIR/back-end/fastapi" || exit 1

if [ ! -d ".venv" ]; then
    echo "Criando venv backend..."
    "$PYTHON" -m venv .venv

    if [ $? -ne 0 ]; then
        echo "[ERROR] Não foi possível criar o venv do backend."
        exit 1
    fi
fi

source .venv/bin/activate

echo "Instalando dependências backend..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ============================================================
# INICIANDO FASTAPI
# ============================================================

echo
echo "=================================="
echo "INICIANDO FASTAPI"
echo "=================================="

# Abre o FastAPI em um novo processo/terminal quando possível.
# Se não houver terminal gráfico, roda em background.

if command -v gnome-terminal >/dev/null 2>&1; then

    gnome-terminal -- bash -c "
        cd '$SCRIPT_DIR/back-end/fastapi' &&
        source .venv/bin/activate &&
        uvicorn main:app --host 127.0.0.1 --port 8000 --reload;
        exec bash
    "

elif command -v konsole >/dev/null 2>&1; then

    konsole --hold -e bash -c "
        cd '$SCRIPT_DIR/back-end/fastapi' &&
        source .venv/bin/activate &&
        uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    " &

elif command -v xterm >/dev/null 2>&1; then

    xterm -hold -e bash -c "
        cd '$SCRIPT_DIR/back-end/fastapi' &&
        source .venv/bin/activate &&
        uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    " &

else

    echo "[INFO] Nenhum terminal gráfico encontrado."
    echo "[INFO] Iniciando FastAPI em background..."

    nohup bash -c "
        cd '$SCRIPT_DIR/back-end/fastapi' &&
        source .venv/bin/activate &&
        uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    " > "$SCRIPT_DIR/fastapi.log" 2>&1 &

    FASTAPI_PID=$!

    echo "[INFO] FastAPI PID: $FASTAPI_PID"
    echo "[INFO] Logs: $SCRIPT_DIR/fastapi.log"

fi

# ============================================================
# AGUARDANDO FASTAPI
# ============================================================

echo
echo "Aguardando FastAPI iniciar..."
echo

API_READY=0

for i in $(seq 1 30); do

    if command -v curl >/dev/null 2>&1; then

        if curl -sf --max-time 2 \
            "http://127.0.0.1:8000/health" \
            >/dev/null 2>&1; then

            API_READY=1
            break

        fi

    else

        if "$PYTHON" -c "
import urllib.request
try:
    response = urllib.request.urlopen(
        'http://127.0.0.1:8000/health',
        timeout=2
    )
    exit(0 if response.status == 200 else 1)
except Exception:
    exit(1)
"; then

            API_READY=1
            break

        fi

    fi

    echo "Aguardando API... tentativa $i/30"
    sleep 2

done

# ============================================================
# CHECK API
# ============================================================

if [ "$API_READY" -eq 0 ]; then

    echo
    echo "============================================================"
    echo "[ERROR] FASTAPI NÃO INICIOU"
    echo "============================================================"
    echo
    echo "A API não respondeu em:"
    echo "http://localhost:8000/health"
    echo
    echo "Os testes não serão executados."
    echo "O Streamlit não será iniciado."
    echo

    exit 1

fi

echo
echo "============================================================"
echo "[OK] FASTAPI ONLINE"
echo "============================================================"
echo
echo "http://localhost:8000/health"
echo

# ============================================================
# PYTEST / DEBUG
# ============================================================

echo
echo "============================================================"
echo "                   PYTEST / DEBUG"
echo "============================================================"
echo

cd "$SCRIPT_DIR/tests" || exit 1

if [ ! -d ".venv" ]; then
    echo "Criando venv de testes..."
    "$PYTHON" -m venv .venv

    if [ $? -ne 0 ]; then
        echo "[ERROR] Não foi possível criar o venv dos testes."
        exit 1
    fi
fi

source .venv/bin/activate

echo "Instalando dependências dos testes..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pytest test_api.py \
    -vv \
    -s \
    --capture=no \
    --log-cli-level=DEBUG \
    --tb=long

PYTEST_EXIT_CODE=$?

echo
echo "============================================================"
echo "                 PYTEST FINALIZADO"
echo "============================================================"
echo
echo "[INFO] Exit code do pytest: $PYTEST_EXIT_CODE"
echo

if [ "$PYTEST_EXIT_CODE" -ne 0 ]; then

    echo "============================================================"
    echo "                   [ERROR] PYTEST"
    echo "============================================================"
    echo
    echo "OS TESTES FALHARAM."
    echo
    echo "O STREAMLIT NÃO SERÁ INICIADO."
    echo
    echo "Corrija os erros acima e execute novamente."
    echo

    exit "$PYTEST_EXIT_CODE"

fi

echo "============================================================"
echo "             [OK] PYTEST PASSOU"
echo "============================================================"
echo

# ============================================================
# FRONTEND SETUP
# ============================================================

echo
echo "=================================="
echo "FRONTEND SETUP"
echo "=================================="

cd "$SCRIPT_DIR/front-end/streamlit" || exit 1

if [ ! -d ".venv" ]; then
    echo "Criando venv frontend..."
    "$PYTHON" -m venv .venv

    if [ $? -ne 0 ]; then
        echo "[ERROR] Não foi possível criar o venv do frontend."
        exit 1
    fi
fi

source .venv/bin/activate

echo "Instalando dependências frontend..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# ============================================================
# INICIANDO STREAMLIT
# ============================================================

echo "Iniciando Streamlit..."

if command -v gnome-terminal >/dev/null 2>&1; then

    gnome-terminal -- bash -c "
        cd '$SCRIPT_DIR/front-end/streamlit' &&
        source .venv/bin/activate &&
        python -m streamlit run app.py;
        exec bash
    "

elif command -v konsole >/dev/null 2>&1; then

    konsole --hold -e bash -c "
        cd '$SCRIPT_DIR/front-end/streamlit' &&
        source .venv/bin/activate &&
        python -m streamlit run app.py
    " &

elif command -v xterm >/dev/null 2>&1; then

    xterm -hold -e bash -c "
        cd '$SCRIPT_DIR/front-end/streamlit' &&
        source .venv/bin/activate &&
        python -m streamlit run app.py
    " &

else

    echo "[INFO] Nenhum terminal gráfico encontrado."
    echo "[INFO] Iniciando Streamlit em background..."

    nohup bash -c "
        cd '$SCRIPT_DIR/front-end/streamlit' &&
        source .venv/bin/activate &&
        python -m streamlit run app.py
    " > "$SCRIPT_DIR/streamlit.log" 2>&1 &

    STREAMLIT_PID=$!

    echo "[INFO] Streamlit PID: $STREAMLIT_PID"
    echo "[INFO] Logs: $SCRIPT_DIR/streamlit.log"

fi

# ============================================================
# FINAL
# ============================================================

echo
echo "=================================="
echo "SMART VET ONLINE"
echo "=================================="
echo
echo "FastAPI Docs:"
echo "http://localhost:8000/docs"
echo
echo "Streamlit UI:"
echo "http://localhost:8501"
echo

exit 0
