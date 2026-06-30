#!/bin/bash

set -e

echo "=================================="
echo "Smart VET - Local Setup"
echo "=================================="

# GARANTE EXECUÇÃO A PARTIR DA PASTA DO SCRIPT
cd "$(dirname "$0")"

# CHECK PYTHON
if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python 3 não encontrado."
    echo "Instale Python 3.13.14"
    exit 1
fi

# CHECK BACKEND .ENV
if [ ! -f "back-end/fastapi/.env" ]; then
    echo "[ERROR] Backend .env não encontrado."
    echo
    echo "SOLUÇÃO:"
    echo "cp back-end/fastapi/envExample.txt back-end/fastapi/.env"
    exit 1
fi

# CHECK FRONTEND .ENV
if [ ! -f "front-end/streamlit/.env" ]; then
    echo "[ERROR] Frontend .env não encontrado."
    echo
    echo "SOLUÇÃO:"
    echo "cp front-end/streamlit/envExample.txt front-end/streamlit/.env"
    exit 1
fi

echo
echo "=================================="
echo "BACKEND SETUP"
echo "=================================="

cd back-end/fastapi

if [ ! -d ".venv" ]; then
    echo "Criando venv backend..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Instalando dependências backend..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Iniciando FastAPI..."
if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -c "source .venv/bin/activate && uvicorn main:app --reload; exec bash"
elif command -v xterm >/dev/null 2>&1; then
    xterm -e "bash -c 'source .venv/bin/activate && uvicorn main:app --reload'"
else
    nohup bash -c "source .venv/bin/activate && uvicorn main:app --reload" >/dev/null 2>&1 &
fi

deactivate
cd ../..

echo
echo "=================================="
echo "FRONTEND SETUP"
echo "=================================="

cd front-end/streamlit

if [ ! -d ".venv" ]; then
    echo "Criando venv frontend..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Instalando dependências frontend..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Iniciando Streamlit..."
if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal -- bash -c "source .venv/bin/activate && python -m streamlit run app.py; exec bash"
elif command -v xterm >/dev/null 2>&1; then
    xterm -e "bash -c 'source .venv/bin/activate && python -m streamlit run app.py'"
else
    nohup bash -c "source .venv/bin/activate && python -m streamlit run app.py" >/dev/null 2>&1 &
fi

deactivate
cd ../..

echo
echo "=================================="
echo "SMART VET ONLINE"
echo "=================================="
echo "FastAPI Docs:"
echo "http://localhost:8000/docs"
echo
echo "Streamlit UI:"
echo "http://localhost:8501"
echo

wait