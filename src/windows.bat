@echo off
title Smart VET - Local Runner
setlocal

echo ==================================
echo Smart VET - Local Setup
echo ==================================

:: GARANTE QUE O SCRIPT RODE A PARTIR DA PROPRIA PASTA
cd /d "%~dp0"

:: CHECK PYTHON
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python nao encontrado.
    echo Instale Python 3.13.14
    pause
    exit /b 1
)

:: CHECK BACKEND .ENV
if not exist "back-end\fastapi\.env" (
    echo [ERROR] Backend .env nao encontrado.
    echo.
    echo SOLUCAO:
    echo copy back-end\fastapi\envExample.txt back-end\fastapi\.env
    pause
    exit /b 1
)

:: CHECK FRONTEND .ENV
if not exist "front-end\streamlit\.env" (
    echo [ERROR] Frontend .env nao encontrado.
    echo.
    echo SOLUCAO:
    echo copy front-end\streamlit\envExample.txt front-end\streamlit\.env
    pause
    exit /b 1
)

echo.
echo ==================================
echo BACKEND SETUP
echo ==================================

cd back-end\fastapi

if not exist ".venv" (
    echo Criando venv backend...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Instalando dependencias backend...
pip install --upgrade pip
pip install -r requirements.txt

echo Iniciando FastAPI...
start "Smart VET API" cmd /k "call .venv\Scripts\activate.bat && uvicorn main:app --reload"

cd ..\..

echo.
echo ==================================
echo FRONTEND SETUP
echo ==================================

cd front-end\streamlit

if not exist ".venv" (
    echo Criando venv frontend...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Instalando dependencias frontend...
pip install --upgrade pip
pip install -r requirements.txt

echo Iniciando Streamlit...
start "Smart VET UI" cmd /k "call .venv\Scripts\activate.bat && python -m streamlit run app.py"

cd ..\..

echo.
echo ==================================
echo SMART VET ONLINE
echo ==================================
echo FastAPI Docs:
echo http://localhost:8000/docs
echo.
echo Streamlit UI:
echo http://localhost:8501
echo.
pause