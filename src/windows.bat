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
    echo Instale Python 3.13.
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 13) else 1)"
if errorlevel 1 (
    echo [ERROR] Versao do Python nao suportada:
    python --version
    echo Instale Python 3.13.
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

cd /d "%~dp0back-end\fastapi"

if not exist ".venv" (
    echo Criando venv backend...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Instalando dependencias backend...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ==================================
echo INICIANDO FASTAPI
echo ==================================

start "Smart VET API" cmd /k "cd /d %~dp0back-end\fastapi && call .venv\Scripts\activate.bat && uvicorn main:app --host 127.0.0.1 --port 8000"

echo.
echo Aguardando FastAPI iniciar...
echo.

set "API_READY=0"

for /L %%i in (1,1,60) do (
    powershell -NoProfile -Command "try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"

    if not errorlevel 1 (
        set "API_READY=1"
        goto API_READY
    )

    echo Aguardando API... tentativa %%i/60
    timeout /t 2 /nobreak >nul
)

:API_READY

if "%API_READY%"=="0" (
    echo.
    echo ============================================================
    echo [ERROR] FASTAPI NAO INICIOU
    echo ============================================================
    echo.
    echo A API nao respondeu em:
    echo http://localhost:8000/health
    echo.
    echo O Streamlit nao sera iniciado.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [OK] FASTAPI ONLINE
echo ============================================================
echo.
echo http://localhost:8000/health
echo.

cd /d "%~dp0"

echo.
echo ==================================
echo FRONTEND SETUP
echo ==================================

if not exist "%~dp0front-end\streamlit\" (
    echo [ERROR] Pasta Streamlit nao encontrada.
    echo Caminho esperado:
    echo %~dp0front-end\streamlit
    pause
    exit /b 1
)

cd /d "%~dp0front-end\streamlit"

if not exist ".venv" (
    echo Criando venv frontend...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Instalando dependencias frontend...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Iniciando Streamlit...
start "Smart VET UI" cmd /k "cd /d %~dp0front-end\streamlit && call .venv\Scripts\activate.bat && python -m streamlit run app.py"

cd /d "%~dp0"

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
