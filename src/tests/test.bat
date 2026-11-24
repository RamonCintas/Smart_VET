@echo off
title Smart VET - Test Runner
setlocal

echo ==================================
echo Smart VET - TEST RUNNER
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

:: CHECK TESTS .ENV
if not exist ".env" (
    echo [ERROR] Tests .env nao encontrado.
    echo.
    echo SOLUCAO:
    echo copy envExample.txt .env
    echo Configure TEST_API_KEY com o mesmo valor de API_KEY no backend.
    pause
    exit /b 1
)

:: CHECK TEST FILE
if not exist "test_api.py" (
    echo [ERROR] test_api.py nao encontrado.
    pause
    exit /b 1
)

echo.
echo ==================================
echo TEST ENVIRONMENT SETUP
echo ==================================

if not exist ".venv" (
    echo Criando venv de testes...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Instalando dependencias de testes...
python -m pip install --upgrade pip

if errorlevel 1 (
    echo [ERROR] Falha ao atualizar pip.
    pause
    exit /b 1
)

python -m pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Falha ao instalar dependencias.
    pause
    exit /b 1
)

echo.
echo ==================================
echo EXECUTANDO PYTEST
echo ==================================
echo.

python -m pytest test_api.py ^
    -vv ^
    -s ^
    --capture=no ^
    --log-cli-level=DEBUG ^
    --tb=long

set "PYTEST_EXIT_CODE=%ERRORLEVEL%"

echo.
echo ==================================
echo PYTEST FINALIZADO
echo ==================================
echo.

if not "%PYTEST_EXIT_CODE%"=="0" (
    echo [ERROR] OS TESTES FALHARAM.
    echo Exit code: %PYTEST_EXIT_CODE%
    echo.
    pause
    exit /b %PYTEST_EXIT_CODE%
)

echo [OK] TODOS OS TESTES PASSARAM!
echo.

pause
exit /b 0
