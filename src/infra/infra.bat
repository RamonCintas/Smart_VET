@echo off
setlocal EnableExtensions EnableDelayedExpansion

title SMART VET - Infrastructure Bootstrap

REM ============================================================
REM SMART VET
REM INFRASTRUCTURE BOOTSTRAP
REM ============================================================

echo.
echo ============================================================
echo SMART VET
echo Infrastructure Bootstrap
echo ============================================================
echo.

REM ============================================================
REM DIRETORIOS
REM ============================================================

pushd "%~dp0"

set "INFRA_DIR=."
set "SRC_DIR=.."
set "PROJECT_DIR=..\.."

set "FASTAPI_DIR=..\back-end\fastapi"
set "STREAMLIT_DIR=..\front-end\streamlit"

set "FASTAPI_DOCKERFILE=..\back-end\fastapi\Dockerfile"
set "STREAMLIT_DOCKERFILE=..\front-end\streamlit\Dockerfile"

set "ENV_FILE=.\.env"
set "TERRAFORM_FILE=.\main.tf"

echo [INFO] Diretorio atual:
cd
echo.

echo [INFO] Infra:
echo %INFRA_DIR%
echo.

echo [INFO] SRC:
echo %SRC_DIR%
echo.

echo [INFO] Project:
echo %PROJECT_DIR%
echo.

echo [INFO] FastAPI:
echo %FASTAPI_DIR%
echo.

echo [INFO] Streamlit:
echo %STREAMLIT_DIR%
echo.

echo [INFO] .env:
echo %ENV_FILE%
echo.

echo [INFO] main.tf:
echo %TERRAFORM_FILE%
echo.

REM ============================================================
REM VALIDAR DIRETORIOS
REM ============================================================

echo ============================================================
echo VALIDACAO DA ESTRUTURA
echo ============================================================
echo.

if not exist "%INFRA_DIR%\." (
    echo [ERROR] Diretorio infra nao encontrado.
    popd
    pause
    exit /b 1
)

if not exist "%SRC_DIR%\." (
    echo [ERROR] Diretorio src nao encontrado:
    echo %SRC_DIR%
    popd
    pause
    exit /b 1
)

if not exist "%PROJECT_DIR%\." (
    echo [ERROR] Diretorio raiz do projeto nao encontrado:
    echo %PROJECT_DIR%
    popd
    pause
    exit /b 1
)

if not exist "%FASTAPI_DIR%\." (
    echo [ERROR] Diretorio FastAPI nao encontrado:
    echo %FASTAPI_DIR%
    popd
    pause
    exit /b 1
)

if not exist "%STREAMLIT_DIR%\." (
    echo [ERROR] Diretorio Streamlit nao encontrado:
    echo %STREAMLIT_DIR%
    popd
    pause
    exit /b 1
)

echo [OK] Infra encontrado.
echo [OK] SRC encontrado.
echo [OK] Projeto encontrado.
echo [OK] FastAPI encontrado.
echo [OK] Streamlit encontrado.
echo.

REM ============================================================
REM VALIDAR ARQUIVOS
REM ============================================================

echo ============================================================
echo VERIFICANDO ARQUIVOS OBRIGATORIOS
echo ============================================================
echo.

if not exist "%ENV_FILE%" (
    echo [ERROR] Arquivo .env nao encontrado:
    echo %ENV_FILE%
    popd
    pause
    exit /b 1
)

if not exist "%TERRAFORM_FILE%" (
    echo [ERROR] main.tf nao encontrado:
    echo %TERRAFORM_FILE%
    popd
    pause
    exit /b 1
)

if not exist "%FASTAPI_DOCKERFILE%" (
    echo [ERROR] Dockerfile FastAPI nao encontrado:
    echo %FASTAPI_DOCKERFILE%
    popd
    pause
    exit /b 1
)

if not exist "%STREAMLIT_DOCKERFILE%" (
    echo [ERROR] Dockerfile Streamlit nao encontrado:
    echo %STREAMLIT_DOCKERFILE%
    popd
    pause
    exit /b 1
)

echo [OK] .env encontrado.
echo [OK] main.tf encontrado.
echo [OK] Dockerfile FastAPI encontrado.
echo [OK] Dockerfile Streamlit encontrado.
echo.

REM ============================================================
REM CARREGAR .ENV
REM ============================================================

echo [1/8] Carregando configuracoes do .env...
echo.

for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
    if not "%%A"=="" (
        if not "%%A:~0,1%"=="#" (
            set "%%A=%%B"
        )
    )
)

echo [OK] Configuracoes carregadas.
echo.

REM ============================================================
REM VALIDAR CONFIGURACOES OBRIGATORIAS
REM ============================================================

echo ============================================================
echo VALIDANDO CONFIGURACOES
echo ============================================================
echo.

if "%GITHUB_OWNER%"=="" (
    echo [ERROR] GITHUB_OWNER nao configurado.
    popd
    pause
    exit /b 1
)

if "%GITHUB_TOKEN%"=="" (
    echo [ERROR] GITHUB_TOKEN nao configurado.
    popd
    pause
    exit /b 1
)

if "%GITHUB_REPOSITORY%"=="" (
    echo [ERROR] GITHUB_REPOSITORY nao configurado.
    popd
    pause
    exit /b 1
)

if "%GITHUB_VISIBILITY%"=="" (
    echo [ERROR] GITHUB_VISIBILITY nao configurado.
    popd
    pause
    exit /b 1
)

if "%RENDER_API_KEY%"=="" (
    echo [ERROR] RENDER_API_KEY nao configurado.
    popd
    pause
    exit /b 1
)

if "%RENDER_OWNER_ID%"=="" (
    echo [ERROR] RENDER_OWNER_ID nao configurado.
    popd
    pause
    exit /b 1
)

if "%GIT_USER_NAME%"=="" (
    echo [ERROR] GIT_USER_NAME nao configurado.
    popd
    pause
    exit /b 1
)

if "%GIT_USER_EMAIL%"=="" (
    echo [ERROR] GIT_USER_EMAIL nao configurado.
    popd
    pause
    exit /b 1
)

if "%FASTAPI_SERVICE_NAME%"=="" (
    echo [ERROR] FASTAPI_SERVICE_NAME nao configurado.
    popd
    pause
    exit /b 1
)

if "%STREAMLIT_SERVICE_NAME%"=="" (
    echo [ERROR] STREAMLIT_SERVICE_NAME nao configurado.
    popd
    pause
    exit /b 1
)

echo [OK] Configuracoes obrigatorias presentes.
echo.

REM ============================================================
REM FERRAMENTAS
REM ============================================================

echo [2/8] Verificando ferramentas...
echo.

where terraform >nul 2>&1

if errorlevel 1 (
    echo [ERROR] Terraform nao encontrado no PATH.
    popd
    pause
    exit /b 1
)

where git >nul 2>&1

if errorlevel 1 (
    echo [ERROR] Git nao encontrado no PATH.
    popd
    pause
    exit /b 1
)

echo [OK] Terraform:
terraform version | findstr /r "^Terraform"

echo [OK] Git:
git --version

echo.

REM ============================================================
REM CONFIGURAR VARIAVEIS TERRAFORM
REM ============================================================

echo [3/8] Configurando variaveis do Terraform...
echo.

REM ------------------------------------------------------------
REM GITHUB
REM ------------------------------------------------------------

set "TF_VAR_github_owner=%GITHUB_OWNER%"
set "TF_VAR_github_token=%GITHUB_TOKEN%"
set "TF_VAR_github_repository=%GITHUB_REPOSITORY%"
set "TF_VAR_github_visibility=%GITHUB_VISIBILITY%"

REM ------------------------------------------------------------
REM RENDER
REM ------------------------------------------------------------

set "TF_VAR_render_api_key=%RENDER_API_KEY%"
set "TF_VAR_render_owner_id=%RENDER_OWNER_ID%"
set "TF_VAR_render_region=%RENDER_REGION%"
set "TF_VAR_render_plan=%RENDER_PLAN%"

REM ------------------------------------------------------------
REM FASTAPI
REM ------------------------------------------------------------

set "TF_VAR_fastapi_service_name=%FASTAPI_SERVICE_NAME%"
set "TF_VAR_fastapi_branch=%FASTAPI_BRANCH%"
set "TF_VAR_fastapi_root_directory=%FASTAPI_ROOT_DIRECTORY%"

REM ------------------------------------------------------------
REM STREAMLIT
REM ------------------------------------------------------------

set "TF_VAR_streamlit_service_name=%STREAMLIT_SERVICE_NAME%"
set "TF_VAR_streamlit_branch=%STREAMLIT_BRANCH%"
set "TF_VAR_streamlit_root_directory=%STREAMLIT_ROOT_DIRECTORY%"

REM ------------------------------------------------------------
REM API
REM ------------------------------------------------------------

set "TF_VAR_api_env=%API_ENV%"
set "TF_VAR_API_KEY=%API_KEY%"
set "TF_VAR_allowed_origins=%ALLOWED_ORIGINS%"
set "TF_VAR_fastapi_url=%FASTAPI_URL%"

REM ------------------------------------------------------------
REM OUTRAS CONFIGURACOES API
REM ------------------------------------------------------------

set "TF_VAR_SMART_VET_API_KEY=%SMART_VET_API_KEY%"
set "TF_VAR_max_upload_bytes=%MAX_UPLOAD_BYTES%"

REM ------------------------------------------------------------
REM HUGGING FACE
REM ------------------------------------------------------------

set "TF_VAR_hf_data_repo=%HF_DATA_REPO%"
set "TF_VAR_hf_data_filename=%HF_DATA_FILENAME%"

set "TF_VAR_hf_image_repo=%HF_IMAGE_REPO%"
set "TF_VAR_hf_image_filename=%HF_IMAGE_FILENAME%"

set "TF_VAR_hf_model_repo=%HF_MODEL_REPO%"
set "TF_VAR_hf_model_filename=%HF_MODEL_FILENAME%"

REM ------------------------------------------------------------
REM LLM PRINCIPAL
REM ------------------------------------------------------------

set "TF_VAR_llm_provider=%LLM_PROVIDER%"
set "TF_VAR_llm_api_key=%LLM_API_KEY%"
set "TF_VAR_llm_base_url=%LLM_BASE_URL%"
set "TF_VAR_llm_model=%LLM_MODEL%"
set "TF_VAR_llm_json_mode=%LLM_JSON_MODE%"

REM ------------------------------------------------------------
REM LLM FALLBACK 1
REM ------------------------------------------------------------

set "TF_VAR_llm_fallback_1_provider=%LLM_FALLBACK_1_PROVIDER%"
set "TF_VAR_llm_fallback_1_api_key=%LLM_FALLBACK_1_API_KEY%"
set "TF_VAR_llm_fallback_1_base_url=%LLM_FALLBACK_1_BASE_URL%"
set "TF_VAR_llm_fallback_1_model=%LLM_FALLBACK_1_MODEL%"
set "TF_VAR_llm_fallback_1_json_mode=%LLM_FALLBACK_1_JSON_MODE%"

REM ------------------------------------------------------------
REM LLM CONFIGURACAO
REM ------------------------------------------------------------

set "TF_VAR_llm_timeout_seconds=%LLM_TIMEOUT_SECONDS%"
set "TF_VAR_llm_temperature=%LLM_TEMPERATURE%"

echo [OK] Variaveis Terraform configuradas.
echo.

REM ============================================================
REM INICIALIZAR TERRAFORM
REM ============================================================

echo [4/8] Terraform Init...
echo.

terraform init

if errorlevel 1 (
    echo.
    echo [ERROR] Terraform init falhou.
    popd
    pause
    exit /b 1
)

echo.
echo [OK] Terraform init concluido.
echo.

REM ============================================================
REM FORMATAR TERRAFORM
REM ============================================================

echo [5/8] Terraform Format...
echo.

terraform fmt

if errorlevel 1 (
    echo.
    echo [ERROR] Terraform fmt falhou.
    popd
    pause
    exit /b 1
)

echo.
echo [OK] Terraform fmt concluido.
echo.

REM ============================================================
REM VALIDAR TERRAFORM
REM ============================================================

echo [6/8] Terraform Validate...
echo.

terraform validate

if errorlevel 1 (
    echo.
    echo [ERROR] Terraform validate falhou.
    popd
    pause
    exit /b 1
)

echo.
echo [OK] Terraform configuration valida.
echo.

REM ============================================================
REM CRIAR REPOSITORIO GITHUB
REM ============================================================

echo [7/8] Criando repositorio GitHub...
echo.

terraform apply -auto-approve -target=github_repository.smart_vet

if errorlevel 1 (
    echo.
    echo [ERROR] Criacao do repositorio GitHub falhou.
    echo.
    popd
    pause
    exit /b 1
)

echo.
echo [OK] Repositorio GitHub criado/verificado.
echo.

REM ============================================================
REM OBTER URL DO GITHUB
REM ============================================================

echo [INFO] Obtendo URL do repositorio GitHub...
echo.

set "REPOSITORY_URL="

for /f "delims=" %%A in ('terraform output -raw github_clone_url 2^>nul') do (
    set "REPOSITORY_URL=%%A"
)

if "!REPOSITORY_URL!"=="" (
    echo.
    echo [ERROR] Nao foi possivel obter a URL do repositorio.
    echo.
    popd
    pause
    exit /b 1
)

echo [OK] Repository:
echo !REPOSITORY_URL!
echo.

REM ============================================================
REM CONFIGURAR GIT LOCAL
REM ============================================================

echo [INFO] Configurando Git local...
echo.

cd /d "%PROJECT_DIR%"

if not exist ".git" (
    echo [INFO] Inicializando repositorio Git...

    git init

    if errorlevel 1 (
        echo.
        echo [ERROR] git init falhou.
        popd
        pause
        exit /b 1
    )
)

git config user.name "%GIT_USER_NAME%"

if errorlevel 1 (
    echo.
    echo [ERROR] Nao foi possivel configurar GIT_USER_NAME.
    popd
    pause
    exit /b 1
)

git config user.email "%GIT_USER_EMAIL%"

if errorlevel 1 (
    echo.
    echo [ERROR] Nao foi possivel configurar GIT_USER_EMAIL.
    popd
    pause
    exit /b 1
)

git branch -M main

REM ============================================================
REM CONFIGURAR ORIGIN
REM ============================================================

git remote remove origin >nul 2>&1

git remote add origin "!REPOSITORY_URL!"

if errorlevel 1 (
    echo.
    echo [ERROR] Nao foi possivel configurar o remote origin.
    popd
    pause
    exit /b 1
)

echo [OK] Git configurado.
echo.

REM ============================================================
REM GITIGNORE
REM ============================================================

if not exist ".gitignore" (

    echo [INFO] Criando .gitignore...

    (
        echo # Environment
        echo .env
        echo .env.*
        echo !.env.example
        echo.
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo .pytest_cache/
        echo .mypy_cache/
        echo .ruff_cache/
        echo .coverage
        echo htmlcov/
        echo.
        echo # Virtual environments
        echo .venv/
        echo venv/
        echo env/
        echo.
        echo # Terraform
        echo .terraform/
        echo *.tfstate
        echo *.tfstate.*
        echo *.tfvars
        echo *.tfvars.json
        echo *.tfplan
        echo crash.log
        echo crash.*.log
        echo.
        echo # IDE
        echo .vscode/
        echo .idea/
        echo.
        echo # Logs
        echo *.log
    ) > ".gitignore"

    if errorlevel 1 (
        echo.
        echo [ERROR] Nao foi possivel criar .gitignore.
        popd
        pause
        exit /b 1
    )

    echo [OK] .gitignore criado.
)

echo.

REM ============================================================
REM GIT ADD
REM ============================================================

echo [INFO] Adicionando projeto ao Git...
echo.

git add .

if errorlevel 1 (
    echo.
    echo [ERROR] git add falhou.
    popd
    pause
    exit /b 1
)

echo [OK] Arquivos adicionados.
echo.

REM ============================================================
REM GIT COMMIT
REM ============================================================

echo [INFO] Criando commit inicial...
echo.

git diff --cached --quiet

if errorlevel 1 (

    git commit -m "Initial Smart VET bootstrap"

    if errorlevel 1 (
        echo.
        echo [ERROR] git commit falhou.
        popd
        pause
        exit /b 1
    )

    echo [OK] Commit criado.

) else (

    echo [INFO] Nenhuma alteracao nova para commit.

)

echo.

REM ============================================================
REM AUTENTICACAO GITHUB
REM ============================================================

echo [INFO] Preparando autenticacao GitHub...
echo.

set "GIT_ASKPASS=%TEMP%\smart_vet_git_askpass_%RANDOM%_%RANDOM%.cmd"
set "GIT_TERMINAL_PROMPT=0"

(
    echo @echo off
    echo echo %%1 ^| findstr /I "Username" ^>nul
    echo if not errorlevel 1 ^(
    echo     echo %GITHUB_OWNER%
    echo ^) else ^(
    echo     echo %GITHUB_TOKEN%
    echo ^)
) > "!GIT_ASKPASS!"

if not exist "!GIT_ASKPASS!" (
    echo.
    echo [ERROR] Nao foi possivel preparar autenticacao GitHub.
    popd
    pause
    exit /b 1
)

echo [OK] Autenticacao preparada.
echo.

REM ============================================================
REM PUSH GITHUB
REM ============================================================

echo [INFO] Enviando projeto para GitHub...
echo.

git -c credential.helper= push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] git push falhou.
    echo.

    del /q /f "!GIT_ASKPASS!" >nul 2>&1

    set "GIT_ASKPASS="
    set "GIT_TERMINAL_PROMPT="

    popd
    pause
    exit /b 1
)

echo.
echo [OK] Projeto enviado para GitHub.
echo.

REM ============================================================
REM LIMPAR AUTENTICACAO
REM ============================================================

del /q /f "!GIT_ASKPASS!" >nul 2>&1

set "GIT_ASKPASS="
set "GIT_TERMINAL_PROMPT="

REM ============================================================
REM VOLTAR PARA INFRA
REM ============================================================

cd /d "%~dp0"

REM ============================================================
REM RENDER - UNICO APPLY COMPLETO
REM ============================================================

echo [8/8] Criando infraestrutura Render...
echo.

echo ============================================================
echo APLICANDO TERRAFORM
echo ============================================================
echo.
echo O Terraform ira criar:
echo.
echo - FastAPI
echo - Streamlit
echo.
echo As configuracoes de ambiente serao aplicadas
echo durante a criacao dos servicos.
echo.
echo Nao sera executado um segundo terraform apply.
echo ============================================================
echo.

terraform apply -auto-approve

if errorlevel 1 (
    echo.
    echo [ERROR] Terraform apply falhou.
    echo.
    popd
    pause
    exit /b 1
)

echo.
echo [OK] Infraestrutura Render criada/verificada.
echo.

REM ============================================================
REM OBTER URLS
REM ============================================================

echo [INFO] Obtendo URLs dos servicos Render...
echo.

set "GENERATED_FASTAPI_URL="
set "GENERATED_STREAMLIT_URL="

for /f "delims=" %%A in ('terraform output -raw fastapi_url 2^>nul') do (
    set "GENERATED_FASTAPI_URL=%%A"
)

for /f "delims=" %%A in ('terraform output -raw streamlit_url 2^>nul') do (
    set "GENERATED_STREAMLIT_URL=%%A"
)

if "!GENERATED_FASTAPI_URL!"=="" (
    echo.
    echo [ERROR] URL do FastAPI nao encontrada.
    echo.
    popd
    pause
    exit /b 1
)

if "!GENERATED_STREAMLIT_URL!"=="" (
    echo.
    echo [ERROR] URL do Streamlit nao encontrada.
    echo.
    popd
    pause
    exit /b 1
)

echo [OK] FastAPI:
echo !GENERATED_FASTAPI_URL!
echo.

echo [OK] Streamlit:
echo !GENERATED_STREAMLIT_URL!
echo.

REM ============================================================
REM FINAL
REM ============================================================

echo.
echo ============================================================
echo SMART VET CONFIGURADO COM SUCESSO
echo ============================================================
echo.

echo GitHub:
echo !REPOSITORY_URL!

echo.
echo FastAPI:
echo !GENERATED_FASTAPI_URL!

echo.
echo Streamlit:
echo !GENERATED_STREAMLIT_URL!

echo.
echo ALLOWED_ORIGINS:
echo configurado pelo Terraform

echo.
echo FASTAPI_URL:
echo configurado pelo Terraform

echo.
echo LLM:
echo configuracao enviada ao Terraform

echo.
echo ============================================================
echo.
echo Bootstrap inicial concluido.
echo.
echo Estrutura:
echo.
echo FastAPI:
echo src\back-end\fastapi
echo.
echo Streamlit:
echo src\front-end\streamlit
echo.
echo Infra:
echo src\infra
echo.
echo ============================================================
echo.

popd

pause
exit /b 0
