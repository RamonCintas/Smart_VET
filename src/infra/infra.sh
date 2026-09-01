#!/usr/bin/env bash

# ============================================================
# SMART VET
# INFRASTRUCTURE BOOTSTRAP - LINUX
# ============================================================

set -Eeuo pipefail

# ============================================================
# CORES
# ============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================
# FUNÇÕES
# ============================================================

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

ok() {
    echo -e "${GREEN}[OK]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

fail() {
    error "$*"
    exit 1
}

cleanup() {
    if [[ -n "${GIT_ASKPASS:-}" && -f "${GIT_ASKPASS}" ]]; then
        rm -f "${GIT_ASKPASS}" || true
    fi

    unset GIT_ASKPASS
    unset GIT_TERMINAL_PROMPT
}

trap cleanup EXIT

# ============================================================
# CABEÇALHO
# ============================================================

echo
echo "============================================================"
echo "SMART VET"
echo "Infrastructure Bootstrap - Linux"
echo "============================================================"
echo

# ============================================================
# DIRETÓRIO DO SCRIPT
# ============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

INFRA_DIR="."
SRC_DIR=".."
PROJECT_DIR="../.."

FASTAPI_DIR="../back-end/fastapi"
STREAMLIT_DIR="../front-end/streamlit"

FASTAPI_DOCKERFILE="../back-end/fastapi/Dockerfile"
STREAMLIT_DOCKERFILE="../front-end/streamlit/Dockerfile"

ENV_FILE="./.env"
TERRAFORM_FILE="./main.tf"

info "Diretório atual:"
pwd
echo

info "Infra:"
echo "${INFRA_DIR}"
echo

info "SRC:"
echo "${SRC_DIR}"
echo

info "Project:"
echo "${PROJECT_DIR}"
echo

info "FastAPI:"
echo "${FASTAPI_DIR}"
echo

info "Streamlit:"
echo "${STREAMLIT_DIR}"
echo

info ".env:"
echo "${ENV_FILE}"
echo

info "main.tf:"
echo "${TERRAFORM_FILE}"
echo

# ============================================================
# VALIDAR DIRETÓRIOS
# ============================================================

echo "============================================================"
echo "VALIDAÇÃO DA ESTRUTURA"
echo "============================================================"
echo

[[ -d "${INFRA_DIR}" ]] \
    || fail "Diretório infra não encontrado."

[[ -d "${SRC_DIR}" ]] \
    || fail "Diretório src não encontrado: ${SRC_DIR}"

[[ -d "${PROJECT_DIR}" ]] \
    || fail "Diretório raiz do projeto não encontrado: ${PROJECT_DIR}"

[[ -d "${FASTAPI_DIR}" ]] \
    || fail "Diretório FastAPI não encontrado: ${FASTAPI_DIR}"

[[ -d "${STREAMLIT_DIR}" ]] \
    || fail "Diretório Streamlit não encontrado: ${STREAMLIT_DIR}"

ok "Infra encontrado."
ok "SRC encontrado."
ok "Projeto encontrado."
ok "FastAPI encontrado."
ok "Streamlit encontrado."
echo

# ============================================================
# VALIDAR ARQUIVOS
# ============================================================

echo "============================================================"
echo "VERIFICANDO ARQUIVOS OBRIGATÓRIOS"
echo "============================================================"
echo

[[ -f "${ENV_FILE}" ]] \
    || fail "Arquivo .env não encontrado: ${ENV_FILE}"

[[ -f "${TERRAFORM_FILE}" ]] \
    || fail "main.tf não encontrado: ${TERRAFORM_FILE}"

[[ -f "${FASTAPI_DOCKERFILE}" ]] \
    || fail "Dockerfile FastAPI não encontrado: ${FASTAPI_DOCKERFILE}"

[[ -f "${STREAMLIT_DOCKERFILE}" ]] \
    || fail "Dockerfile Streamlit não encontrado: ${STREAMLIT_DOCKERFILE}"

ok ".env encontrado."
ok "main.tf encontrado."
ok "Dockerfile FastAPI encontrado."
ok "Dockerfile Streamlit encontrado."
echo

# ============================================================
# CARREGAR .ENV
# ============================================================

echo "[1/8] Carregando configurações do .env..."
echo

# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

ok "Configurações carregadas."
echo

# ============================================================
# VALIDAR CONFIGURAÇÕES OBRIGATÓRIAS
# ============================================================

echo "============================================================"
echo "VALIDANDO CONFIGURAÇÕES"
echo "============================================================"
echo

REQUIRED_VARS=(
    GITHUB_OWNER
    GITHUB_TOKEN
    GITHUB_REPOSITORY
    GITHUB_VISIBILITY

    RENDER_API_KEY
    RENDER_OWNER_ID

    GIT_USER_NAME
    GIT_USER_EMAIL

    FASTAPI_SERVICE_NAME
    STREAMLIT_SERVICE_NAME
)

for VAR_NAME in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!VAR_NAME:-}" ]]; then
        fail "${VAR_NAME} não configurado."
    fi
done

ok "Configurações obrigatórias presentes."
echo

# ============================================================
# FERRAMENTAS
# ============================================================

echo "[2/8] Verificando ferramentas..."
echo

command -v terraform >/dev/null 2>&1 \
    || fail "Terraform não encontrado no PATH."

command -v git >/dev/null 2>&1 \
    || fail "Git não encontrado no PATH."

ok "Terraform:"
terraform version | head -n 1

echo

ok "Git:"
git --version

echo

# ============================================================
# CONFIGURAR VARIÁVEIS TERRAFORM
# ============================================================

echo "[3/8] Configurando variáveis do Terraform..."
echo

# ------------------------------------------------------------
# GITHUB
# ------------------------------------------------------------

export TF_VAR_github_owner="${GITHUB_OWNER}"
export TF_VAR_github_token="${GITHUB_TOKEN}"
export TF_VAR_github_repository="${GITHUB_REPOSITORY}"
export TF_VAR_github_visibility="${GITHUB_VISIBILITY}"

# ------------------------------------------------------------
# RENDER
# ------------------------------------------------------------

export TF_VAR_render_api_key="${RENDER_API_KEY}"
export TF_VAR_render_owner_id="${RENDER_OWNER_ID}"
export TF_VAR_render_region="${RENDER_REGION:-}"
export TF_VAR_render_plan="${RENDER_PLAN:-}"

# ------------------------------------------------------------
# FASTAPI
# ------------------------------------------------------------

export TF_VAR_fastapi_service_name="${FASTAPI_SERVICE_NAME}"
export TF_VAR_fastapi_branch="${FASTAPI_BRANCH:-}"
export TF_VAR_fastapi_root_directory="${FASTAPI_ROOT_DIRECTORY:-}"

# ------------------------------------------------------------
# STREAMLIT
# ------------------------------------------------------------

export TF_VAR_streamlit_service_name="${STREAMLIT_SERVICE_NAME}"
export TF_VAR_streamlit_branch="${STREAMLIT_BRANCH:-}"
export TF_VAR_streamlit_root_directory="${STREAMLIT_ROOT_DIRECTORY:-}"

# ------------------------------------------------------------
# API
# ------------------------------------------------------------

export TF_VAR_api_env="${API_ENV:-}"
export TF_VAR_API_KEY="${API_KEY:-}"
export TF_VAR_allowed_origins="${ALLOWED_ORIGINS:-}"
export TF_VAR_fastapi_url="${FASTAPI_URL:-}"

# ------------------------------------------------------------
# OUTRAS CONFIGURAÇÕES API
# ------------------------------------------------------------

export TF_VAR_SMART_VET_API_KEY="${SMART_VET_API_KEY:-}"
export TF_VAR_max_upload_bytes="${MAX_UPLOAD_BYTES:-}"

# ------------------------------------------------------------
# HUGGING FACE
# ------------------------------------------------------------

export TF_VAR_hf_data_repo="${HF_DATA_REPO:-}"
export TF_VAR_hf_data_filename="${HF_DATA_FILENAME:-}"

export TF_VAR_hf_image_repo="${HF_IMAGE_REPO:-}"
export TF_VAR_hf_image_filename="${HF_IMAGE_FILENAME:-}"

export TF_VAR_hf_model_repo="${HF_MODEL_REPO:-}"
export TF_VAR_hf_model_filename="${HF_MODEL_FILENAME:-}"

# ------------------------------------------------------------
# LLM PRINCIPAL
# ------------------------------------------------------------

export TF_VAR_llm_provider="${LLM_PROVIDER:-}"
export TF_VAR_llm_api_key="${LLM_API_KEY:-}"
export TF_VAR_llm_base_url="${LLM_BASE_URL:-}"
export TF_VAR_llm_model="${LLM_MODEL:-}"
export TF_VAR_llm_json_mode="${LLM_JSON_MODE:-}"

# ------------------------------------------------------------
# LLM FALLBACK 1
# ------------------------------------------------------------

export TF_VAR_llm_fallback_1_provider="${LLM_FALLBACK_1_PROVIDER:-}"
export TF_VAR_llm_fallback_1_api_key="${LLM_FALLBACK_1_API_KEY:-}"
export TF_VAR_llm_fallback_1_base_url="${LLM_FALLBACK_1_BASE_URL:-}"
export TF_VAR_llm_fallback_1_model="${LLM_FALLBACK_1_MODEL:-}"
export TF_VAR_llm_fallback_1_json_mode="${LLM_FALLBACK_1_JSON_MODE:-}"

# ------------------------------------------------------------
# LLM CONFIGURAÇÃO
# ------------------------------------------------------------

export TF_VAR_llm_timeout_seconds="${LLM_TIMEOUT_SECONDS:-}"
export TF_VAR_llm_temperature="${LLM_TEMPERATURE:-}"

ok "Variáveis Terraform configuradas."
echo

# ============================================================
# INICIALIZAR TERRAFORM
# ============================================================

echo "[4/8] Terraform Init..."
echo

terraform init \
    || fail "Terraform init falhou."

echo
ok "Terraform init concluído."
echo

# ============================================================
# FORMATAR TERRAFORM
# ============================================================

echo "[5/8] Terraform Format..."
echo

terraform fmt \
    || fail "Terraform fmt falhou."

echo
ok "Terraform fmt concluído."
echo

# ============================================================
# VALIDAR TERRAFORM
# ============================================================

echo "[6/8] Terraform Validate..."
echo

terraform validate \
    || fail "Terraform validate falhou."

echo
ok "Terraform configuration válida."
echo

# ============================================================
# CRIAR REPOSITÓRIO GITHUB
# ============================================================

echo "[7/8] Criando repositório GitHub..."
echo

terraform apply \
    -auto-approve \
    -target=github_repository.smart_vet \
    || fail "Criação do repositório GitHub falhou."

echo
ok "Repositório GitHub criado/verificado."
echo

# ============================================================
# OBTER URL DO GITHUB
# ============================================================

info "Obtendo URL do repositório GitHub..."
echo

REPOSITORY_URL="$(
    terraform output -raw github_clone_url 2>/dev/null || true
)"

[[ -n "${REPOSITORY_URL}" ]] \
    || fail "Não foi possível obter a URL do repositório."

ok "Repository:"
echo "${REPOSITORY_URL}"
echo

# ============================================================
# CONFIGURAR GIT LOCAL
# ============================================================

info "Configurando Git local..."
echo

cd "${PROJECT_DIR}"

if [[ ! -d ".git" ]]; then

    info "Inicializando repositório Git..."

    git init \
        || fail "git init falhou."
fi

git config user.name "${GIT_USER_NAME}" \
    || fail "Não foi possível configurar GIT_USER_NAME."

git config user.email "${GIT_USER_EMAIL}" \
    || fail "Não foi possível configurar GIT_USER_EMAIL."

git branch -M main

# ============================================================
# CONFIGURAR ORIGIN
# ============================================================

git remote remove origin >/dev/null 2>&1 || true

git remote add origin "${REPOSITORY_URL}" \
    || fail "Não foi possível configurar o remote origin."

ok "Git configurado."
echo

# ============================================================
# GITIGNORE
# ============================================================

if [[ ! -f ".gitignore" ]]; then

    info "Criando .gitignore..."

    cat > ".gitignore" <<'EOF'
# Environment
.env
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Virtual environments
.venv/
venv/
env/

# Terraform
.terraform/
*.tfstate
*.tfstate.*
*.tfvars
*.tfvars.json
*.tfplan
crash.log
crash.*.log

# IDE
.vscode/
.idea/

# Logs
*.log
EOF

    ok ".gitignore criado."
fi

echo

# ============================================================
# GIT ADD
# ============================================================

info "Adicionando projeto ao Git..."
echo

git add . \
    || fail "git add falhou."

ok "Arquivos adicionados."
echo

# ============================================================
# GIT COMMIT
# ============================================================

info "Criando commit inicial..."
echo

if ! git diff --cached --quiet; then

    git commit -m "Initial Smart VET bootstrap" \
        || fail "git commit falhou."

    ok "Commit criado."

else

    info "Nenhuma alteração nova para commit."

fi

echo

# ============================================================
# AUTENTICAÇÃO GITHUB
# ============================================================

info "Preparando autenticação GitHub..."
echo

GIT_ASKPASS="$(mktemp "${TMPDIR:-/tmp}/smart_vet_git_askpass.XXXXXX.sh")"

cat > "${GIT_ASKPASS}" <<EOF
#!/usr/bin/env bash

case "\${1:-}" in
    *Username*)
        printf '%s\n' '${GITHUB_OWNER}'
        ;;
    *Password*|*password*|*Token*|*token*)
        printf '%s\n' '${GITHUB_TOKEN}'
        ;;
    *)
        printf '%s\n' '${GITHUB_TOKEN}'
        ;;
esac
EOF

chmod 700 "${GIT_ASKPASS}"

[[ -f "${GIT_ASKPASS}" ]] \
    || fail "Não foi possível preparar autenticação GitHub."

export GIT_ASKPASS
export GIT_TERMINAL_PROMPT=0

ok "Autenticação preparada."
echo

# ============================================================
# PUSH GITHUB
# ============================================================

info "Enviando projeto para GitHub..."
echo

git -c credential.helper= push -u origin main \
    || fail "git push falhou."

echo
ok "Projeto enviado para GitHub."
echo

# ============================================================
# LIMPAR AUTENTICAÇÃO
# ============================================================

cleanup

# ============================================================
# VOLTAR PARA INFRA
# ============================================================

cd "${SCRIPT_DIR}"

# ============================================================
# RENDER - ÚNICO APPLY COMPLETO
# ============================================================

echo "[8/8] Criando infraestrutura Render..."
echo

echo "============================================================"
echo "APLICANDO TERRAFORM"
echo "============================================================"
echo
echo "O Terraform irá criar:"
echo
echo "- FastAPI"
echo "- Streamlit"
echo
echo "As configurações de ambiente serão aplicadas"
echo "durante a criação dos serviços."
echo
echo "Não será executado um segundo terraform apply."
echo "============================================================"
echo

terraform apply -auto-approve \
    || fail "Terraform apply falhou."

echo
ok "Infraestrutura Render criada/verificada."
echo

# ============================================================
# OBTER URLS
# ============================================================

info "Obtendo URLs dos serviços Render..."
echo

GENERATED_FASTAPI_URL="$(
    terraform output -raw fastapi_url 2>/dev/null || true
)"

GENERATED_STREAMLIT_URL="$(
    terraform output -raw streamlit_url 2>/dev/null || true
)"

[[ -n "${GENERATED_FASTAPI_URL}" ]] \
    || fail "URL do FastAPI não encontrada."

[[ -n "${GENERATED_STREAMLIT_URL}" ]] \
    || fail "URL do Streamlit não encontrada."

ok "FastAPI:"
echo "${GENERATED_FASTAPI_URL}"
echo

ok "Streamlit:"
echo "${GENERATED_STREAMLIT_URL}"
echo

# ============================================================
# FINAL
# ============================================================

echo
echo "============================================================"
echo "SMART VET CONFIGURADO COM SUCESSO"
echo "============================================================"
echo

echo "GitHub:"
echo "${REPOSITORY_URL}"

echo
echo "FastAPI:"
echo "${GENERATED_FASTAPI_URL}"

echo
echo "Streamlit:"
echo "${GENERATED_STREAMLIT_URL}"

echo
echo "ALLOWED_ORIGINS:"
echo "configurado pelo Terraform"

echo
echo "FASTAPI_URL:"
echo "configurado pelo Terraform"

echo
echo "LLM:"
echo "configuração enviada ao Terraform"

echo
echo "============================================================"
echo
echo "Bootstrap inicial concluído."
echo
echo "Estrutura:"
echo
echo "FastAPI:"
echo "src/back-end/fastapi"
echo
echo "Streamlit:"
echo "src/front-end/streamlit"
echo
echo "Infra:"
echo "src/infra"
echo
echo "============================================================"
echo

exit 0
