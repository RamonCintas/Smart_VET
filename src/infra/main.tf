terraform {
  required_version = ">= 1.8.0"

  required_providers {
    github = {
      source  = "integrations/github"
      version = "~> 6.13"
    }

    render = {
      source  = "render-oss/render"
      version = "~> 1.9"
    }
  }
}

# ============================================================
# VARIABLES - GITHUB
# ============================================================

variable "github_owner" {
  type = string
}

variable "github_token" {
  type      = string
  sensitive = true
}

variable "github_repository" {
  type = string
}

variable "github_visibility" {
  type = string

  validation {
    condition = contains(
      ["private", "public"],
      var.github_visibility
    )

    error_message = "github_visibility deve ser private ou public."
  }
}

# ============================================================
# VARIABLES - RENDER
# ============================================================

variable "render_api_key" {
  type      = string
  sensitive = true
}

variable "render_owner_id" {
  type = string
}

variable "render_region" {
  type    = string
  default = "oregon"

  validation {
    condition = contains(
      [
        "oregon",
        "ohio",
        "virginia",
        "frankfurt",
        "singapore"
      ],
      var.render_region
    )

    error_message = "render_region deve ser uma região válida do Render."
  }
}

variable "render_plan" {
  type    = string
  default = "free"
}

# ============================================================
# VARIABLES - FASTAPI
# ============================================================

variable "fastapi_service_name" {
  type = string
}

variable "fastapi_branch" {
  type    = string
  default = "main"
}

variable "fastapi_root_directory" {
  type    = string
  default = "src/back-end/fastapi"
}

# ============================================================
# VARIABLES - STREAMLIT
# ============================================================

variable "streamlit_service_name" {
  type = string
}

variable "streamlit_branch" {
  type    = string
  default = "main"
}

variable "streamlit_root_directory" {
  type    = string
  default = "src/front-end/streamlit"
}

# ============================================================
# VARIABLES - API
# ============================================================

variable "api_env" {
  type    = string
  default = "production"
}

variable "API_KEY" {
  type      = string
  default   = ""
  sensitive = true
}

variable "SMART_VET_API_KEY" {
  type      = string
  default   = ""
  sensitive = true
}

variable "max_upload_bytes" {
  type    = string
  default = ""
}

variable "allowed_origins" {
  type    = string
  default = ""
}

variable "fastapi_url" {
  type    = string
  default = ""
}

# ============================================================
# VARIABLES - HUGGING FACE
# ============================================================

variable "hf_data_repo" {
  type    = string
  default = ""
}

variable "hf_data_filename" {
  type    = string
  default = ""
}

variable "hf_image_repo" {
  type    = string
  default = ""
}

variable "hf_image_filename" {
  type    = string
  default = ""
}

variable "hf_model_repo" {
  type    = string
  default = ""
}

variable "hf_model_filename" {
  type    = string
  default = ""
}

# ============================================================
# VARIABLES - LLM PRINCIPAL
# ============================================================

variable "llm_provider" {
  type    = string
  default = ""
}

variable "llm_api_key" {
  type      = string
  default   = ""
  sensitive = true
}

variable "llm_base_url" {
  type    = string
  default = ""
}

variable "llm_model" {
  type    = string
  default = ""
}

variable "llm_json_mode" {
  type    = string
  default = ""
}

# ============================================================
# VARIABLES - LLM FALLBACK 1
# ============================================================

variable "llm_fallback_1_provider" {
  type    = string
  default = ""
}

variable "llm_fallback_1_api_key" {
  type      = string
  default   = ""
  sensitive = true
}

variable "llm_fallback_1_base_url" {
  type    = string
  default = ""
}

variable "llm_fallback_1_model" {
  type    = string
  default = ""
}

variable "llm_fallback_1_json_mode" {
  type    = string
  default = ""
}

# ============================================================
# VARIABLES - LLM CONFIGURATION
# ============================================================

variable "llm_timeout_seconds" {
  type    = string
  default = ""
}

variable "llm_temperature" {
  type    = string
  default = ""
}

# ============================================================
# PROVIDER - GITHUB
# ============================================================

provider "github" {
  owner = var.github_owner
  token = var.github_token
}

# ============================================================
# PROVIDER - RENDER
# ============================================================

provider "render" {
  api_key  = var.render_api_key
  owner_id = var.render_owner_id
}

# ============================================================
# LOCALS
# ============================================================

locals {
  github_repo_url = "https://github.com/${var.github_owner}/${var.github_repository}"

  fastapi_url = "https://${var.fastapi_service_name}.onrender.com"

  streamlit_url = "https://${var.streamlit_service_name}.onrender.com"

  effective_allowed_origins = (
    var.allowed_origins != ""
    ? var.allowed_origins
    : local.streamlit_url
  )

  effective_fastapi_url = (
    var.fastapi_url != ""
    ? var.fastapi_url
    : local.fastapi_url
  )
}

# ============================================================
# GITHUB REPOSITORY
# ============================================================

resource "github_repository" "smart_vet" {
  name        = var.github_repository
  description = "Smart VET - Sistema inteligente para análise veterinária"

  visibility = var.github_visibility

  has_issues   = true
  has_projects = true
  has_wiki     = false

  allow_squash_merge = true
  allow_merge_commit = true
  allow_rebase_merge = true

  delete_branch_on_merge = true

  auto_init = false
}

# ============================================================
# RENDER - FASTAPI
# ============================================================

resource "render_web_service" "fastapi" {
  name   = var.fastapi_service_name
  plan   = var.render_plan
  region = var.render_region

  root_directory = var.fastapi_root_directory

  runtime_source = {
    docker = {
      repo_url        = local.github_repo_url
      branch          = var.fastapi_branch
      auto_deploy     = true
      context         = "."
      dockerfile_path = "Dockerfile"
    }
  }

  env_vars = {

    # ----------------------------------------------------------
    # API
    # ----------------------------------------------------------

    API_ENV = {
      value = var.api_env
    }

    API_KEY = {
      value = var.API_KEY
    }

    SMART_VET_API_KEY = {
      value = var.SMART_VET_API_KEY
    }

    MAX_UPLOAD_BYTES = {
      value = var.max_upload_bytes
    }

    ALLOWED_ORIGINS = {
      value = local.effective_allowed_origins
    }

    # ----------------------------------------------------------
    # HUGGING FACE
    # ----------------------------------------------------------

    HF_DATA_REPO = {
      value = var.hf_data_repo
    }

    HF_DATA_FILENAME = {
      value = var.hf_data_filename
    }

    HF_IMAGE_REPO = {
      value = var.hf_image_repo
    }

    HF_IMAGE_FILENAME = {
      value = var.hf_image_filename
    }

    HF_MODEL_REPO = {
      value = var.hf_model_repo
    }

    HF_MODEL_FILENAME = {
      value = var.hf_model_filename
    }

    # ----------------------------------------------------------
    # LLM PRINCIPAL
    # ----------------------------------------------------------

    LLM_PROVIDER = {
      value = var.llm_provider
    }

    LLM_API_KEY = {
      value = var.llm_api_key
    }

    LLM_BASE_URL = {
      value = var.llm_base_url
    }

    LLM_MODEL = {
      value = var.llm_model
    }

    LLM_JSON_MODE = {
      value = var.llm_json_mode
    }

    # ----------------------------------------------------------
    # LLM FALLBACK 1
    # ----------------------------------------------------------

    LLM_FALLBACK_1_PROVIDER = {
      value = var.llm_fallback_1_provider
    }

    LLM_FALLBACK_1_API_KEY = {
      value = var.llm_fallback_1_api_key
    }

    LLM_FALLBACK_1_BASE_URL = {
      value = var.llm_fallback_1_base_url
    }

    LLM_FALLBACK_1_MODEL = {
      value = var.llm_fallback_1_model
    }

    LLM_FALLBACK_1_JSON_MODE = {
      value = var.llm_fallback_1_json_mode
    }

    # ----------------------------------------------------------
    # LLM CONFIGURATION
    # ----------------------------------------------------------

    LLM_TIMEOUT_SECONDS = {
      value = var.llm_timeout_seconds
    }

    LLM_TEMPERATURE = {
      value = var.llm_temperature
    }
  }
}

# ============================================================
# RENDER - STREAMLIT
# ============================================================

resource "render_web_service" "streamlit" {
  name   = var.streamlit_service_name
  plan   = var.render_plan
  region = var.render_region

  root_directory = var.streamlit_root_directory

  runtime_source = {
    docker = {
      repo_url        = local.github_repo_url
      branch          = var.streamlit_branch
      auto_deploy     = true
      context         = "."
      dockerfile_path = "Dockerfile"
    }
  }

  env_vars = {

    API_KEY = {
      value = var.API_KEY
    }

    SMART_VET_API_KEY = {
      value = var.SMART_VET_API_KEY
    }

    FASTAPI_URL = {
      value = local.effective_fastapi_url
    }
  }
}

# ============================================================
# OUTPUT - GITHUB
# ============================================================

output "github_repo_url" {
  description = "URL do repositorio GitHub"

  value = github_repository.smart_vet.html_url
}

output "github_clone_url" {
  description = "URL HTTPS para clonar o repositorio"

  value = github_repository.smart_vet.http_clone_url
}

# ============================================================
# OUTPUT - FASTAPI
# ============================================================

output "fastapi_service_id" {
  description = "ID do servico FastAPI no Render"

  value = render_web_service.fastapi.id
}

output "fastapi_url" {
  description = "URL publica do FastAPI"

  value = render_web_service.fastapi.url
}

# ============================================================
# OUTPUT - STREAMLIT
# ============================================================

output "streamlit_service_id" {
  description = "ID do servico Streamlit no Render"

  value = render_web_service.streamlit.id
}

output "streamlit_url" {
  description = "URL publica do Streamlit"

  value = render_web_service.streamlit.url
}
