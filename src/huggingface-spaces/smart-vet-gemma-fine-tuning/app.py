
# ============================================================
# SMART VET - GEMMA 4 E2B IT - COMPARADOR E API V6
# Hugging Face Spaces + Gradio + ZeroGPU
#
# MODELO BASE:
#   unsloth/gemma-4-E2B-it
#
# ADAPTER:
#   ramoncg/tech-challenge-gemma-llm-fine-tuning-adapter-v6
#
# AVALIADOR:
#   deepseek-ai/DeepSeek-V4-Flash-0731
#
# API DE PRODUCAO:
#   /gemma_finetune
#
# A API executa SOMENTE o Gemma V6.
# Nao executa o modelo base.
# Nao executa o DeepSeek.
# Nao utiliza RAG.
#
# O FastAPI sera responsavel pelo roteamento:
#   - Sem paciente: Gemma V6 por esta API.
#   - Com paciente: LLM em nuvem + RAG filtrado.
#
# Comparador e API compartilham a mesma fila de GPU.
# ============================================================


# ============================================================
# 1. IMPORTACOES
# IMPORTANTE: spaces antes do torch no ZeroGPU.
# ============================================================

import spaces

import os
import json
import logging

import torch
import gradio as gr

from huggingface_hub import (
    hf_hub_download,
    InferenceClient,
)

from peft import PeftModel

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
)


# ============================================================
# 2. CONFIGURACOES
# ============================================================

MODEL_ID = "unsloth/gemma-4-E2B-it"

ADAPTER_ID = (
    "ramoncg/tech-challenge-gemma-llm-fine-tuning-adapter-v6"
)

DEEPSEEK_MODEL_ID = (
    "deepseek-ai/DeepSeek-V4-Flash-0731"
)

DEEPSEEK_PROVIDER = "auto"

GITHUB_URL = (
    "https://github.com/RamonCintas/Smart_VET"
)

# URL pública do Space.
SPACE_URL = (
    "https://ramoncg-smart-vet-gemma-fine-tuning.hf.space"
)

# Autenticacao da interface e da API Gradio.
SMARTVET_USER = os.getenv(
    "SMARTVET_SPACE_USER"
)

SMARTVET_PASSWORD = os.getenv(
    "SMARTVET_SPACE_PASSWORD"
)

# Token Hugging Face:
# - Download do modelo e adapter.
# - Acesso aos Inference Providers do DeepSeek.
HF_TOKEN = os.getenv(
    "HF_TOKEN"
)

HF_DOWNLOAD_ARGS = (
    {"token": HF_TOKEN}
    if HF_TOKEN
    else {}
)


# ============================================================
# 3. LIMITES DE GERACAO
# ============================================================

MAX_NEW_TOKENS_ICD = 96

MAX_NEW_TOKENS_NER = 256

MAX_NEW_TOKENS_FREE_QA = 768

# Limite da pergunta enviada pelo FastAPI.
MAX_API_PROMPT_CHARS = 12000

# DeepSeek.
DEEPSEEK_MAX_TOKENS = 3000

DEEPSEEK_TIMEOUT = 180

MAX_PROMPT_CHARS = 6000

MAX_ANSWER_CHARS = 7000

# Tempo de reserva para a API de atendimento.
API_GPU_DURATION = 180


# ============================================================
# 4. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(
    "smart-vet-comparator"
)


# ============================================================
# 5. AUTENTICACAO
# ============================================================

if not SMARTVET_USER or not SMARTVET_PASSWORD:

    raise RuntimeError(
        "Configure SMARTVET_SPACE_USER e "
        "SMARTVET_SPACE_PASSWORD nos Secrets do Space."
    )


# ============================================================
# 6. CARREGAR TRAINING MANIFEST
# ============================================================

logger.info(
    "Carregando training_manifest.json do adapter V6."
)

try:

    manifest_path = hf_hub_download(
        repo_id=ADAPTER_ID,
        filename="training_manifest.json",
        repo_type="model",
        **HF_DOWNLOAD_ARGS,
    )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as manifest_file:

        training_manifest = json.load(
            manifest_file
        )

    data_audit = training_manifest.get(
        "data_audit",
        {},
    )

    train_audit = data_audit.get(
        "train",
        {},
    )

    ICD_LABELS = sorted(
        str(label)
        for label in train_audit.get(
            "icd_label_frequency",
            {},
        ).keys()
    )

    DISEASE_LABELS = sorted(
        str(label).strip().upper()
        for label in train_audit.get(
            "disease_entity_label_frequency",
            {},
        ).keys()
    )

    ANONYMISATION_LABELS = sorted(
        str(label).strip().upper()
        for label in train_audit.get(
            "annonymisation_entity_label_frequency",
            {},
        ).keys()
    )

    if not ICD_LABELS:
        raise RuntimeError(
            "Vocabulario ICD-11 nao encontrado."
        )

    if not DISEASE_LABELS:
        raise RuntimeError(
            "Rotulos Disease NER nao encontrados."
        )

    if not ANONYMISATION_LABELS:
        raise RuntimeError(
            "Rotulos Anonymisation NER nao encontrados."
        )

    logger.info(
        "Vocabularios carregados | "
        "ICD=%s | Disease=%s | Anonymisation=%s",
        len(ICD_LABELS),
        len(DISEASE_LABELS),
        len(ANONYMISATION_LABELS),
    )

except Exception:

    logger.exception(
        "Erro ao carregar training_manifest.json."
    )

    raise RuntimeError(
        "Nao foi possivel carregar o training_manifest.json. "
        "Verifique se o arquivo esta publicado no repositorio "
        "do adapter V6."
    )


ICD_LABELS_TEXT = "\n".join(
    f"- {label}"
    for label in ICD_LABELS
)




# ============================================================
# 8. CARREGAR PROCESSOR
# ============================================================

logger.info(
    "Carregando processor: %s",
    MODEL_ID,
)

try:

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        **HF_DOWNLOAD_ARGS,
    )

    logger.info(
        "Processor carregado com sucesso."
    )

except Exception:

    logger.exception(
        "Erro ao carregar processor."
    )

    raise


# ============================================================
# 9. CARREGAR MODELO BASE
# ============================================================

logger.info(
    "Carregando modelo base original: %s",
    MODEL_ID,
)

try:

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="cuda",
        low_cpu_mem_usage=True,
        **HF_DOWNLOAD_ARGS,
    )

    logger.info(
        "Modelo base carregado | classe=%s",
        base_model.__class__.__name__,
    )

except Exception:

    logger.exception(
        "Erro ao carregar modelo base original."
    )

    raise


# ============================================================
# 10. CARREGAR ADAPTER PEFT V6
# ============================================================

logger.info(
    "Carregando adapter PEFT V6."
)

try:

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_ID,
        is_trainable=False,
        torch_device="cpu",
        low_cpu_mem_usage=False,
        **HF_DOWNLOAD_ARGS,
    )

    model.eval()

    logger.info(
        "Adapter PEFT V6 carregado com sucesso."
    )

except Exception:

    logger.exception(
        "Erro ao carregar adapter PEFT V6."
    )

    raise


logger.info(
    "MODELO INICIALIZADO | Base=%s | Adapter=%s",
    MODEL_ID,
    ADAPTER_ID,
)


# ============================================================
# 11. CATEGORIAS
# ============================================================

TASK_LABELS = {

    "icd11_classification":
        "ICD-11 — classificacao multilabel",

    "disease_ner":
        "Disease NER — extracao de doencas",

    "annonymisation_ner":
        "Anonymisation NER — anonimização",

    "free_qa":
        "Pergunta livre — teste exploratorio",
}


TASK_CHOICES = [

    (
        "ICD-11 — classificacao multilabel",
        "icd11_classification",
    ),

    (
        "Disease NER — extracao de doencas",
        "disease_ner",
    ),

    (
        "Anonymisation NER — anonimização",
        "annonymisation_ner",
    ),

    (
        "Pergunta livre — avaliacao exploratoria",
        "free_qa",
    ),
]


# ============================================================
# 12. EXEMPLOS AUTOMATICOS
# ============================================================

TASK_EXAMPLES = {

    "icd11_classification": """
Cao, macho, da raca Golden Retriever, 8 anos, apresenta
vomitos ha tres dias, diarreia aquosa, perda de apetite
e apatia. O tutor relata que o animal ingeriu restos de
alimentos encontrados no lixo ha aproximadamente quatro
dias. Ao exame clinico, apresenta sensibilidade abdominal
e sinais de desidratacao. Suspeita clinica de gastroenterite
aguda de possivel origem alimentar.
""".strip(),

    "disease_ner": """
Felino, femea, 6 anos, apresenta historico de dermatite
atopica e episodios recorrentes de otite externa.
O animal foi atendido com prurido intenso, eritema e
lesoes cutaneas. O veterinario considera como diagnostico
diferencial a dermatofitose e suspeita de infeccao
secundaria por Staphylococcus pseudintermedius.
Nao ha evidencias clinicas de leishmaniose.
""".strip(),

    "annonymisation_ner": """
No dia 12/10/2026, a Dra. Ana Souza realizou atendimento
na Clinica VetVida, localizada em Sao Paulo.
O paciente foi encaminhado pelo Hospital Veterinario
Central para avaliacao especializada.
O relatorio foi registrado no protocolo VET-204 e
encaminhado ao responsavel tecnico, Dr. Carlos Mendes.
""".strip(),

    "free_qa": """
Cao, macho, 2 anos, apresenta diarreia liquida ha dois
dias, febre de 40 graus Celsius medida com termometro,
falta de apetite e episodios de vomito. O tutor informa
que a vacinacao esta atrasada.

Quais informacoes adicionais devem ser investigadas,
quais sao os principais riscos clinicos e qual deve
ser a prioridade do atendimento?
""".strip(),
}


# ============================================================
# 13. CARREGAR EXEMPLO DA CATEGORIA
# ============================================================

def load_task_example(task_id):

    return TASK_EXAMPLES.get(
        task_id,
        "",
    )


# ============================================================
# 14. PROMPT DE PERGUNTA LIVRE
#
# Esta e a rota utilizada pelo FastAPI quando nenhum
# paciente estiver selecionado.
#
# O texto enviado pelo FastAPI pode conter historico
# recente da conversa, preparado pelo proprio backend.
#
# Nao existe RAG nesta funcao.
# ============================================================

FREE_SYSTEM_PROMPT = """
Voce e o assistente veterinario geral do Smart VET.

Responda em portugues brasileiro, com clareza e organizacao.

Utilize somente as informacoes presentes na solicitacao
e conhecimentos veterinarios gerais.

Nao invente dados clinicos, exames, diagnosticos ou referencias.

Diferencie sinais clinicos, hipoteses diagnosticas e
diagnosticos confirmados.

Nao prescreva medicamentos ou doses sem avaliacao profissional.

Nao substitua a avaliacao de um medico-veterinario.

Em situacoes potencialmente graves, destaque a necessidade
de avaliacao veterinaria imediata.

Se faltarem informacoes, indique quais dados adicionais
seriam importantes.

Quando nao tiver seguranca sobre uma informacao, informe isso.
"""


# ============================================================
# 15. CONSTRUCAO DOS PROMPTS DAS TAREFAS
# ============================================================

def build_task_messages(
    task_id: str,
    sentence: str,
):

    if not sentence or not sentence.strip():

        raise ValueError(
            "Digite uma narrativa ou pergunta."
        )

    sentence = sentence.strip()

    # --------------------------------------------------------
    # ICD-11
    # --------------------------------------------------------

    if task_id == "icd11_classification":

        task_prompt = (

            "You are annotating a veterinary electronic health "
            "record for the PetEVAL ICD-11 syndromic multi-label task. "

            "This is benchmark annotation, NOT clinical decision-making. "

            "Assign every supported chapter label present in the record; "
            "a consultation may have more than one label. "

            "Use the exact allowed label names below and do not create synonyms. "

            "Ignore routine, non-syndromic check-ups and treatment instructions "
            "unless they document a relevant condition. "

            "If no ICD-11 chapter is supported by the narrative, return []. "

            "Do not infer diagnoses beyond the supplied text.\n\n"

            "Allowed ICD-11 labels "
            "(exact spelling from the training annotations):\n"

            f"{ICD_LABELS_TEXT}\n\n"

            "Veterinary EHR narrative (unaltered source text):\n"

            f"{sentence}\n\n"

            "Return ONLY a valid JSON array of strings."
        )

        return [
            {
                "role": "user",
                "content": task_prompt,
            }
        ], MAX_NEW_TOKENS_ICD

    # --------------------------------------------------------
    # DISEASE NER
    # --------------------------------------------------------

    if task_id == "disease_ner":

        disease_labels_text = ", ".join(
            DISEASE_LABELS
        )

        task_prompt = (

            "You are performing the PetEVAL veterinary "
            "disease-extraction NER task. "

            "Extract the disease/condition mentions that are "
            "explicitly stated in the record. "

            "The annotation policy includes named conditions described "
            "as diagnoses, suspected or differential diagnoses, syndromic "
            "condition descriptions, and pathogens, unless the mention "
            "is explicitly negated. "

            "Do not infer a disease from symptoms alone, medication, "
            "treatment, or general findings. "

            "Do not extract symptoms as diseases unless the source "
            "annotation treats the phrase as a disease/condition. "

            "Copy every entity exactly as it appears, preserving spelling, "
            "capitalization, punctuation, and whitespace. "

            f"Use only these dataset entity labels: {disease_labels_text}. "

            "Do NOT generate character offsets. "

            "Include each annotated occurrence in source order; "
            "return [] when none is annotated.\n\n"

            "Veterinary EHR narrative (unaltered source text):\n"

            f"{sentence}\n\n"

            "Return ONLY a JSON array of objects. "
            "Each object must contain exactly: "
            "entity (string), label (string)."
        )

        return [
            {
                "role": "user",
                "content": task_prompt,
            }
        ], MAX_NEW_TOKENS_NER

    # --------------------------------------------------------
    # ANONYMISATION NER
    # --------------------------------------------------------

    if task_id == "annonymisation_ner":

        anonymisation_labels_text = ", ".join(
            ANONYMISATION_LABELS
        )

        task_prompt = (

            "You are performing the PetEVAL anonymisation/"
            "de-identification NER task. "

            "Extract ONLY the privacy-related entities annotated "
            "in the record. "

            "Use the exact dataset categories and meanings: "

            "PER = pet, owner, or veterinary staff names; "

            "LOC = cities, towns, countries, and geographic locations; "

            "ORG = veterinary practices, clinics, referral hospitals, "
            "shelters, kennels, laboratories, and groomers; "

            "TIME = dates, specific times, periods, or durations "
            "when annotated; "

            "MISC = explicit unique identifiers or contact identifiers "
            "such as microchip, passport, insurance/policy, "
            "clinician-registration codes, phone numbers, "
            "or email addresses. "

            "Do not label ordinary medicines, treatments, procedures, "
            "medical equipment, or clinical conditions as MISC. "

            "Do not label animal names as MISC; use PER. "

            "Do not infer identifiers that are not explicitly "
            "present/annotated. "

            "Copy each entity exactly as it appears, preserving spelling, "
            "capitalization, punctuation, and whitespace. "

            f"Allowed labels observed in PetEVAL training annotations: "
            f"{anonymisation_labels_text}. "

            "Do NOT generate character offsets. "

            "Include each annotated occurrence in source order; "
            "return [] when none is annotated.\n\n"

            "Veterinary EHR narrative (unaltered source text):\n"

            f"{sentence}\n\n"

            "Return ONLY a JSON array of objects. "
            "Each object must contain exactly: "
            "entity (string), label (string)."
        )

        return [
            {
                "role": "user",
                "content": task_prompt,
            }
        ], MAX_NEW_TOKENS_NER

    # --------------------------------------------------------
    # PERGUNTA LIVRE
    # --------------------------------------------------------

    if task_id == "free_qa":

        return [
            {
                "role": "system",
                "content": FREE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": sentence,
            },
        ], MAX_NEW_TOKENS_FREE_QA

    raise ValueError(
        f"Categoria nao reconhecida: {task_id}"
    )


# ============================================================
# 16. PREPARACAO DOS TENSORES
# ============================================================

def prepare_inputs(messages):

    formatted_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = processor(
        text=formatted_prompt,
        return_tensors="pt",
    ).to("cuda")

    input_length = int(
        inputs["input_ids"].shape[-1]
    )

    return inputs, input_length


def clone_inputs(inputs):

    return {
        key: (
            value.clone()
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in inputs.items()
    }


# ============================================================
# 17. GERACAO DE RESPOSTA GEMMA
# ============================================================

def generate_answer(
    inputs,
    input_length,
    max_new_tokens,
):

    generation_kwargs = {

        "max_new_tokens": int(max_new_tokens),

        "do_sample": False,

        "use_cache": True,

        "num_beams": 1,
    }

    tokenizer = getattr(
        processor,
        "tokenizer",
        None,
    )

    if tokenizer is not None:

        if tokenizer.pad_token_id is not None:

            generation_kwargs["pad_token_id"] = (
                tokenizer.pad_token_id
            )

        if tokenizer.eos_token_id is not None:

            generation_kwargs["eos_token_id"] = (
                tokenizer.eos_token_id
            )

    outputs = model.generate(
        **clone_inputs(inputs),
        **generation_kwargs,
    )

    generated_tokens = outputs[0][input_length:]

    answer = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# 18. TEMPO DINAMICO DO COMPARADOR
# ============================================================

def estimate_gpu_duration(
    task_id: str,
    prompt: str,
):

    durations = {

        "icd11_classification": 75,

        "disease_ner": 90,

        "annonymisation_ner": 90,

        "free_qa": 180,
    }

    return durations.get(
        task_id,
        180,
    )


# ============================================================
# 19. API DE PRODUCAO - GEMMA V6
#
# IMPORTANTE:
# - Executa somente o Gemma V6.
# - Nao desativa o adapter.
# - Nao gera resposta com modelo base.
# - Nao chama DeepSeek.
# - Nao utiliza RAG.
#
# O FastAPI envia uma pergunta simples ou uma mensagem
# composta com historico recente da conversa.
#
# Endpoint Gradio:
#   /gemma_finetune
#
# Chamada pelo FastAPI:
#   client.predict(prompt, api_name="/gemma_finetune")
# ============================================================

@spaces.GPU(duration=API_GPU_DURATION)
def gemma_finetune_for_api(prompt: str):

    if not isinstance(prompt, str):

        raise ValueError(
            "A entrada deve ser uma mensagem textual."
        )

    prompt = prompt.strip()

    if not prompt:

        raise ValueError(
            "A mensagem enviada pelo FastAPI esta vazia."
        )

    if len(prompt) > MAX_API_PROMPT_CHARS:

        raise ValueError(
            "A mensagem excedeu o limite de caracteres "
            f"permitido ({MAX_API_PROMPT_CHARS})."
        )

    try:

        logger.info(
            "API GEMMA V6 | Inicio da geracao | caracteres=%s",
            len(prompt),
        )

        # Sempre utiliza a modalidade free_qa.
        # Nao existe RAG nesta funcao.

        messages, max_new_tokens = build_task_messages(
            "free_qa",
            prompt,
        )

        inputs, input_length = prepare_inputs(
            messages
        )

        model.eval()

        # O adapter permanece ativo.
        with torch.inference_mode():

            answer = generate_answer(
                inputs,
                input_length,
                max_new_tokens,
            )

        if not answer:

            raise RuntimeError(
                "O Gemma V6 retornou uma resposta vazia."
            )

        logger.info(
            "API GEMMA V6 | Resposta concluida | "
            "entrada_tokens=%s | saida_max=%s",
            input_length,
            max_new_tokens,
        )

        return answer

    except Exception:

        logger.exception(
            "Erro na API de geracao Gemma V6."
        )

        raise


# ============================================================
# 20. COMPARADOR GEMMA BASE X V6
#
# O comparador continua sendo utilizado na interface de
# avaliacao. Ele nao sera chamado pelo FastAPI.
# ============================================================

@spaces.GPU(duration=estimate_gpu_duration)
def compare_models(
    task_id: str,
    prompt: str,
):

    if not prompt or not prompt.strip():

        raise ValueError(
            "Digite uma narrativa ou pergunta."
        )

    try:

        messages, max_new_tokens = build_task_messages(
            task_id,
            prompt,
        )

        inputs, input_length = prepare_inputs(
            messages
        )

        model.eval()

        logger.info(
            "Comparacao iniciada | tarefa=%s | "
            "entrada=%s tokens | saida=%s tokens",
            task_id,
            input_length,
            max_new_tokens,
        )

        # ----------------------------------------------------
        # GEMMA BASE SEM ADAPTER
        # ----------------------------------------------------

        logger.info(
            "Gerando resposta com Gemma Base."
        )

        with torch.inference_mode():

            with model.disable_adapter():

                base_answer = generate_answer(
                    inputs,
                    input_length,
                    max_new_tokens,
                )

        logger.info(
            "Resposta Gemma Base concluida."
        )

        # ----------------------------------------------------
        # GEMMA V6 COM ADAPTER ATIVO
        # ----------------------------------------------------

        logger.info(
            "Gerando resposta com adapter V6."
        )

        with torch.inference_mode():

            finetuned_answer = generate_answer(
                inputs,
                input_length,
                max_new_tokens,
            )

        logger.info(
            "Resposta Gemma V6 concluida."
        )

        logger.info(
            "Comparacao finalizada com sucesso."
        )

        return (
            base_answer,
            finetuned_answer,
        )

    except Exception:

        logger.exception(
            "Erro durante a comparacao."
        )

        raise


# ============================================================
# 21. LIMITAR TAMANHO DOS TEXTOS DO RELATORIO
# ============================================================

def truncate_report_text(
    value: str,
    max_chars: int,
):

    value = str(value or "").strip()

    if len(value) <= max_chars:

        return value

    return (
        value[:max_chars]
        + "\n\n[Texto reduzido para avaliacao.]"
    )


# ============================================================
# 22. CONSTRUIR PROMPT DO DEEPSEEK
# ============================================================

def build_deepseek_report_prompt(
    task_id,
    original_prompt,
    base_answer,
    finetuned_answer,
):

    task_name = TASK_LABELS.get(
        task_id,
        task_id,
    )

    if task_id == "icd11_classification":

        task_context = (
            "Classificacao multilabel ICD-11. "
            "Verifique se os rotulos retornados pertencem "
            "ao vocabulario autorizado, se representam "
            "o texto e se ha classificacoes omitidas ou indevidas."
        )

        vocabulary = ICD_LABELS

    elif task_id == "disease_ner":

        task_context = (
            "Disease NER. Verifique doencas, condicoes, patogenos, "
            "negacoes, diagnosticos diferenciais, omissoes "
            "e extracoes indevidas."
        )

        vocabulary = DISEASE_LABELS

    elif task_id == "annonymisation_ner":

        task_context = (
            "Anonymisation NER. Verifique entidades pessoais, "
            "limites exatos dos trechos, categorias e informacoes "
            "omitidas ou identificadas indevidamente."
        )

        vocabulary = ANONYMISATION_LABELS

    else:

        task_context = (
            "Pergunta livre. Avalie clareza, completude, "
            "consistencia, seguranca e encaminhamento profissional. "
            "Esta modalidade e exploratoria e nao foi uma das "
            "tarefas supervisionadas diretamente no treinamento V6."
        )

        vocabulary = []

    original_prompt = truncate_report_text(
        original_prompt,
        MAX_PROMPT_CHARS,
    )

    base_answer = truncate_report_text(
        base_answer,
        MAX_ANSWER_CHARS,
    )

    finetuned_answer = truncate_report_text(
        finetuned_answer,
        MAX_ANSWER_CHARS,
    )

    vocabulary_text = (
        json.dumps(
            vocabulary,
            ensure_ascii=False,
            indent=2,
        )
        if vocabulary
        else "Nao aplicavel."
    )

    return f"""
Voce e um avaliador tecnico independente do projeto SMART VET.

Compare duas respostas geradas a partir da MESMA entrada:

1. Gemma 4 Base, sem adapter PEFT.
2. Gemma 4 com adapter PEFT V6 ativado.

Escreva o relatorio em portugues brasileiro.

REGRAS OBRIGATORIAS:

1. Seja imparcial. Nao favoreca automaticamente o V6.
2. Nao considere uma resposta melhor apenas por ser mais longa.
3. Diferencie diferenca textual de melhoria de qualidade.
4. Nao invente informacoes, entidades, diagnosticos ou referencias.
5. Nao invente resultados numericos.
6. Sem ground truth, nao afirme que calculou precisao, recall ou F1.
7. Use o vocabulario permitido quando a tarefa exigir.
8. Verifique negacoes, omissoes e entidades extraidas indevidamente.
9. Nao substitua a validacao de um profissional veterinario.
10. Pergunta livre e uma avaliacao exploratoria.

REGRAS ADICIONAIS:

- O vocabulario fornecido e uma lista fechada.
- Nunca recomende como resultado valido um rotulo que nao esteja nela.
- Diferencie rotulo permitido, rotulo sustentado pelo texto e rotulo
  apenas clinicamente plausivel.
- A ausencia de multiplos rotulos nao e automaticamente um erro.
- Nao transforme suspeitas em diagnosticos confirmados.
- Nao considere a ingestao de alimentos contaminados, isoladamente,
  evidencia suficiente de infeccao parasitaria.
- Se identificar uma contradicao na analise, revise antes de finalizar.
- Quando nao houver ground truth, nao declare superioridade clinica definitiva.

CATEGORIA:
{task_name}

CONTEXTO:
{task_context}

VOCABULARIO AUTORIZADO:
{vocabulary_text}

ENTRADA ORIGINAL:
{original_prompt}

RESPOSTA DO GEMMA BASE:
{base_answer}

RESPOSTA DO GEMMA V6:
{finetuned_answer}

ESTRUTURA DO RELATORIO:

# Relatorio comparativo Smart VET

## 1. Resumo executivo

## 2. Avaliacao do Gemma Base

## 3. Avaliacao do Gemma V6

## 4. Comparacao lado a lado

## 5. Erros, omissoes e inconsistencias

## 6. Resultado desta amostra

Escolha uma classificacao:
- V6 aparentemente melhor nesta amostra.
- Gemma Base aparentemente melhor nesta amostra.
- Resultados equivalentes.
- Evidencia insuficiente.

Justifique a classificacao.

## 7. Proximos testes recomendados

## 8. Conclusao

Nao atribua notas de 0 a 100 sem metodologia e ground truth.

Finalize com:

"Relatorio auxiliar gerado por IA. A avaliacao final
deve ser realizada por um profissional responsavel."
"""


# ============================================================
# 23. RELATORIO AUTOMATICO COM DEEPSEEK
# ============================================================


def generate_deepseek_report(
    task_id,
    original_prompt,
    base_answer,
    finetuned_answer,
):

    if not HF_TOKEN:

        return (
            "### DeepSeek nao configurado\n\n"
            "O Secret HF_TOKEN nao foi encontrado."
        )

    if not base_answer or not finetuned_answer:

        return (
            "### Relatorio nao gerado\n\n"
            "Nao foram encontradas as duas respostas."
        )

    try:

        client = InferenceClient(
            model=DEEPSEEK_MODEL_ID,
            provider=DEEPSEEK_PROVIDER,
            token=HF_TOKEN,
            timeout=DEEPSEEK_TIMEOUT,
        )

        report_prompt = build_deepseek_report_prompt(
            task_id,
            original_prompt,
            base_answer,
            finetuned_answer,
        )

        system_prompt = """
Voce e o avaliador independente do Smart VET.

Sua funcao e produzir um relatorio comparativo final,
em portugues brasileiro.

Seja imparcial, tecnico, objetivo e criterioso.

Nao invente fatos, metricas ou referencias.

Nao exponha raciocinio interno.
Apresente somente a avaliacao final solicitada.

Entregue o relatorio completo em Markdown.
"""

        logger.info(
            "Solicitando relatorio ao DeepSeek | "
            "modelo=%s | tarefa=%s | reasoning=disabled",
            DEEPSEEK_MODEL_ID,
            task_id,
        )

        response = client.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": report_prompt,
                },
            ],
            max_tokens=DEEPSEEK_MAX_TOKENS,
            temperature=0.2,
            extra_body={
                "reasoning_effort": "none",
            },
        )

        if not response.choices:

            raise RuntimeError(
                "O provedor nao retornou nenhuma escolha."
            )

        choice = response.choices[0]
        message = choice.message

        # ----------------------------------------------------
        # EXTRAIR SOMENTE O CONTEUDO FINAL
        # ----------------------------------------------------

        report = getattr(
            message,
            "content",
            None,
        )

        # Compatibilidade com respostas serializadas.
        if not report and hasattr(message, "model_dump"):

            message_data = message.model_dump()

            report = message_data.get(
                "content"
            )

        # ----------------------------------------------------
        # DIAGNOSTICO SE A RESPOSTA CONTINUAR VAZIA
        #
        # Nao registramos nem exibimos o raciocinio interno.
        # ----------------------------------------------------

        if not report:

            if hasattr(message, "model_dump"):

                message_data = message.model_dump()

            elif hasattr(message, "__dict__"):

                message_data = message.__dict__

            else:

                message_data = {}

            logger.error(
                "DeepSeek sem conteudo final | "
                "finish_reason=%s | message_fields=%s | "
                "usage=%s",
                getattr(
                    choice,
                    "finish_reason",
                    None,
                ),
                list(message_data.keys()),
                getattr(
                    response,
                    "usage",
                    None,
                ),
            )

            raise RuntimeError(
                "DeepSeek retornou resposta sem conteudo final. "
                "Consulte finish_reason e os campos registrados "
                "nos logs."
            )

        report = str(report).strip()

        if not report:

            raise RuntimeError(
                "O conteudo final do DeepSeek esta vazio."
            )

        logger.info(
            "Relatorio DeepSeek gerado com sucesso | "
            "caracteres=%s | finish_reason=%s",
            len(report),
            getattr(
                choice,
                "finish_reason",
                None,
            ),
        )

        return report

    except Exception as exc:

        logger.exception(
            "Erro ao gerar relatorio com DeepSeek."
        )

        return (
            "### Nao foi possivel gerar o relatorio DeepSeek\n\n"
            f"**Erro:** `{type(exc).__name__}`\n\n"
            "As respostas do Gemma Base e do Gemma V6 "
            "foram preservadas.\n\n"
            "Verifique os detalhes do erro nos logs."
        )


# ============================================================
# 24. INTERFACE GRADIO
# ============================================================

with gr.Blocks(
    title="Smart VET - Comparador Gemma V6",
) as demo:

    gr.Markdown(
        """
# Smart VET - Gemma Fine-Tuning

Compare o Gemma 4 E2B IT em duas condicoes:

- **Gemma Base:** modelo original, sem adapter PEFT.
- **Gemma V6:** modelo original com adapter PEFT V6.

Depois da comparacao, o DeepSeek-V4-Flash-0731
gera automaticamente um relatorio tecnico.

O mesmo Space tambem disponibiliza uma API de producao
para o FastAPI utilizar somente o Gemma V6.
"""
    )

    gr.Markdown(
        f"""
**Repositorio do projeto:**

[Smart VET — GitHub]({GITHUB_URL})
"""
    )

    

    

    gr.Markdown(
        f"""
**Modelo base:** `{MODEL_ID}`

**Adapter:** `{ADAPTER_ID}`

**Avaliador:** `{DEEPSEEK_MODEL_ID}`

**API de producao:** `/gemma_finetune`
"""
    )

    # --------------------------------------------------------
    # CATEGORIA
    # --------------------------------------------------------

    task_selector = gr.Dropdown(
        choices=TASK_CHOICES,
        value="icd11_classification",
        label="Selecione a categoria de avaliacao",
        interactive=True,
    )

    prompt_input = gr.Textbox(
        value=TASK_EXAMPLES[
            "icd11_classification"
        ],
        label="Mensagem pronta / narrativa veterinaria",
        placeholder="Digite ou cole o texto...",
        lines=9,
        max_lines=18,
    )

    task_selector.change(
        fn=load_task_example,
        inputs=[
            task_selector,
        ],
        outputs=[
            prompt_input,
        ],
    )

    # --------------------------------------------------------
    # COMPARADOR
    # --------------------------------------------------------

    generate_button = gr.Button(
        "Comparar Gemma Base x Gemma V6",
        variant="primary",
        size="lg",
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown(
                """
### Gemma 4 — Modelo Base

Adapter PEFT desativado.
"""
            )

            base_response_output = gr.Textbox(
                label="Resposta do Gemma Base",
                lines=22,
                max_lines=32,
                interactive=False,
                buttons=["copy"],
            )

        with gr.Column():

            gr.Markdown(
                """
### Gemma 4 — Fine-Tuning V6

Adapter PEFT V6 ativado.
"""
            )

            finetuned_response_output = gr.Textbox(
                label="Resposta do Gemma V6",
                lines=22,
                max_lines=32,
                interactive=False,
                buttons=["copy"],
            )

    gr.Markdown(
        """
---
## Relatorio comparativo — DeepSeek V4 Flash
"""
    )

    deepseek_report_output = gr.Markdown(
        "Aguardando a comparacao."
    )

    # --------------------------------------------------------
    # COMPARACAO
    #
    # Compartilha a fila GEMMA_GPU com a API de producao.
    # --------------------------------------------------------

    compare_event = generate_button.click(
        fn=compare_models,
        inputs=[
            task_selector,
            prompt_input,
        ],
        outputs=[
            base_response_output,
            finetuned_response_output,
        ],
        api_name="compare",
        concurrency_limit=1,
        concurrency_id="gemma_gpu",
    )

    # --------------------------------------------------------
    # RELATORIO DEEPSEEK AUTOMATICO
    # --------------------------------------------------------

    compare_event.success(
        fn=generate_deepseek_report,
        inputs=[
            task_selector,
            prompt_input,
            base_response_output,
            finetuned_response_output,
        ],
        outputs=[
            deepseek_report_output,
        ],
        concurrency_limit=1,
    )

    # ========================================================
    # 25. ENDPOINT API DE PRODUCAO PARA FASTAPI
    #
    # COMPONENTES OCULTOS:
    # O FastAPI envia o texto para api_prompt_input e recebe
    # a resposta em api_response_output.
    #
    # O endpoint e chamado por:
    # client.predict(prompt, api_name="/gemma_finetune")
    #
    # A autenticacao e feita com:
    # auth=(SMARTVET_USER, SMARTVET_PASSWORD)
    # ========================================================

    api_prompt_input = gr.Textbox(
        label="Entrada da API FastAPI",
        visible=False,
    )

    api_response_output = gr.Textbox(
        label="Resposta da API Gemma V6",
        visible=False,
    )

    api_generate_button = gr.Button(
        "API Gemma V6",
        visible=False,
    )

    api_generate_button.click(
        fn=gemma_finetune_for_api,
        inputs=[
            api_prompt_input,
        ],
        outputs=[
            api_response_output,
        ],
        api_name="gemma_finetune",
        api_visibility="undocumented",
        concurrency_limit=1,
        concurrency_id="gemma_gpu",
    )


# ============================================================
# 26. FILA E INICIALIZACAO
# ============================================================

demo.queue(
    max_size=4,
    default_concurrency_limit=1,
)

demo.launch(
    auth=(
        SMARTVET_USER,
        SMARTVET_PASSWORD,
    ),
    show_error=True,
)
