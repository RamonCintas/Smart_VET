"""Aplicação principal Streamlit do Smart VET.

Este módulo configura a página e organiza a navegação. Cada aba delega sua
interface ao módulo responsável; comunicação com backend ocorre por HTTP.
"""
from __future__ import annotations

import streamlit as st

# Deve ser a primeira chamada Streamlit que configura a página.
st.set_page_config(
    page_title="Smart VET",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from agente import FASTAPI_URL, api_headers, render_agente  # noqa: E402
from painel import load_tabular_options, render_painel  # noqa: E402
from tabular import render_tabular  # noqa: E402
from vision import render_vision  # noqa: E402


# Um único conjunto de headers é compartilhado pelas abas que recebem headers.
# api_headers() adiciona X-SmartVet-Patient-ID quando há paciente selecionado
# no agente; tabular e visão usam esse contexto para anexar resultados ao prontuário.
headers = api_headers()
current_patient_id = st.session_state.get("smartvet_current_patient_id")
if current_patient_id:
    st.info("Contexto clínico compartilhado ativo. Os resultados de tabular/visão serão associados ao paciente selecionado quando os endpoints forem executados.")
else:
    st.caption("Para anexar exames ao prontuário, selecione primeiro um paciente na aba Agente Veterinário. Sem paciente selecionado, os exames não serão associados a um prontuário.")

# Navegação principal: cada módulo conserva sua própria responsabilidade.
tab_painel, tab_clinica, tab_dermatologica, tab_agente = st.tabs(
    [
        "📊 Painel e auditoria",
        "🩺 Triagem clínica",
        "🔬 Triagem dermatológica",
        "🌐 Agente veterinário",
    ]
)

with tab_painel:
    render_painel(api_url=FASTAPI_URL, headers=headers)

with tab_clinica:
    animals, symptoms, _, tabular_error = load_tabular_options(
        api_url=FASTAPI_URL,
        headers=headers,
    )
    if tabular_error:
        st.error(tabular_error)
        st.info("Consulte a aba Painel e auditoria para verificar a disponibilidade da API e do dataset.")
    elif not animals or not symptoms:
        st.warning("Não existem animais ou sintomas disponíveis para a triagem clínica.")
    else:
        # O módulo Streamlit de triagem recebe a função api_headers, conforme seu contrato.
        render_tabular(
            FASTAPI_URL=FASTAPI_URL,
            api_headers=api_headers,
            animals=animals,
            symptoms=symptoms,
        )

with tab_dermatologica:
    # Mantém o contrato original do módulo de visão computacional.
    render_vision(FASTAPI_URL, api_headers)

with tab_agente:
    render_agente(
        FASTAPI_URL=FASTAPI_URL,
        api_headers=headers,
    )
