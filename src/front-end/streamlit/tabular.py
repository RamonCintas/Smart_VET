import requests
import streamlit as st


# =========================================================
# INTERPRETATION
# =========================================================

def render_interpretation(interpretation):
    if not isinstance(interpretation, dict):
        return

    st.divider()
    st.subheader("Interpretação com LLM")

    generated_by = interpretation.get("generated_by", "rules")
    provider = interpretation.get("provider", "rules")
    model_used = interpretation.get("model_used", "N/A")

    if generated_by == "llm":
        st.success(f"Gerado por LLM: `{provider}` / `{model_used}`")
    else:
        st.info("Interpretação local baseada em regras.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Resumo:** {interpretation.get('summary', 'N/A')}")
    with col2:
        st.metric("Nível de risco", interpretation.get("risk_level", "N/A"))

    explanation = interpretation.get("explanation")
    if explanation:
        st.write(explanation)

    insights = interpretation.get("actionable_insights", [])
    limitations = interpretation.get("limitations", [])

    if insights:
        st.write("**Insights acionáveis**")
        for item in insights:
            st.write(f"- {item}")

    if limitations:
        with st.expander("Limitações da interpretação"):
            for item in limitations:
                st.write(f"- {item}")


# =========================================================
# HELPERS
# =========================================================

def format_probability(value):
    if value is None:
        return "N/A"

    try:
        value = float(value)
        if 0 <= value <= 1:
            value *= 100
        return f"{value:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


# =========================================================
# TABULAR MODEL
# =========================================================

def render_tabular(FASTAPI_URL, api_headers, animals, symptoms):
    st.subheader("🩺 Modelo Tabular")

    if not animals or not symptoms:
        st.warning("Não existem opções disponíveis para o diagnóstico.")
        return

    animal_name = st.selectbox(
        "Animal",
        animals,
        index=animals.index("dog") if "dog" in animals else 0,
    )

    st.write("### Selecione os sintomas")

    default_symptoms = ["fever", "vomiting", "diarrhea", "fatigue", "dehydration"]
    selected_symptoms = []
    cols = st.columns(5)

    for index in range(5):
        preferred = default_symptoms[index]
        default_index = symptoms.index(preferred) if preferred in symptoms else 0

        with cols[index]:
            symptom = st.selectbox(
                f"Sintoma {index + 1}",
                symptoms,
                index=default_index,
                key=f"tabular_symptom_{index}",
            )
            selected_symptoms.append(symptom)

    if st.button("Realizar Diagnóstico Clínico", type="primary"):
        if len(set(selected_symptoms)) != len(selected_symptoms):
            st.warning("Selecione cinco sintomas diferentes.")
            return

        payload = {
            "animal_name": animal_name,
            "symptoms1": selected_symptoms[0],
            "symptoms2": selected_symptoms[1],
            "symptoms3": selected_symptoms[2],
            "symptoms4": selected_symptoms[3],
            "symptoms5": selected_symptoms[4],
        }

        response = None
        try:
            with st.spinner("Executando modelo de triagem clínica..."):
                response = requests.post(
                    f"{FASTAPI_URL}/predict/condition",
                    json=payload,
                    headers=api_headers(),
                    timeout=120,
                )

            if response.status_code == 401:
                st.error("Autenticação ausente ou inválida.")
                return
            if response.status_code == 403:
                st.error("A API key foi rejeitada.")
                return

            response.raise_for_status()
            result = response.json()

            if not isinstance(result, dict):
                st.error("Resposta inválida da API.")
                return

            st.success("Diagnóstico concluído.")

            prediction = result.get("prediction_binary")
            probability = result.get("probability_yes")
            prediction_label = result.get("prediction_label")

            if prediction == 1:
                st.error("⚠️ Animal em possível estado crítico.")
            elif prediction == 0:
                st.success("🟢 Animal sem indicação de estado crítico.")
            elif prediction_label:
                st.info(f"Resultado: {prediction_label}")
            else:
                st.warning("A API não retornou uma classificação reconhecida.")

            st.metric("Probabilidade de risco", format_probability(probability))
            render_interpretation(result.get("interpretation"))

            with st.expander("Resposta completa da API"):
                st.json(result)

        except requests.exceptions.Timeout:
            st.error("A API demorou demais para executar o diagnóstico.")
        except requests.exceptions.ConnectionError:
            st.error("Não foi possível conectar à FastAPI.")
        except requests.exceptions.HTTPError as error:
            st.error(f"Erro HTTP durante o diagnóstico: {error}")
            if response is not None:
                st.code(response.text)
        except (ValueError, KeyError) as error:
            st.error(f"Erro ao interpretar a resposta: {error}")
        except requests.exceptions.RequestException as error:
            st.error(f"Erro na requisição: {error}")
