import os
import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

FASTAPI_URL = os.getenv(
    "FASTAPI_URL",
    "http://localhost:8000"
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Smart VET",
    page_icon="🐾",
    layout="wide"
)

st.title("Smart VET")
st.caption("Sistema de triagem veterinária")


# =========================================================
# API HEALTH CHECK
# =========================================================

api_ok = False

animals = []
symptoms = []

try:

    response = requests.get(
        f"{FASTAPI_URL}/health",
        timeout=15
    )

    response.raise_for_status()

    health = response.json()

    api_ok = True

    animals = health["tabular"]["animals"]

    symptoms = health["tabular"]["symptoms"]

except Exception as e:

    st.error(f"Erro ao conectar na API: {e}")

    st.stop()

with st.expander("🟢 Status da API", expanded=False):

    st.success("API conectada")

    st.write(f"**Ambiente:** {health['environment']}")

    st.write("### Modelos")

    st.write(
        f"📊 Tabular: `{health['models']['tabular_model']}`"
    )

    st.write(
        f"🗂 Dataset: `{health['models']['dataset']}`"
    )

    st.write(
        f"🖼 Imagem: `{health['models']['image_model']}`"
    )

    st.divider()

    st.write("### Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Animais",
        health["tabular"]["animal_count"]
    )

    col2.metric(
        "Sintomas",
        health["tabular"]["symptom_count"]
    )

    col3.metric(
        "Registros",
        health["tabular"]["dataset_rows"]
    )



# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs([
    "Diagnóstico clínico",
    "Diagnóstico por imagem"
])


# =========================================================
# TAB 1 - TABULAR MODEL
# =========================================================

with tab1:
    st.subheader("Modelo tabular")

    animal_name = st.selectbox(
        "Animal",
        animals,
        index=animals.index("dog") if "dog" in animals else 0
    )

    col1, col2,col3, col4, col5 = st.columns(5)

    with col1:
        symptoms1 = st.selectbox(
            "Sintoma 1",
            symptoms,
            index=symptoms.index("fever") if "fever" in symptoms else 0
        )
    with col2:
        symptoms2 = st.selectbox(
            "Sintoma 2",
            symptoms,
            index=symptoms.index("vomiting") if "vomiting" in symptoms else 0
        )
    with col3:
        symptoms3 = st.selectbox(
            "Sintoma 3",
            symptoms,
            index=symptoms.index("diarrhea") if "diarrhea" in symptoms else 0
        )

    with col4:
        symptoms4 = st.selectbox(
            "Sintoma 4",
            symptoms,
            index=symptoms.index("fatigue") if "fatigue" in symptoms else 0
        )
    with col5:
        symptoms5 = st.selectbox(
            "Sintoma 5",
            symptoms,
            index=symptoms.index("dehydration") if "dehydration" in symptoms else 0
        )
    if st.button("Realizar Diagnóstico Clínico"):
        payload = {
            "animal_name": animal_name,
            "symptoms1": symptoms1,
            "symptoms2": symptoms2,
            "symptoms3": symptoms3,
            "symptoms4": symptoms4,
            "symptoms5": symptoms5
        }

        try:
            response = requests.post(
                f"{FASTAPI_URL}/predict/condition",
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            st.success("Diagnóstico concluído")

            prediction = result["prediction_binary"]
            probability = result["probability_yes"]

            if prediction == 1:
                st.error("Animal em possível estado crítico")
            else:
                st.success("Animal sem sinais críticos")

            st.metric(
                "Probabilidade de risco",
                f"{probability * 100:.2f}%" if probability else "N/A"
            )

            with st.expander("Resposta completa"):
                st.json(result)

        except Exception as e:
            st.error(f"Erro: {e}")


# =========================================================
# TAB 2 - IMAGE MODEL (UI PROFISSIONAL)
# =========================================================

with tab2:
    st.subheader("🧠 Diagnóstico por imagem veterinário")

    uploaded_file = st.file_uploader(
        "Envie uma imagem do animal",
        type=["png", "jpg", "jpeg", "webp"]
    )

    image = None

    if uploaded_file:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="📷 Imagem enviada (paciente)",
            use_container_width=True
        )

    if st.button("🚀 Executar diagnóstico por imagem"):

        if not uploaded_file:
            st.warning("Envie uma imagem primeiro.")
        else:
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    f"{FASTAPI_URL}/predict/image",
                    files=files,
                    timeout=120
                )

                response.raise_for_status()
                results = response.json()

                if not results.get("success", True):
                    st.warning(results.get("message", "Imagem inválida."))
                    st.stop()

                st.success(results.get("message", "✅ Diagnóstico concluído"))

                dog_breeds = results.get("dog_breeds", [])
                skin_diseases = results.get("skin_diseases", [])

                # =========================================================
                # 🧠 RESUMO INTELIGENTE 
                # =========================================================
                best_skin = skin_diseases[0] if skin_diseases else None

                if best_skin:
                    label = best_skin["label"]
                    conf = best_skin["confidence"]

                    if label == "healthy":
                        st.success(f"🟢 Pele: {label}")
                    else:
                        st.warning(f"⚠️ Pele: {label}")

                    st.metric("Confiança", f"{conf * 100:.2f}%")

                st.divider()

                # =========================================================
                # 🐶 RAÇAS (FOCO VISUAL)
                # =========================================================
                st.subheader("🐶 Possíveis raças")

                if not dog_breeds:
                    st.info("Nenhuma raça detectada.")
                else:
                    cols = st.columns(5)

                    for i, dog in enumerate(dog_breeds[:5]):
                        with cols[i]:

                            st.image(
                                dog.get("reference_image"),
                                use_container_width=True
                            )

                            st.markdown(
                                f"**{dog['label']}**"
                            )
                            st.caption(
                                f"{dog['confidence']*100:.2f}%"
                            )

                st.divider()

                # =========================================================
                # 🧬 DOENÇAS DE PELE (FOCO CLÍNICO)
                # =========================================================
                st.subheader("🧬 Possíveis condições dermatológicas")

                if not skin_diseases:
                    st.info("Nenhuma condição detectada.")
                else:
                    for skin in skin_diseases:

                        col1, col2 = st.columns([1, 3])

                        with col1:
                            st.image(
                                skin.get("reference_image"),
                                width=120
                            )

                        with col2:
                            st.write(f"**{skin['label']}**")
                            st.progress(float(skin["confidence"]))
                            st.caption(f"{skin['confidence']*100:.2f}%")

            except Exception as e:
                st.error(f"Erro: {e}")


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(f"FastAPI URL: {FASTAPI_URL}")
