import requests
import streamlit as st

from PIL import Image, UnidentifiedImageError

from tabular import render_interpretation


# =========================================================
# HELPERS
# =========================================================

def render_prediction_item(item, index=None):
    if not isinstance(item, dict):
        st.write(item)
        return

    label = item.get("label", item.get("class_name", "Classe não identificada"))
    confidence = item.get("confidence", item.get("score"))
    reference_image = item.get("reference_image")

    col1, col2 = st.columns([1, 3])
    with col1:
        if reference_image:
            st.image(reference_image, width=120)

    with col2:
        st.write(f"**{label}**")
        if confidence is not None:
            try:
                confidence = float(confidence)
                confidence = min(max(confidence, 0.0), 1.0)
                st.progress(confidence)
                st.caption(f"Confiança: {confidence * 100:.2f}%")
            except (TypeError, ValueError):
                st.caption(f"Confiança: {confidence}")


def render_prediction_list(title, predictions):
    st.subheader(title)
    if not predictions:
        st.info("Nenhuma previsão retornada.")
        return

    for index, item in enumerate(predictions):
        render_prediction_item(item, index)


# =========================================================
# VISION MODEL
# =========================================================

def render_vision(FASTAPI_URL, api_headers):
    st.subheader("🔬 Modelo de Visão Computacional")
    st.write(
        "Envie uma imagem do animal para análise "
        "e apoio à triagem dermatológica."
    )

    uploaded_file = st.file_uploader(
        "Envie uma imagem do animal",
        type=["png", "jpg", "jpeg", "webp"],
        key="vision_upload",
    )

    # =====================================================
    # IMAGE PREVIEW
    # =====================================================
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            image.verify()

            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Imagem enviada para análise",
                width="stretch",
            )
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            st.error("O arquivo enviado não é uma imagem válida.")
            return

    # =====================================================
    # IMAGE PREDICTION
    # =====================================================
    if st.button("Realizar Diagnóstico Dermatológico", type="primary"):
        if not uploaded_file:
            st.warning("Envie uma imagem primeiro.")
            return

        response = None
        try:
            image_bytes = uploaded_file.getvalue()
            if not image_bytes:
                st.warning("O arquivo enviado está vazio.")
                return

            files = {
                "file": (
                    uploaded_file.name,
                    image_bytes,
                    uploaded_file.type or "application/octet-stream",
                )
            }

            with st.spinner(
                "Analisando imagem com o modelo de visão computacional..."
            ):
                response = requests.post(
                    f"{FASTAPI_URL}/predict/image",
                    files=files,
                    headers=api_headers(),
                    timeout=180,
                )

            if response.status_code == 401:
                st.error("Autenticação ausente ou inválida.")
                return
            if response.status_code == 403:
                st.error("A API key foi rejeitada.")
                return
            if response.status_code == 413:
                st.error("A imagem ultrapassa o tamanho máximo permitido pela API.")
                return

            response.raise_for_status()
            results = response.json()

            if not isinstance(results, dict):
                st.error("A API retornou uma resposta inválida.")
                return
            if results.get("success") is False:
                st.warning(
                    results.get("message", "Não foi possível processar a imagem.")
                )
                return

            st.success(results.get("message", "Análise da imagem concluída."))

            predictions = results.get("predictions")
            if isinstance(predictions, list):
                render_prediction_list("🔬 Resultados da análise", predictions)

            dog_breeds = results.get("dog_breeds", [])
            if dog_breeds:
                render_prediction_list("🐶 Possíveis raças", dog_breeds)

            skin_diseases = results.get("skin_diseases", [])
            if skin_diseases:
                render_prediction_list(
                    "🧬 Possíveis condições dermatológicas",
                    skin_diseases,
                )

            render_interpretation(results.get("interpretation"))

            with st.expander("Resposta completa da API"):
                st.json(results)

        except requests.exceptions.Timeout:
            st.error("O processamento da imagem excedeu o tempo limite.")
        except requests.exceptions.ConnectionError:
            st.error("Não foi possível conectar à API de visão computacional.")
        except requests.exceptions.HTTPError as error:
            st.error(f"Erro HTTP durante a análise: {error}")
            if response is not None:
                st.code(response.text)
        except requests.exceptions.RequestException as error:
            st.error(f"Erro na requisição de imagem: {error}")
        except (ValueError, KeyError) as error:
            st.error(f"Erro ao interpretar o resultado: {error}")
