import streamlit as st
import whisper
import tempfile
import os

st.set_page_config(page_title="Whisper Transcriber", layout="centered")

st.title("Whisper Audio Transcription")

st.write("Загрузите аудиофайл (mp3 или wav), выберите модель и язык.")

# Выбор модели
model_size = st.selectbox(
    "Выберите модель Whisper",
    ["tiny", "base", "small", "medium", "large"]
)

# Выбор языка
language = st.selectbox(
    "Выберите язык аудио",
    ["auto", "en", "ru", "de", "fr", "es", "it", "zh"]
)

uploaded_file = st.file_uploader(
    "Загрузите аудио файл",
    type=["mp3", "wav"]
)

@st.cache_resource
def load_model(size):
    return whisper.load_model(size)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🚀 Расшифровать"):
        with st.spinner("Обработка аудио..."):

            # Сохраняем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                model = load_model(model_size)

                options = {}
                if language != "auto":
                    options["language"] = language

                result = model.transcribe(tmp_path, **options)

                st.success("✅ Готово!")

                st.subheader("📄 Текст:")
                st.write(result["text"])

            except Exception as e:
                st.error(f"Ошибка: {str(e)}")

            finally:
                os.remove(tmp_path)
