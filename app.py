import streamlit as st
import whisper
import tempfile

st.set_page_config(page_title="Whisper Transcriber")

st.title("🎙️ Speech-to-Text")

# Модель
model_size = st.selectbox(
    "Модель",
    ["tiny", "base", "small", "medium"]
)

# Язык
language = st.selectbox(
    "Язык",
    ["auto", "ru", "en", "de", "fr", "es"]
)

uploaded_file = st.file_uploader(
    "Загрузите аудио",
    type=["wav", "mp3"]
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Распознать"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Загрузка модели..."):
            model = whisper.load_model(model_size)

        with st.spinner("Распознавание..."):
            options = {}
            if language != "auto":
                options["language"] = language

            result = model.transcribe(tmp_path, **options)

        st.success("Готово!")
        st.write(result["text"])
