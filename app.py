import streamlit as st
import whisper
import tempfile
import librosa
import numpy as np
import soundfile as sf

# Заголовок
st.title("🎙️ Speech-to-Text (Whisper)")
st.write("Загрузите аудио (WAV или MP3), выберите язык и модель")

# Выбор модели
model_size = st.selectbox(
    "Выберите модель",
    ["tiny", "base", "small", "medium", "large"]
)

# Выбор языка
language = st.selectbox(
    "Выберите язык",
    ["auto", "ru", "en", "de", "fr", "es"]
)

# Загрузка файла
uploaded_file = st.file_uploader(
    "Загрузите аудио",
    type=["wav", "mp3"]
)

def load_audio(file):
    """Загрузка аудио без ffmpeg"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    # librosa читает и wav и mp3
    audio, sr = librosa.load(tmp_path, sr=16000)
    return audio

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🔍 Расшифровать"):
        with st.spinner("Загрузка модели..."):
            model = whisper.load_model(model_size)

        with st.spinner("Обработка аудио..."):
            audio = load_audio(uploaded_file)

            # Whisper ожидает numpy float32
            audio = np.array(audio, dtype=np.float32)

            options = {}
            if language != "auto":
                options["language"] = language

            result = model.transcribe(audio, **options)

        st.success("Готово!")

        st.subheader("📄 Текст:")
        st.write(result["text"])
