import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import librosa
import numpy as np

st.title("🎙️ Speech-to-Text")

model_size = st.selectbox(
    "Модель",
    ["tiny", "base", "small", "medium"]
)

language = st.selectbox(
    "Язык",
    ["auto", "ru", "en", "de", "fr", "es"]
)

uploaded_file = st.file_uploader("Аудио", type=["wav", "mp3"])

@st.cache_resource
def load_model(size):
    return WhisperModel(size, compute_type="int8")

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Распознать"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Загружаем через librosa (БЕЗ ffmpeg)
        audio, sr = librosa.load(tmp_path, sr=16000)
        audio = np.array(audio, dtype=np.float32)

        model = load_model(model_size)

        segments, info = model.transcribe(
            audio,
            language=None if language == "auto" else language
        )

        text = " ".join([seg.text for seg in segments])

        st.success("Готово")
        st.write(text)
