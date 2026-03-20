import streamlit as st
import tempfile
import numpy as np
import librosa
import time

from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Whisper Pro", layout="wide")

st.title("🎙️ Whisper Speech Platform")

# =========================
# НАСТРОЙКИ
# =========================
models = ["tiny", "base", "small", "medium"]

languages = {
    "Авто": None,
    "Русский": "ru",
    "Английский": "en",
    "Немецкий": "de",
    "Французский": "fr",
    "Испанский": "es"
}

translate_map = {
    "Русский": "ru",
    "Английский": "en",
    "Немецкий": "de"
}

model_size = st.selectbox("Модель", models)
lang_display = st.selectbox("Язык", list(languages.keys()))
lang_code = languages[lang_display]

translate_to = st.selectbox(
    "Перевести в",
    ["Не переводить", "Русский", "Английский", "Немецкий"]
)

uploaded_file = st.file_uploader("Загрузите аудио", type=["wav", "mp3"])

# =========================
# КЭШ
# =========================
@st.cache_resource
def load_model(size):
    return WhisperModel(size, compute_type="int8")

# =========================
# АУДИО
# =========================
def load_audio(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        path = tmp.name

    audio, sr = librosa.load(path, sr=16000)
    duration = len(audio) / sr

    return audio, duration

# =========================
# ПСЕВДО СПИКЕРЫ
# =========================
def assign_speakers(segments):
    speakers = []
    current_speaker = 1

    prev_end = 0

    for seg in segments:
        # если пауза > 1.5 сек → новый спикер
        if seg.start - prev_end > 1.5:
            current_speaker += 1

        speakers.append({
            "speaker": f"Speaker {current_speaker}",
            "start": seg.start,
            "end": seg.end,
            "text": seg.text
        })

        prev_end = seg.end

    return speakers

# =========================
# ПЕРЕВОД
# =========================
def translate_text(text):
    if translate_to == "Не переводить":
        return text

    target_lang = translate_map[translate_to]

    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# =========================
# ОСНОВНОЙ ПРОЦЕСС
# =========================
if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🚀 Распознать"):

        start_time = time.time()

        audio, duration = load_audio(uploaded_file)

        with st.spinner("Загрузка модели..."):
            model = load_model(model_size)

        with st.spinner("Распознавание..."):
            segments, info = model.transcribe(
                audio,
                language=lang_code
            )

            segments = list(segments)

        end_time = time.time()

        # =========================
        # ТЕКСТ
        # =========================
        full_text = " ".join([seg.text for seg in segments])

        # =========================
        # СПИКЕРЫ (псевдо)
        # =========================
        speaker_segments = assign_speakers(segments)

        # =========================
        # ПЕРЕВОД
        # =========================
        translated_text = translate_text(full_text)

        # =========================
        # СТАТИСТИКА
        # =========================
        num_segments = len(segments)
        avg_segment_length = duration / num_segments if num_segments else 0
        processing_time = end_time - start_time

        st.success("Готово!")

        # =========================
        # СТАТИСТИКА
        # =========================
        st.subheader("📊 Статистика")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("⏱ Длительность аудио (сек)", round(duration, 2))
        col2.metric("🧩 Сегментов", num_segments)
        col3.metric("📏 Ср. длина сегмента", round(avg_segment_length, 2))
        col4.metric("⚡ Время обработки", round(processing_time, 2))

        st.write(f"🌍 Определён язык: {info.language}")

        # =========================
        # ТЕКСТ
        # =========================
        st.subheader("📄 Текст")
        st.write(full_text)

        # =========================
        # ПЕРЕВОД
        # =========================
        if translate_to != "Не переводить":
            st.subheader("🌍 Перевод")
            st.write(translated_text)

        # =========================
        # СПИКЕРЫ
        # =========================
        st.subheader("🗣 Спикеры (эвристика)")

        for seg in speaker_segments:
            st.write(
                f"[{seg['start']:.2f}-{seg['end']:.2f}] "
                f"{seg['speaker']}: {seg['text']}"
            )

        # =========================
        # СКАЧИВАНИЕ
        # =========================
        st.download_button(
            "⬇️ Скачать текст",
            full_text,
            file_name="transcription.txt"
        )

        if translate_to != "Не переводить":
            st.download_button(
                "⬇️ Скачать перевод",
                translated_text,
                file_name="translation.txt"
            )
