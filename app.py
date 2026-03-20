import streamlit as st
import tempfile
import numpy as np
import librosa

# Engines
from faster_whisper import WhisperModel
from vosk import Model as VoskModel, KaldiRecognizer
import wave
import json

from transformers import pipeline
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Multi Speech-to-Text", layout="wide")

st.title("🎙️ Multi Speech Recognition Platform")

# =========================
# ЯЗЫКИ (человеческие)
# =========================
languages = {
    "Авто": None,
    "Русский": "ru",
    "Английский": "en",
    "Немецкий": "de",
    "Французский": "fr",
    "Испанский": "es"
}

# =========================
# ВЫБОР ДВИЖКА
# =========================
engine = st.selectbox(
    "Выберите движок",
    ["faster-whisper", "vosk", "wav2vec2"]
)

# =========================
# ВЫБОР МОДЕЛИ
# =========================
if engine == "faster-whisper":
    model_name = st.selectbox("Модель", ["tiny", "base", "small", "medium"])

elif engine == "vosk":
    st.info("Язык будет определён автоматически")
    model_name = None

elif engine == "wav2vec2":
    st.info("Язык будет определён автоматически")
    model_name = None

# =========================
# ЯЗЫК (только для whisper)
# =========================
lang_display = st.selectbox("Язык (для Whisper)", list(languages.keys()))
lang_code = languages[lang_display]

# =========================
# ПЕРЕВОД
# =========================
translate_to = st.selectbox(
    "Перевести в",
    ["Не переводить", "Русский", "Английский", "Немецкий"]
)

translate_map = {
    "Русский": "ru",
    "Английский": "en",
    "Немецкий": "de"
}

# =========================
# МАППИНГ МОДЕЛЕЙ
# =========================
vosk_models = {
    "ru": "vosk-model-small-ru",
    "en": "vosk-model-small-en-us"
}

wav2vec_models = {
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "en": "facebook/wav2vec2-base-960h"
}

# =========================
# ЗАГРУЗКА
# =========================
uploaded_file = st.file_uploader("Загрузите аудио", type=["wav", "mp3"])

# =========================
# КЭШИРОВАНИЕ МОДЕЛЕЙ
# =========================
@st.cache_resource
def load_whisper(model_name):
    return WhisperModel(model_name, compute_type="int8")

@st.cache_resource
def load_lang_detector():
    return WhisperModel("tiny", compute_type="int8")

@st.cache_resource
def load_wav2vec(model_name):
    return pipeline("automatic-speech-recognition", model=model_name)

# =========================
# ЗАГРУЗКА АУДИО
# =========================
def load_audio(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        path = tmp.name

    audio, sr = librosa.load(path, sr=16000)
    return audio, path

# =========================
# ОПРЕДЕЛЕНИЕ ЯЗЫКА
# =========================
def detect_language(audio):
    model = load_lang_detector()
    segments, info = model.transcribe(audio)
    return info.language

# =========================
# WHISPER
# =========================
def transcribe_whisper(audio):
    model = load_whisper(model_name)

    segments, info = model.transcribe(
        audio,
        language=lang_code
    )

    result_text = ""
    segments_data = []

    for seg in segments:
        text = seg.text
        result_text += text + " "
        segments_data.append({
            "start": seg.start,
            "end": seg.end,
            "text": text
        })

    return result_text, segments_data

# =========================
# VOSK
# =========================
def transcribe_vosk(path, model_name):
    wf = wave.open(path, "rb")
    model = VoskModel(model_name)

    rec = KaldiRecognizer(model, wf.getframerate())

    result_text = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            result_text += res.get("text", "") + " "

    final = json.loads(rec.FinalResult())
    result_text += final.get("text", "")

    return result_text, None

# =========================
# WAV2VEC2
# =========================
def transcribe_wav2vec(audio, model_name):
    pipe = load_wav2vec(model_name)
    result = pipe(audio)
    return result["text"], None

# =========================
# ПЕРЕВОД
# =========================
def translate_text(text):
    if translate_to == "Не переводить":
        return text

    target_lang = translate_map[translate_to]

    translated = GoogleTranslator(
        source="auto",
        target=target_lang
    ).translate(text)

    return translated

# =========================
# ОСНОВНОЙ ПРОЦЕСС
# =========================
if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🚀 Распознать"):
        audio, path = load_audio(uploaded_file)

        with st.spinner("Определение языка..."):
            detected_lang = detect_language(audio)
            st.info(f"Определён язык: {detected_lang}")

        with st.spinner("Распознавание..."):

            if engine == "faster-whisper":
                text, segments = transcribe_whisper(audio)

            elif engine == "vosk":
                model_name = vosk_models.get(detected_lang, "vosk-model-small-en-us")
                text, segments = transcribe_vosk(path, model_name)

            elif engine == "wav2vec2":
                model_name = wav2vec_models.get(detected_lang, "facebook/wav2vec2-base-960h")
                text, segments = transcribe_wav2vec(audio, model_name)

        translated_text = translate_text(text)

        st.success("Готово!")

        # =========================
        # ВЫВОД
        # =========================
        st.subheader("📄 Текст")
        st.write(text)

        if translate_to != "Не переводить":
            st.subheader("🌍 Перевод")
            st.write(translated_text)

        # =========================
        # СЕГМЕНТЫ (только whisper)
        # =========================
        if segments:
            st.subheader("🧩 Сегменты")
            for seg in segments:
                st.write(
                    f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}"
                )

        # =========================
        # СКАЧИВАНИЕ
        # =========================
        st.download_button(
            "⬇️ Скачать текст",
            text,
            file_name="transcription.txt"
        )

        if translate_to != "Не переводить":
            st.download_button(
                "⬇️ Скачать перевод",
                translated_text,
                file_name="translation.txt"
            )
