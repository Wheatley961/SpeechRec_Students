import streamlit as st
import whisper
import tempfile
import os

# Настройка страницы
st.set_page_config(
    page_title="Whisper Transcriber",
    page_icon="🎙️",
    layout="wide"
)

# Заголовок
st.title("AI Транскрибация Аудио (Whisper)")
st.markdown("""
Загрузите аудиофайл (MP3 или WAV), выберите модель и язык. 
Модель будет работать локально в вашем браузере/сервере.
""")

# --- Сайдбар с настройками ---
with st.sidebar:
    st.header("Настройки")
    
    # Выбор модели
    model_option = st.selectbox(
        "Выберите модель Whisper",
        ["tiny", "base", "small", "medium", "large"],
        index=2, # По умолчанию small
        help="Чем больше модель, тем точнее результат, но медленнее обработка и больше потребление памяти."
    )
    
    # Выбор языка
    language_option = st.selectbox(
        "Язык аудио",
        ["Automatic Detection", "Russian", "English", "German", "French", "Spanish", "Italian", "Portuguese", "Chinese", "Japanese"],
        index=0
    )
    
    # Перевод интерфейса выбора языка в код для Whisper
    lang_map = {
        "Automatic Detection": None,
        "Russian": "ru",
        "English": "en",
        "German": "de",
        "French": "fr",
        "Spanish": "es",
        "Italian": "it",
        "Portuguese": "pt",
        "Chinese": "zh",
        "Japanese": "ja"
    }
    selected_lang = lang_map[language_option]

# --- Функция загрузки модели с кэшированием ---
@st.cache_resource
def load_whisper_model(model_name):
    """
    Загружает модель один раз и сохраняет в кэш Streamlit.
    """
    return whisper.load_model(model_name)

# --- Основная логика ---
uploaded_file = st.file_uploader("Загрузите аудиофайл", type=["mp3", "wav", "m4a", "flac"])

if uploaded_file is not None:
    # Отображение аудио плеера
    st.audio(uploaded_file)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Файл:** {uploaded_file.name} | **Размер:** {round(uploaded_file.size / 1024 / 1024, 2)} MB")
    
    with col2:
        transcribe_btn = st.button("Начать транскрибацию", type="primary")

    if transcribe_btn:
        try:
            with st.spinner("Загрузка модели и обработка аудио... Это может занять время в зависимости от размера файла и выбранной модели."):
                
                # 1. Загрузка модели
                model = load_whisper_model(model_option)
                
                # 2. Сохранение загруженного файла во временную директорию
                # Whisper лучше работает с путями к файлам, чем с байтовыми потоками
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 3. Настройка параметров транскрибации
                options = {
                    "language": selected_lang,
                    "task": "transcribe" # или "translate" если нужно перевести в английский
                }
                
                # 4. Запуск транскрибации
                result = model.transcribe(tmp_file_path, **options)
                text_output = result["text"]
                
                # 5. Очистка временного файла
                os.unlink(tmp_file_path)
                
                # 6. Вывод результата
                st.success("Транскрибация завершена!")
                st.subheader("Результат:")
                st.text_area("Текст", value=text_output, height=300)
                
                # 7. Кнопка скачивания
                st.download_button(
                    label="Скачать текст (.txt)",
                    data=text_output,
                    file_name=f"transcript_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Произошла ошибка: {e}")
            st.info("Убедитесь, что у вас установлен FFmpeg в системе.")

else:
    st.info("Пожалуйста, загрузите аудиофайл выше, чтобы начать.")

# Футер
st.markdown("---")
st.caption("Powered by OpenAI Whisper & Streamlit")
