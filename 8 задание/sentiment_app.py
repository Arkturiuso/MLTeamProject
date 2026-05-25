import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Анализ тональности", layout="wide")
st.title("Приложение анализа эмоциональной окраски комментариев")

API_URL = "http://127.0.0.1:8000/predict"

tab1, tab2, tab3 = st.tabs(["Анализ тональности", "Статистика датасета", "ℹСправка"])

with tab1:
    st.subheader("Введите комментарий для анализа")
    user_input = st.text_area("Текст комментария:", height=180)

    if st.button("Анализировать", type="primary"):
        if user_input.strip():
            with st.spinner("Анализируем текст..."):
                try:
                    response = requests.post(API_URL, json={"text": user_input})
                    if response.status_code == 200:
                        result = response.json()

                        if result["sentiment"] == "positive":
                            st.success("**Позитивный**")
                        else:
                            st.error("**Негативный**")

                        st.info(f"**Уверенность модели:** {result['confidence'] * 100:.1f}%")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Позитив", f"{result['probability_positive'] * 100:.1f}%")
                        with col2:
                            st.metric("Негатив", f"{result['probability_negative'] * 100:.1f}%")

                        st.write("**Обработанный текст:**", result.get("clean_text", ""))
                    else:
                        st.error(f"Ошибка API: {response.status_code}")
                except Exception as e:
                    st.error("Не удалось подключиться к API. Убедитесь, что api.py запущен.")
        else:
            st.warning("Введите текст для анализа!")

with tab2:
    st.subheader("Статистика датасета")
    if st.button("Загрузить статистику"):
        try:
            pos = pd.read_csv('pos.csv', sep=';', header=None, usecols=[3, 4], names=['text', 'sentiment'],
                              encoding='utf-8', engine='python')
            neg = pd.read_csv('neg.csv', sep=';', header=None, usecols=[3, 4], names=['text', 'sentiment'],
                              encoding='utf-8', engine='python')
            df = pd.concat([pos, neg], ignore_index=True)

            total = len(df)
            pos_count = (df['sentiment'] == 1).sum()
            neg_count = (df['sentiment'] == -1).sum()

            st.write(f"**Всего комментариев:** {total:,}")
            st.write(f"**Позитивных:** {pos_count:,} ({pos_count / total * 100:.1f}%)")
            st.write(f"**Негативных:** {neg_count:,} ({neg_count / total * 100:.1f}%)")

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots()
                sns.countplot(x=df['sentiment'], ax=ax)
                ax.set_title("Распределение тональности")
                st.pyplot(fig)

            with col2:
                df['length'] = df['text'].astype(str).str.len()
                fig, ax = plt.subplots()
                sns.histplot(df['length'], bins=50, ax=ax)
                ax.set_title("Длина комментариев")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка: {e}")

with tab3:
    st.subheader("Справка по командам")
    st.markdown("""
    **API Endpoints:**
    - `POST /predict` — основной метод анализа текста
    - `GET /` — проверка работы API

    **Как запускать:**
    1. Запусти `api.py` командой: `uvicorn api:app --reload`
    2. Запусти Streamlit: `streamlit run sentiment_app.py`
    """)

