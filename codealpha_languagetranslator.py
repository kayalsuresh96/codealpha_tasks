import streamlit as st
from googletrans import Translator, LANGUAGES

# Create translator object
translator = Translator()

st.title("🌍 Language Translation Tool")

# Input text
text = st.text_area("Enter text to translate:")

# Language selection
languages = list(LANGUAGES.values())

source_lang = st.selectbox("Select Source Language:", languages)
target_lang = st.selectbox("Select Target Language:", languages)

# Translate button
if st.button("Translate"):
    if text:
        # Get language codes
        source_code = list(LANGUAGES.keys())[list(LANGUAGES.values()).index(source_lang)]
        target_code = list(LANGUAGES.keys())[list(LANGUAGES.values()).index(target_lang)]

        translated = translator.translate(text, src=source_code, dest=target_code)

        st.success("Translated Text:")
        st.write(translated.text)
    else:
        st.warning("Please enter text to translate.")
