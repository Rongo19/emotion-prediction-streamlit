import streamlit as st
import pickle

st.set_page_config(page_title="Emotion Predictor", page_icon="😊")

# Load model and vectorizer
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

emotion_labels = {
    0: "Sadness 😢",
    1: "Anger 😠",
    2: "Fear 😨",
    3: "Love ❤️",
    4: "Surprise 😲",
    5: "Joy 😄"
}

st.title("💬 Emotion Prediction from Text")
st.write("Enter a sentence to predict the emotion")

text = st.text_area("Enter text")

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        emotion = emotion_labels.get(pred, "Unknown")
        st.success(f"Predicted Emotion: **{emotion}**")
