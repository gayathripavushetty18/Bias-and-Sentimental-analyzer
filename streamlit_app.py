import os
import streamlit as st
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

from PIL import Image
import pytesseract
import altair as alt
import pandas as pd

import biased_modell  # your module with zero-shot, sentiment funcs

# GitHub Token and Azure client setup
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    st.error("Please set your GITHUB_TOKEN.")
    st.stop()

endpoint = "https://models.github.ai/inference"
model_name = "openai/gpt-4.1"
client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(GITHUB_TOKEN))

# Streamlit UI
st.set_page_config(page_title="News Article Analyzer", layout="wide")
st.title("📰 News Article Summarizer, Sentiment & Bias Detector")

article_text = st.text_area("Paste your news article here:", height=300)

uploaded_file = st.file_uploader("Or upload an image of a news article:", type=["png", "jpg", "jpeg"])

extracted_text = None
if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(image)
        st.subheader("📄 Extracted Text from Image")
        st.write(extracted_text)
    except Exception as e:
        st.error(f"Failed to extract text from image: {e}")

input_text = article_text if article_text.strip() else (extracted_text or "")

if st.button("Analyze"):
    if not input_text.strip():
        st.warning("Please enter or upload some article text/image.")
    else:
        with st.spinner("Generating summary..."):
            try:
                user_prompt = f"Please summarize the following news article:\n\n{input_text}"
                response = client.complete(
                    messages=[
                        SystemMessage("You are a helpful assistant that summarizes news articles."),
                        UserMessage(user_prompt)
                    ],
                    temperature=0.7,
                    top_p=1,
                    model=model_name
                )
                summary = response.choices[0].message.content.strip()
                st.subheader("📝 Summary")
                st.success(summary)
            except Exception as e:
                st.error(f"Summary failed: {e}")
                st.stop()

        with st.spinner("Analyzing sentiment..."):
            sentiment = biased_modell.get_final_sentiment(input_text)
            st.subheader("❤ Sentiment")
            st.info(f"Detected sentiment: *{sentiment}*")

        with st.spinner("Bias Detection..."):
            zero_shot = biased_modell.zero_shot_bias(input_text)
            st.subheader("⚖ Bias Detection")

            if zero_shot and "Bias_scores" in zero_shot and zero_shot["Bias_scores"]:
                scores = zero_shot["Bias_scores"]
                df_scores = pd.DataFrame(list(scores.items()), columns=["Bias", "Score"])

                chart = alt.Chart(df_scores).mark_bar().encode(
                    x=alt.X("Bias", sort=None, axis=alt.Axis(labelAngle=360)),
                    y=alt.Y("Score", axis=alt.Axis(title="Confidence Score")),
                    color=alt.Color("Bias", legend=None)
                ).properties(width=500, height=350)

                st.altair_chart(chart, use_container_width=False)

                st.subheader("📊 Bias Scores")

                # Display bias scores as lines, no table
                for index, row in df_scores.iterrows():
                    st.write(f"- *{row['Bias']}*: {row['Score']:.3f}")

                st.markdown(f"*Predicted Bias:* {zero_shot.get('Bias_label', 'Unknown')}")
            else:
                st.write("No bias scores available.")
                st.write("No bias scores available.")
