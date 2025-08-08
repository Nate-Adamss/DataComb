import streamlit as st
import pandas as pd
import openai
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

st.set_page_config(page_title="DataComb", page_icon="👾", layout="wide")
st.title("👾 DataComb")
st.write("Upload your dataset and ask questions in plain English.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Here's a preview of your data:")
    st.dataframe(df.head())

    if st.secrets.get("OPENAI_API_KEY"):
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        llm = OpenAI(api_token=openai.api_key)
        sdf = SmartDataframe(df, config={"llm": llm})

        history = st.session_state.get("chat_history", [])
        user_input = st.text_input("Ask a question about your data:")

        if user_input:
            with st.spinner("Thinking..."):
                try:
                    response = sdf.chat(user_input)
                    history.append(("You", user_input))
                    history.append(("DataComb", response))
                    st.session_state["chat_history"] = history
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        for speaker, message in history:
            if speaker == "You":
                st.markdown(f"**🧑 You:** {message}")
            else:
                st.markdown(f"**👾 DataComb:** {message}")

    else:
        st.warning("No API key provided. Add OPENAI_API_KEY in Streamlit Cloud > Settings > Secrets.")
