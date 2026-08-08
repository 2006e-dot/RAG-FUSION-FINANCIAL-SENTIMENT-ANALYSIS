import os
import pandas as pd
import streamlit as st

from retriever import rag_fusion_retrieve
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


MODEL_NAME="finance-bot-3b:latest"
MAX_RETRIES=3


st.set_page_config(
    page_title="Financial Sentiment RAG",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #888;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
        border: 1px solid #444;
    }

    .positive {
        font-size: 32px;
        font-weight: 700;
    }

    .negative {
        font-size: 32px;
        font-weight: 700;
    }

    .neutral {
        font-size: 32px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">Financial Sentiment RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">RAG Fusion + ChromaDB + Finance-Bot 3B</div>',
    unsafe_allow_html=True
)


@st.cache_resource
def load_llm():

    return ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )


llm=load_llm()


prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert financial sentiment classifier.

Classify the headline into EXACTLY ONE label using the provided context.

Labels:

- positive
- neutral
- negative

Rules:

- positive: clearly indicates favourable financial performance, growth, gains, upgrades, rallies, or positive developments.
- negative: clearly indicates losses, declines, downgrades, lawsuits, weak earnings, risks, or negative developments.
- neutral: reports factual information (appointments, filings, approvals, listings, mergers, launches, board changes, routine announcements) without expressing positive or negative market sentiment.

Use the context only to resolve ambiguity in the headline — do not invent sentiment that isn't supported.

Respond with only the label, lowercase, no punctuation, no explanation."""
        ),
        (
            "human",
            """Context:
{context}

Headline:
{headline}

Answer:"""
        )
    ]
)

chain=prompt|llm


def parse_prediction(text):

    text=text.lower().strip()

    if "positive" in text:
        return "positive"

    if "negative" in text:
        return "negative"

    if "neutral" in text:
        return "neutral"

    return "parse_error"


def predict(headline,context):

    for _ in range(MAX_RETRIES):

        try:

            response=chain.invoke(
                {
                    "headline":headline,
                    "context":context
                }
            )

            label=parse_prediction(response.content)

            if label!="parse_error":
                return label

        except Exception:
            pass

    return "parse_error"


def get_prediction_color(label):

    if label=="positive":
        return "positive"

    if label=="negative":
        return "negative"

    if label=="neutral":
        return "neutral"

    return "Error"


if "history" not in st.session_state:
    st.session_state.history=[]


tab1,tab2=st.tabs(
    [
        "Single Prediction",
        "Batch Prediction"
    ]
)


with tab1:

    st.subheader("Analyze Financial Headline")

    headline=st.text_area(
        "Enter a financial headline",
        placeholder="Example: Apple reports record quarterly revenue and profit",
        height=120
    )

    col1,col2=st.columns([1,5])

    with col1:

        predict_button=st.button(
            "Analyze",
            type="primary",
            use_container_width=True
        )

    if predict_button:

        if not headline.strip():

            st.warning("Please enter a headline.")

        else:

            with st.spinner("Retrieving relevant financial context..."):

                try:

                    docs=rag_fusion_retrieve(headline)

                except Exception as e:

                    st.error(
                        f"RAG retrieval failed: {str(e)}"
                    )

                    docs=[]

            if docs:

                context="\n\n".join(docs)

                with st.spinner("Finance-Bot is classifying the headline..."):

                    prediction=predict(
                        headline,
                        context
                    )

                st.markdown("### Prediction")

                icon=get_prediction_color(prediction)

                st.markdown(
                    f"""
                    <div class="prediction-box">
                        <div class="{prediction}">
                            {icon} {prediction.upper()}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### Retrieved Context")

                with st.expander(
                    f"View {len(docs)} retrieved documents",
                    expanded=False
                ):

                    for i,doc in enumerate(docs,1):

                        st.markdown(
                            f"**Document {i}**"
                        )

                        st.write(doc)

                        if i<len(docs):
                            st.divider()


                st.session_state.history.insert(
                    0,
                    {
                        "headline":headline,
                        "prediction":prediction,
                        "documents":len(docs)
                    }
                )

            else:

                st.warning(
                    "No relevant documents were retrieved."
                )


with tab2:

    st.subheader("Batch Financial Sentiment Prediction")

    uploaded_file=st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file:

        df=pd.read_csv(uploaded_file)

        st.write(
            f"Loaded **{len(df)} rows**"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        title_column="title"

        if title_column not in df.columns:

            st.error(
                "CSV must contain a `title` column."
            )

        else:

            if st.button(
                "Run Batch Prediction",
                type="primary"
            ):

                predictions=[]

                progress=st.progress(0)

                status=st.empty()

                for i,row in df.iterrows():

                    headline=str(row["title"])

                    status.text(
                        f"Processing {i+1}/{len(df)}"
                    )

                    try:

                        docs=rag_fusion_retrieve(
                            headline
                        )

                        context="\n\n".join(docs)

                        prediction=predict(
                            headline,
                            context
                        )

                    except Exception:

                        prediction="parse_error"

                    predictions.append(prediction)

                    progress.progress(
                        (i+1)/len(df)
                    )

                df["predicted_label"]=predictions

                status.text("Prediction completed.")

                st.success(
                    "Batch prediction completed successfully."
                )

                st.dataframe(
                    df,
                    use_container_width=True
                )

                csv=df.to_csv(index=False).encode(
                    "utf-8"
                )

                st.download_button(
                    "Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )


st.divider()


if st.session_state.history:

    st.subheader("Recent Predictions")

    history_df=pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )