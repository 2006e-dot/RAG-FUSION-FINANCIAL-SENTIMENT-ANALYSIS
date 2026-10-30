import tempfile
from functools import lru_cache
from urllib.parse import quote

import gradio as gr
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from retriever import rag_fusion_retrieve


MODEL_NAME = "finance-bot-3b:latest"
MAX_RETRIES = 3


# -----------------------------
# ICONS (inline SVG, no image files needed)
# -----------------------------

ICON_PATHS = {
    "logo": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "flat": '<path d="M5 12h14"/>',
    "warn": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "ok": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "error": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
}


def svg(name, size=20):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f'{ICON_PATHS[name]}</svg>'
    )


def icon_uri(name):
    # Used for button icons through a CSS mask
    raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'{ICON_PATHS[name]}</svg>'
    )
    return "data:image/svg+xml;utf8," + quote(raw)


# -----------------------------
# LLM CONFIGURATION
# -----------------------------

@lru_cache(maxsize=1)
def load_llm():

    return ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )


llm = load_llm()


# -----------------------------
# PROMPT
# -----------------------------

prompt = ChatPromptTemplate.from_messages(
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

chain = prompt | llm


# -----------------------------
# PREDICTION FUNCTIONS
# -----------------------------

def parse_prediction(text):

    text = text.lower().strip()

    if "positive" in text:
        return "positive"

    if "negative" in text:
        return "negative"

    if "neutral" in text:
        return "neutral"

    return "parse_error"


def predict(headline, context):

    for _ in range(MAX_RETRIES):

        try:

            response = chain.invoke(
                {
                    "headline": headline,
                    "context": context
                }
            )

            label = parse_prediction(response.content)

            if label != "parse_error":
                return label

        except Exception as e:

            print(f"Prediction attempt failed: {e}")

    return "parse_error"


# -----------------------------
# HTML HELPERS
# -----------------------------

RESULT_STYLE = {
    "positive": ("pos", "up", "Positive", "The headline points to favourable news for the market."),
    "negative": ("neg", "down", "Negative", "The headline points to unfavourable news for the market."),
    "neutral": ("neu", "flat", "Neutral", "The headline reports facts without a clear market lean."),
    "parse_error": ("err", "warn", "No result", "The model did not return a valid label. Try again."),
}


def format_prediction(label):

    css, icon, title, note = RESULT_STYLE.get(label, RESULT_STYLE["parse_error"])

    return f"""
<div class="result {css}">
    <div class="result-icon">{svg(icon, 28)}</div>
    <div>
        <div class="result-label">{title}</div>
        <div class="result-note">{note}</div>
    </div>
</div>
"""


def empty_result():

    return """
<div class="result idle">
    <div class="result-note">Enter a headline and select Analyze to see the sentiment here.</div>
</div>
"""


def notice(kind, text):

    # kind: ok, warn, error
    return f'<div class="notice {kind}">{svg(kind, 18)}<span>{text}</span></div>'


# -----------------------------
# HISTORY
# -----------------------------

def create_history_dataframe(history):

    return pd.DataFrame(
        history,
        columns=[
            "Headline",
            "Prediction",
            "Documents Retrieved"
        ]
    )


# -----------------------------
# SINGLE PREDICTION
# -----------------------------

def single_prediction(headline, history):

    if not headline or not headline.strip():

        return (
            empty_result(),
            "",
            create_history_dataframe(history),
            history,
            notice("warn", "Headline is empty. Type or paste one to continue.")
        )

    try:

        docs = rag_fusion_retrieve(headline)

    except Exception as e:

        return (
            format_prediction("parse_error"),
            "",
            create_history_dataframe(history),
            history,
            notice("error", f"Retrieval failed: {str(e)}")
        )

    if not docs:

        return (
            format_prediction("parse_error"),
            "",
            create_history_dataframe(history),
            history,
            notice("warn", "No related documents were found for this headline.")
        )

    context = "\n\n".join(docs)

    prediction = predict(
        headline,
        context
    )

    history = history.copy()

    history.insert(
        0,
        [
            headline,
            prediction,
            len(docs)
        ]
    )

    context_display = "\n\n".join(
        [
            f"**Document {i}**\n\n{doc}"
            for i, doc in enumerate(docs, 1)
        ]
    )

    return (
        format_prediction(prediction),
        context_display,
        create_history_dataframe(history),
        history,
        notice("ok", f"Done. Used {len(docs)} retrieved documents.")
    )


# -----------------------------
# BATCH PREDICTION
# -----------------------------

def batch_prediction(file_path, progress=gr.Progress()):

    if file_path is None:

        return (
            pd.DataFrame(),
            None,
            notice("warn", "Upload a CSV file first.")
        )

    try:

        df = pd.read_csv(file_path)

    except Exception as e:

        return (
            pd.DataFrame(),
            None,
            notice("error", f"Could not read the CSV: {str(e)}")
        )

    if "title" not in df.columns:

        return (
            pd.DataFrame(),
            None,
            notice("error", "The CSV needs a column named title.")
        )

    if len(df) == 0:

        return (
            pd.DataFrame(),
            None,
            notice("warn", "The CSV has no rows.")
        )

    predictions = []

    total = len(df)

    for i, row in df.iterrows():

        headline = str(row["title"])

        progress(
            (i + 1) / total,
            desc=f"Processing {i + 1}/{total}"
        )

        try:

            docs = rag_fusion_retrieve(headline)

            context = "\n\n".join(docs)

            prediction = predict(
                headline,
                context
            )

        except Exception as e:

            print(f"Error processing row {i}: {e}")

            prediction = "parse_error"

        predictions.append(prediction)

    df["predicted_label"] = predictions

    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".csv"
    )

    output_path = output_file.name

    output_file.close()

    df.to_csv(
        output_path,
        index=False
    )

    return (
        df,
        output_path,
        notice("ok", f"Done. Classified {total} headlines.")
    )


# -----------------------------
# CUSTOM CSS
# -----------------------------

CSS = """
.gradio-container {
    max-width: 980px !important;
    margin: auto !important;
}

/* header */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 22px 0;
}
.app-logo {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: #1d3b53;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.app-title {
    font-size: 26px;
    font-weight: 700;
    line-height: 1.2;
    margin: 0;
}
.app-sub {
    font-size: 14.5px;
    opacity: 0.7;
    margin: 4px 0 0 0;
}

/* result card */
.result {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 22px 24px;
    border-radius: 12px;
    border: 1px solid var(--border-color-primary);
    min-height: 96px;
}
.result-icon {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: #fff;
}
.result-label {
    font-size: 28px;
    font-weight: 700;
    line-height: 1.15;
}
.result-note {
    font-size: 14.5px;
    opacity: 0.75;
    margin-top: 4px;
}
.result.pos { background: rgba(30, 140, 90, 0.10); border-color: rgba(30, 140, 90, 0.45); }
.result.pos .result-icon { background: #1e8c5a; }
.result.pos .result-label { color: #1e8c5a; }
.result.neg { background: rgba(200, 60, 50, 0.10); border-color: rgba(200, 60, 50, 0.45); }
.result.neg .result-icon { background: #c83c32; }
.result.neg .result-label { color: #c83c32; }
.result.neu { background: rgba(190, 130, 30, 0.10); border-color: rgba(190, 130, 30, 0.45); }
.result.neu .result-icon { background: #be821e; }
.result.neu .result-label { color: #be821e; }
.result.err .result-icon { background: #777; }
.result.idle { border-style: dashed; justify-content: center; text-align: center; }

/* status line */
.notice {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14.5px;
    margin: 4px 0;
}
.notice.ok { background: rgba(30, 140, 90, 0.12); color: #1e8c5a; }
.notice.warn { background: rgba(190, 130, 30, 0.14); color: #b07712; }
.notice.error { background: rgba(200, 60, 50, 0.12); color: #c83c32; }

/* section headings */
.section-title {
    font-size: 17px;
    font-weight: 650;
    margin: 18px 0 8px 0;
}

/* buttons with icons */
#analyze-btn::before,
#batch-btn::before {
    content: "";
    display: inline-block;
    width: 17px;
    height: 17px;
    margin-right: 8px;
    vertical-align: -3px;
    background: currentColor;
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-size: contain;
    mask-size: contain;
}
#analyze-btn::before {
    -webkit-mask-image: url("__SEARCH__");
    mask-image: url("__SEARCH__");
}
#batch-btn::before {
    -webkit-mask-image: url("__UPLOAD__");
    mask-image: url("__UPLOAD__");
}
button.primary { font-weight: 650 !important; }

.app-footer {
    text-align: center;
    font-size: 13px;
    opacity: 0.55;
    padding: 24px 0 8px 0;
}
""".replace("__SEARCH__", icon_uri("search")).replace("__UPLOAD__", icon_uri("upload"))


# -----------------------------
# GRADIO APPLICATION
# -----------------------------

theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
)

with gr.Blocks(
    theme=theme,
    css=CSS,
    title="Financial Sentiment"
) as demo:

    history_state = gr.State([])

    gr.HTML(
        f"""
        <div class="app-header">
            <div class="app-logo">{svg("logo", 26)}</div>
            <div>
                <h1 class="app-title">Financial Sentiment</h1>
                <p class="app-sub">Classify market headlines as positive, neutral or negative, using retrieved context.</p>
            </div>
        </div>
        """
    )

    with gr.Tabs():

        # =========================
        # SINGLE PREDICTION
        # =========================

        with gr.Tab("Single headline"):

            headline_input = gr.Textbox(
                label="Headline",
                placeholder="Example: Apple reports record quarterly revenue and profit",
                lines=3
            )

            analyze_button = gr.Button(
                "Analyze",
                variant="primary",
                elem_id="analyze-btn"
            )

            status_output = gr.HTML()

            gr.HTML('<div class="section-title">Result</div>')

            prediction_output = gr.HTML(value=empty_result())

            with gr.Accordion(
                "Retrieved context",
                open=False
            ):

                context_output = gr.Markdown()

            gr.HTML('<div class="section-title">Recent headlines</div>')

            history_output = gr.Dataframe(
                headers=[
                    "Headline",
                    "Prediction",
                    "Documents Retrieved"
                ],
                interactive=False,
                wrap=True
            )

            analyze_button.click(
                fn=single_prediction,
                inputs=[
                    headline_input,
                    history_state
                ],
                outputs=[
                    prediction_output,
                    context_output,
                    history_output,
                    history_state,
                    status_output
                ]
            )

        # =========================
        # BATCH PREDICTION
        # =========================

        with gr.Tab("Batch CSV"):

            gr.Markdown(
                "Upload a CSV with a **title** column. Each row gets a predicted label."
            )

            csv_input = gr.File(
                label="CSV file",
                file_types=[".csv"],
                type="filepath"
            )

            batch_button = gr.Button(
                "Run batch",
                variant="primary",
                elem_id="batch-btn"
            )

            batch_status = gr.HTML()

            batch_output = gr.Dataframe(
                label="Results",
                interactive=False,
                wrap=True
            )

            download_output = gr.File(
                label="Download results"
            )

            batch_button.click(
                fn=batch_prediction,
                inputs=[csv_input],
                outputs=[
                    batch_output,
                    download_output,
                    batch_status
                ]
            )

    gr.HTML(
        """
        <div class="app-footer">
            RAG Fusion, ChromaDB, Finance-Bot 3B, LangChain, Ollama, Gradio
        </div>
        """
    )


# -----------------------------
# LAUNCH
# -----------------------------

if __name__ == "__main__":

    demo.launch(share=True)