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
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "sparkles": '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>',
    "table": '<path d="M12 3v18"/><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "type": '<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/><line x1="12" x2="12" y1="4" y2="20"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
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
    <div class="result-icon"><span class="pulse"></span>{svg(icon, 30)}</div>
    <div class="result-text">
        <div class="result-tag">Predicted sentiment</div>
        <div class="result-label">{title}</div>
        <div class="result-note">{note}</div>
    </div>
</div>
"""


def empty_result():

    return f"""
<div class="result idle">
    <div class="idle-icon">{svg("sparkles", 26)}</div>
    <div class="result-note">Enter a headline and select Analyze to see the sentiment here.</div>
</div>
"""


def notice(kind, text):

    # kind: ok, warn, error
    return f'<div class="notice {kind}">{svg(kind, 18)}<span>{text}</span></div>'


def section(icon, text):

    # small heading with an icon
    return f'<div class="section-title"><span class="section-icon">{svg(icon, 16)}</span>{text}</div>'


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
    max-width: 1020px !important;
    margin: auto !important;
    background:
        radial-gradient(900px 400px at 100% -10%, rgba(59, 130, 246, 0.10), transparent 60%),
        radial-gradient(700px 360px at -10% 0%, rgba(99, 102, 241, 0.08), transparent 60%);
}

/* hero header */
.hero {
    position: relative;
    overflow: hidden;
    margin: 22px 0 18px 0;
    padding: 34px 36px;
    border-radius: 20px;
    color: #fff;
    background: linear-gradient(125deg, #0b1e33 0%, #16385a 45%, #2563eb 100%);
    box-shadow: 0 18px 40px -18px rgba(37, 99, 235, 0.55);
}
.hero::before {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    right: -80px;
    top: -140px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.22), transparent 70%);
}
.hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    left: 38%;
    bottom: -170px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(125, 211, 252, 0.28), transparent 70%);
}
.hero-row {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 20px;
}
.app-logo {
    width: 62px;
    height: 62px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.28);
    backdrop-filter: blur(6px);
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.app-title {
    font-size: 32px;
    font-weight: 750;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0;
    color: #fff;
}
.app-sub {
    font-size: 15.5px;
    color: rgba(255, 255, 255, 0.80);
    margin: 6px 0 0 0;
}
.chips {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 22px;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    color: #fff;
    background: rgba(255, 255, 255, 0.13);
    border: 1px solid rgba(255, 255, 255, 0.25);
}

/* feature cards */
.features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 0 0 14px 0;
}
.feature {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 16px 18px;
    border-radius: 14px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-primary);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 26px -16px rgba(37, 99, 235, 0.6);
}
.feature-icon {
    width: 40px;
    height: 40px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: #fff;
}
.feature-icon.f1 { background: linear-gradient(135deg, #2563eb, #4f46e5); }
.feature-icon.f2 { background: linear-gradient(135deg, #0891b2, #2563eb); }
.feature-icon.f3 { background: linear-gradient(135deg, #059669, #0891b2); }
.feature-title { font-size: 15px; font-weight: 700; margin: 0; }
.feature-text { font-size: 13px; opacity: 0.68; margin: 3px 0 0 0; line-height: 1.4; }

/* tabs */
.tab-nav, div[role="tablist"] {
    border-bottom: 1px solid var(--border-color-primary) !important;
}
button[role="tab"] {
    font-weight: 650 !important;
    font-size: 15px !important;
    padding: 10px 18px !important;
}
button[role="tab"][aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 3px solid #2563eb !important;
}

/* result card */
.result {
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    gap: 22px;
    padding: 26px 28px;
    border-radius: 16px;
    border: 1px solid var(--border-color-primary);
    min-height: 116px;
    animation: rise 0.45s ease both;
}
.result::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 6px;
    background: #777;
}
@keyframes rise {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.result-icon {
    position: relative;
    width: 66px;
    height: 66px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: #fff;
}
.pulse {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background: inherit;
    animation: pulse 2s ease-out infinite;
    z-index: 0;
}
.result-icon svg { position: relative; z-index: 1; }
@keyframes pulse {
    0% { transform: scale(1); opacity: 0.55; }
    100% { transform: scale(1.7); opacity: 0; }
}
.result-tag {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.55;
}
.result-label {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-top: 2px;
}
.result-note {
    font-size: 14.5px;
    opacity: 0.75;
    margin-top: 5px;
}
.result.pos { background: linear-gradient(100deg, rgba(30, 140, 90, 0.16), rgba(30, 140, 90, 0.03)); border-color: rgba(30, 140, 90, 0.45); }
.result.pos::before { background: #1e8c5a; }
.result.pos .result-icon { background: #1e8c5a; }
.result.pos .result-label { color: #1e8c5a; }
.result.neg { background: linear-gradient(100deg, rgba(200, 60, 50, 0.16), rgba(200, 60, 50, 0.03)); border-color: rgba(200, 60, 50, 0.45); }
.result.neg::before { background: #c83c32; }
.result.neg .result-icon { background: #c83c32; }
.result.neg .result-label { color: #c83c32; }
.result.neu { background: linear-gradient(100deg, rgba(190, 130, 30, 0.16), rgba(190, 130, 30, 0.03)); border-color: rgba(190, 130, 30, 0.45); }
.result.neu::before { background: #be821e; }
.result.neu .result-icon { background: #be821e; }
.result.neu .result-label { color: #be821e; }
.result.err .result-icon { background: #777; }
.result.idle {
    flex-direction: column;
    gap: 10px;
    border-style: dashed;
    justify-content: center;
    text-align: center;
    animation: none;
}
.result.idle::before { display: none; }
.idle-icon {
    width: 50px;
    height: 50px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #2563eb;
    background: rgba(37, 99, 235, 0.10);
}

/* status line */
.notice {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 16px;
    border-radius: 10px;
    font-size: 14.5px;
    font-weight: 500;
    margin: 6px 0;
    animation: rise 0.3s ease both;
}
.notice.ok { background: rgba(30, 140, 90, 0.12); color: #1e8c5a; border: 1px solid rgba(30, 140, 90, 0.3); }
.notice.warn { background: rgba(190, 130, 30, 0.14); color: #b07712; border: 1px solid rgba(190, 130, 30, 0.3); }
.notice.error { background: rgba(200, 60, 50, 0.12); color: #c83c32; border: 1px solid rgba(200, 60, 50, 0.3); }

/* section headings */
.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 17px;
    font-weight: 700;
    margin: 22px 0 10px 0;
}
.section-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #2563eb;
    background: rgba(37, 99, 235, 0.12);
}

/* inputs */
textarea, input[type="text"] {
    font-size: 15.5px !important;
}

/* buttons with icons */
#analyze-btn::before,
#batch-btn::before {
    content: "";
    display: inline-block;
    width: 18px;
    height: 18px;
    margin-right: 9px;
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
button.primary {
    font-weight: 700 !important;
    font-size: 15.5px !important;
    padding: 12px 20px !important;
    border: none !important;
    border-radius: 12px !important;
    background: linear-gradient(120deg, #1d4ed8, #2563eb 55%, #0891b2) !important;
    color: #fff !important;
    box-shadow: 0 10px 22px -10px rgba(37, 99, 235, 0.8);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 28px -12px rgba(37, 99, 235, 0.9);
}

/* footer */
.app-footer {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    padding: 28px 0 10px 0;
}
.foot-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    opacity: 0.75;
    border: 1px solid var(--border-color-primary);
}

@media (max-width: 760px) {
    .features { grid-template-columns: 1fr; }
    .hero { padding: 24px 20px; }
    .app-title { font-size: 25px; }
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
        <div class="hero">
            <div class="hero-row">
                <div class="app-logo">{svg("logo", 32)}</div>
                <div>
                    <h1 class="app-title">Financial Sentiment</h1>
                    <p class="app-sub">Classify market headlines as positive, neutral or negative, using retrieved context.</p>
                </div>
            </div>
            <div class="chips">
                <span class="chip">{svg("database", 15)}RAG Fusion</span>
                <span class="chip">{svg("cpu", 15)}Finance-Bot 3B</span>
                <span class="chip">{svg("zap", 15)}Runs locally</span>
            </div>
        </div>

        <div class="features">
            <div class="feature">
                <div class="feature-icon f1">{svg("search", 20)}</div>
                <div>
                    <p class="feature-title">Retrieve</p>
                    <p class="feature-text">Finds related documents for every headline.</p>
                </div>
            </div>
            <div class="feature">
                <div class="feature-icon f2">{svg("target", 20)}</div>
                <div>
                    <p class="feature-title">Classify</p>
                    <p class="feature-text">Labels sentiment as positive, neutral or negative.</p>
                </div>
            </div>
            <div class="feature">
                <div class="feature-icon f3">{svg("download", 20)}</div>
                <div>
                    <p class="feature-title">Export</p>
                    <p class="feature-text">Run a full CSV and download the results.</p>
                </div>
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

            gr.HTML(section("sparkles", "Result"))

            prediction_output = gr.HTML(value=empty_result())

            with gr.Accordion(
                "Retrieved context",
                open=False
            ):

                context_output = gr.Markdown()

            gr.HTML(section("clock", "Recent headlines"))

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
        f"""
        <div class="app-footer">
            <span class="foot-chip">{svg("layers", 13) if "layers" in ICON_PATHS else svg("logo", 13)}Multi query RAG</span>
            <span class="foot-chip">{svg("database", 13)}ChromaDB</span>
            <span class="foot-chip">{svg("cpu", 13)}Finance-Bot 3B</span>
            <span class="foot-chip">{svg("type", 13)}LangChain</span>
            <span class="foot-chip">{svg("zap", 13)}Ollama</span>
            <span class="foot-chip">{svg("table", 13)}Gradio</span>
        </div>
        """
    )


# -----------------------------
# LAUNCH
# -----------------------------

if __name__ == "__main__":

    demo.launch(share=True)