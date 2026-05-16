# ============================================================
#  POSITIVE QUOTE GENERATOR — Streamlit + OpenAI
#  Author  : You!
#  Purpose : Takes a single word from the user and uses
#            OpenAI's GPT-4o model to generate a warm,
#            uplifting quote built around that word.
# ============================================================

# ── SECTION 1 : IMPORTS ──────────────────────────────────────
# streamlit  → builds the interactive web UI
# openai     → official Python SDK to talk to OpenAI's API
# os         → lets us read environment variables (API key)

import streamlit as st
from openai import OpenAI
import os


# ── SECTION 2 : PAGE CONFIGURATION ───────────────────────────
# Must be the very first Streamlit call in the script.
# Sets the browser tab title, icon, and layout width.

st.set_page_config(
    page_title="Positive Quote Generator",
    page_icon="✨",
    layout="centered",
)


# ── SECTION 3 : CUSTOM CSS STYLING ───────────────────────────
# Injects raw CSS into the page so the app looks polished.
# We use a warm amber/cream palette with serif typography —
# intentionally different from generic AI app aesthetics.

st.markdown(
    """
    <style>
      /* ── Google Font import ── */
      @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;1,400&family=Source+Serif+4:wght@300;400&display=swap');

      /* ── Global reset ── */
      html, body, [class*="css"] {
          font-family: 'Source Serif 4', serif;
          background-color: #fdf6ec;
          color: #2c1a0e;
      }

      /* ── App-level container ── */
      .block-container {
          max-width: 720px;
          padding-top: 3rem;
      }

      /* ── Headline ── */
      h1 {
          font-family: 'Playfair Display', serif;
          font-size: 2.6rem !important;
          color: #b5541a;
          letter-spacing: -0.5px;
          margin-bottom: 0.2rem !important;
      }

      /* ── Sub-headline ── */
      .sub {
          font-size: 1.05rem;
          color: #7a4f2e;
          margin-bottom: 2rem;
          font-style: italic;
      }

      /* ── Text input box ── */
      .stTextInput > div > div > input {
          background-color: #fff8f0;
          border: 1.5px solid #d4935a;
          border-radius: 10px;
          padding: 0.7rem 1rem;
          font-size: 1.1rem;
          color: #2c1a0e;
          font-family: 'Source Serif 4', serif;
      }
      .stTextInput > div > div > input:focus {
          border-color: #b5541a;
          box-shadow: 0 0 0 3px rgba(181,84,26,0.15);
      }

      /* ── Button ── */
      .stButton > button {
          background-color: #b5541a;
          color: #fff8f0;
          border: none;
          border-radius: 10px;
          padding: 0.65rem 2.2rem;
          font-size: 1rem;
          font-family: 'Playfair Display', serif;
          letter-spacing: 0.5px;
          transition: background 0.2s ease, transform 0.1s ease;
          width: 100%;
      }
      .stButton > button:hover {
          background-color: #8f3d10;
          transform: translateY(-1px);
      }

      /* ── Quote card ── */
      .quote-card {
          background: linear-gradient(135deg, #fff3e0, #ffe4c4);
          border-left: 5px solid #b5541a;
          border-radius: 14px;
          padding: 1.8rem 2rem;
          margin-top: 2rem;
          box-shadow: 0 6px 24px rgba(181,84,26,0.12);
      }
      .quote-text {
          font-family: 'Playfair Display', serif;
          font-size: 1.35rem;
          font-style: italic;
          line-height: 1.75;
          color: #3b1f0a;
      }
      .quote-word-tag {
          display: inline-block;
          margin-top: 1rem;
          background-color: #b5541a;
          color: #fff8f0;
          font-size: 0.78rem;
          padding: 0.25rem 0.8rem;
          border-radius: 20px;
          letter-spacing: 1px;
          text-transform: uppercase;
          font-family: 'Source Serif 4', serif;
      }

      /* ── Error / warning box ── */
      .stAlert {
          border-radius: 10px;
      }

      /* ── Footer ── */
      .footer {
          text-align: center;
          font-size: 0.78rem;
          color: #b08060;
          margin-top: 3rem;
          padding-bottom: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── SECTION 4 : OPENAI CLIENT INITIALISATION ─────────────────
# We read the API key from Streamlit Secrets (the recommended
# way for deployed apps).  When running locally you can also
# set the environment variable OPENAI_API_KEY and it will be
# picked up automatically by the OpenAI SDK.
#
# HOW TO SET SECRETS ON STREAMLIT CLOUD:
#   Dashboard → your app → Settings → Secrets → paste:
#       OPENAI_API_KEY = "sk-..."
#
# HOW TO SET IT LOCALLY (one-time):
#   Create a file  .streamlit/secrets.toml  in the project root:
#       OPENAI_API_KEY = "sk-..."

@st.cache_resource
def get_openai_client() -> OpenAI:
    """
    Returns an authenticated OpenAI client.
    Tries Streamlit Secrets first, then falls back to the
    OPENAI_API_KEY environment variable.
    """
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "🔑 **API key not found.**  \n"
            "Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` (local) "
            "or to your Streamlit Cloud Secrets (deployed)."
        )
        st.stop()   # Halt execution — nothing else can run without the key.

    return OpenAI(api_key=api_key)


# ── SECTION 5 : QUOTE GENERATION FUNCTION ────────────────────
# This is the core logic of the app.
# We send a carefully crafted prompt to GPT-4o and return
# the model's response as a plain string.

def generate_quote(word: str, client: OpenAI) -> str:
    """
    Calls OpenAI GPT-4o to create a single positive, inspiring
    quote that naturally incorporates the given word.

    Parameters
    ----------
    word   : The keyword supplied by the user.
    client : An authenticated OpenAI client instance.

    Returns
    -------
    A string containing the generated quote.
    """

    # System prompt — defines the AI's persona and constraints
    system_prompt = (
        "You are a thoughtful, uplifting quote writer. "
        "Your quotes are warm, concise (1–2 sentences), and feel "
        "genuinely human — never generic or clichéd. "
        "You always incorporate the given word naturally and meaningfully. "
        "Return ONLY the quote text, no attribution, no quotation marks, "
        "no preamble."
    )

    # User prompt — the actual request with the keyword embedded
    user_prompt = (
        f"Write one original, positive, and inspiring quote that meaningfully "
        f"uses the word: \"{word}\"."
    )

    # ── API CALL ──────────────────────────────────────────────
    # model      : gpt-4o  (latest flagship model as of 2025)
    # max_tokens : 120 is plenty for a 1-2 sentence quote
    # temperature: 0.85 gives creative variety without going off-rails

    response = client.responses.create(
        model="gpt-5-nano",          # ← latest OpenAI model
        max_output_tokens=80,
        temperature=0.85,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    # Extract the text content from the API response object
    quote = response.output_text.strip()
    return quote


# ── SECTION 6 : PAGE HEADER ───────────────────────────────────
# Renders the title, subtitle, and a thin divider.

st.markdown("<h1>✨ Positive Quote Generator</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Give me a single word — I\'ll wrap it in light.</p>',
    unsafe_allow_html=True,
)
st.divider()


# ── SECTION 7 : USER INPUT ────────────────────────────────────
# A single text field for the keyword.
# strip() removes accidental leading/trailing spaces.

word_input = st.text_input(
    label="Enter a word",
    placeholder="e.g. resilience, morning, hope, kindness …",
    max_chars=40,
)

word = word_input.strip()   # clean up whitespace


# ── SECTION 8 : GENERATE BUTTON & OUTPUT ─────────────────────
# When the button is clicked, we:
#   1. Validate that the user actually typed something.
#   2. Initialise the OpenAI client.
#   3. Call generate_quote() inside a spinner (loading indicator).
#   4. Render the quote inside a styled card.

if st.button("Generate Quote ✦"):

    # ── 8a : Validation ───────────────────────────────────────
    if not word:
        st.warning("Please enter a word before generating a quote.")

    elif len(word.split()) > 1:
        # Remind users to stick to a single word for best results
        st.warning("For the best quote, please enter just **one word**.")

    else:
        # ── 8b : API call with loading spinner ────────────────
        client = get_openai_client()

        with st.spinner(f'Crafting your quote around **"{word}"** …'):
            try:
                quote = generate_quote(word, client)

                # ── 8c : Render the quote card ─────────────────
                st.markdown(
                    f"""
                    <div class="quote-card">
                        <div class="quote-text">"{quote}"</div>
                        <div class="quote-word-tag">✦ {word}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as e:
                # Catch any API or network errors and surface them clearly
                st.error(
                    f"⚠️ Something went wrong while calling the OpenAI API:\n\n`{e}`"
                )


# ── SECTION 9 : FOOTER ────────────────────────────────────────
st.markdown(
    '<div class="footer">Powered by OpenAI GPT-4o &nbsp;·&nbsp; Built with Streamlit</div>',
    unsafe_allow_html=True,
)
