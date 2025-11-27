import os
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, session
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

from dotenv import load_dotenv

from google import genai
from google.genai import types


# ---------------------------
# Paths & Flask app setup
# ---------------------------

# BASE_DIR = folder where app.py lives (Optimus/model)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ROOT_DIR = project root (Optimus)
ROOT_DIR = os.path.dirname(BASE_DIR)

TEMPLATE_DIR = os.path.join(ROOT_DIR, "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "static")

# Tell Flask exactly where templates and static live
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")


# ---------------------------
# Environment & Gemini setup
# ---------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in the .env file")

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"


# ---------------------------
# Load data & train models
# ---------------------------

print("[SOH MODEL] Training model...")

DATA_PATH = os.path.join(BASE_DIR, "PulseBat.feather")

data = pd.read_feather(DATA_PATH)
data = data.sort_values(by="SOC", ascending=True)

# Features: 21 voltages + SOC + SOE + target SOH
U_COLS = [f"U{i}" for i in range(1, 22)]
model_data = data[["Qn", "Q", "SOC", "SOE"] + U_COLS + ["SOH"]]

X = model_data[U_COLS + ["SOC", "SOE"]]
Y = model_data["SOH"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, shuffle=False
)

# --- Outlier removal for main model ---
_baseline = LinearRegression().fit(X_train, Y_train)
_resid = Y_train - _baseline.predict(X_train)

med = np.median(_resid)
mad = np.median(np.abs(_resid - med))

if mad == 0:
    keep_mask = np.ones_like(_resid, dtype=bool)
else:
    tol = 3.5 * mad
    keep_mask = np.abs(_resid - med) <= tol

X_train_clean = X_train[keep_mask]
Y_train_clean = Y_train[keep_mask]

# ========= MAIN MODEL (full 21 voltages) =========
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linreg", LinearRegression()),
    ]
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, Y, cv=cv, scoring="r2")
print("[SOH MODEL] CV mean R² (full features):", np.mean(cv_scores))

model.fit(X_train_clean, Y_train_clean)
print("[SOH MODEL] Full-feature model ready.")

# ========= FALLBACK MODEL (mean(U1..U21) + SOC + SOE) =========
X_fallback_train = pd.DataFrame(
    {
        "U_mean": X_train_clean[U_COLS].mean(axis=1),
        "SOC": X_train_clean["SOC"],
        "SOE": X_train_clean["SOE"],
    }
)

fallback_model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("linreg", LinearRegression()),
    ]
)

cv_fallback_scores = cross_val_score(
    fallback_model, X_fallback_train, Y_train_clean, cv=cv, scoring="r2"
)
print("[SOH MODEL] CV mean R² (fallback):", np.mean(cv_fallback_scores))

fallback_model.fit(X_fallback_train, Y_train_clean)
print("[SOH MODEL] Fallback model ready.\n")


# ---------------------------
# Helper functions
# ---------------------------

def predict_soh(U_values, soc, soe):
    """
    U_values: list of 1–21 floats
    soc, soe: floats

    Returns:
        soh (float), used_fallback (bool)
        used_fallback=True means we used the simpler model.
    """
    if not U_values:
        raise ValueError("At least one voltage value is required.")

    U_values = [float(v) for v in U_values]

    # Full model case: exactly 21 voltages
    if len(U_values) == 21:
        vec = np.array(U_values + [soc, soe]).reshape(1, -1)
        soh = float(model.predict(vec)[0])
        return soh, False

    # Fallback case: 1–20 voltages
    mean_u = float(np.mean(U_values))
    vec_fb = np.array([mean_u, soc, soe]).reshape(1, -1)
    soh = float(fallback_model.predict(vec_fb)[0])
    return soh, True


def format_bot_text(text: str) -> str:
    """Convert newlines to <br> for HTML display."""
    return text.replace("\n", "<br>")


def detect_simple_intent(message: str):
    """
    Very lightweight intent detection based on keywords.
    Returns one of:
        "help", "greeting", "goodbye", "thanks", "cancel", "restart", None
    """
    msg = message.lower().strip()

    if not msg:
        return None

    # Help
    if any(word in msg for word in ["help", "what can you do", "how to use"]):
        return "help"

    # Greetings
    if any(word in msg for word in ["hi", "hello", "hey", "good morning", "good evening"]):
        return "greeting"

    # Goodbye
    if any(word in msg for word in ["bye", "goodbye", "see you", "good night"]):
        return "goodbye"

    # Thanks
    if any(word in msg for word in ["thanks", "thank you", "thx", "tysm"]):
        return "thanks"

    # Cancel ongoing flow
    if any(word in msg for word in ["cancel", "stop", "nevermind", "never mind"]):
        return "cancel"

    # Restart / redo
    if any(
        word in msg
        for word in ["restart", "redo", "start over", "reset", "new check", "new test"]
    ):
        return "restart"

    return None


def gemini_reply(prompt: str) -> str:
    """
    Send a prompt to Gemini and get back a single string.
    This version avoids Part.from_text and just passes the prompt string.
    """
    chunks = []
    for chunk in client.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=prompt,  # just a plain string
        config=types.GenerateContentConfig(),
    ):
        # Some SDK versions set .text, some use .candidates[0].content.parts[0].text
        if getattr(chunk, "text", None):
            chunks.append(chunk.text)
        elif getattr(chunk, "candidates", None):
            try:
                text_part = chunk.candidates[0].content.parts[0].text
                chunks.append(text_part)
            except Exception:
                pass

    return "".join(chunks)


# ---------------------------
# Routes: Manual form
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None
    status = None
    used_fallback = False

    # Default threshold 0.8 (80%), can be changed by user.
    threshold = 0.8

    if request.method == "POST":
        # Threshold
        threshold_str = request.form.get("threshold", "0.8").strip()
        if threshold_str:
            try:
                threshold = float(threshold_str)
            except ValueError:
                error = "Threshold must be a numeric value."
                return render_template(
                    "index.html",
                    prediction=prediction,
                    status=status,
                    error=error,
                    threshold=threshold,
                )

        # SOC & SOE
        try:
            soc_str = request.form.get("soc", "").strip()
            soe_str = request.form.get("soe", "").strip()
            soc = float(soc_str)
            soe = float(soe_str)
        except ValueError:
            error = "Please enter valid numeric values for SOC and SOE."
            return render_template(
                "index.html",
                prediction=prediction,
                status=status,
                error=error,
                threshold=threshold,
            )

        # Voltages U1..U21 (allow missing ones -> fallback model if <21)
        U_values = []
        try:
            for i in range(1, 22):
                val_str = request.form.get(f"U{i}", "").strip()
                if val_str == "":
                    # Skip empty fields – fallback model will be used if <21
                    continue
                U_values.append(float(val_str))
        except ValueError:
            error = "Please enter valid numeric values for the voltages."
            return render_template(
                "index.html",
                prediction=prediction,
                status=status,
                error=error,
                threshold=threshold,
            )

        if not U_values:
            error = "Please provide at least one voltage value."
            return render_template(
                "index.html",
                prediction=prediction,
                status=status,
                error=error,
                threshold=threshold,
            )

        try:
            soh, used_fallback = predict_soh(U_values, soc, soe)
        except ValueError as e:
            error = str(e)
            return render_template(
                "index.html",
                prediction=prediction,
                status=status,
                error=error,
                threshold=threshold,
            )

        prediction = round(soh, 4)
        status = "Healthy" if soh >= threshold else "Has a Problem"

    return render_template(
        "index.html",
        prediction=prediction,
        status=status,
        error=error,
        threshold=threshold,
        used_fallback=used_fallback,
    )


# ---------------------------
# Routes: Chatbot
# ---------------------------

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    # Initialize session state
    history = session.get("history", [])
    mode = session.get("mode", "normal")
    soh_data = session.get("soh_data", {})  # to store voltages, soc, soe, threshold

    if request.method == "GET":
        # Reset state when opening the page
        history = []
        mode = "normal"
        soh_data = {}
        session["history"] = history
        session["mode"] = mode
        session["soh_data"] = soh_data
        return render_template("chat.html", history=history)

    # POST (user sent a message)
    user_msg = request.form.get("message", "").strip()
    if not user_msg:
        return render_template("chat.html", history=history)

    history.append(("user", user_msg))

    bot_reply = None
    lower_msg = user_msg.lower()

    # -------- Simple intents (help, greetings, cancel, etc.) --------
    simple_intent = detect_simple_intent(lower_msg)

    if simple_intent == "help":
        bot_reply = (
            "I can do two main things:\n"
            "1. **Check battery SOH** – say something like 'check battery soh' or "
            "'predict soh', and I'll guide you through entering voltages, SOC and SOE.\n"
            "2. **Answer general battery questions** – ask me things like "
            "'how to extend battery life?' and I’ll explain using Gemini."
        )
        bot_reply = format_bot_text(bot_reply)

    elif simple_intent == "greeting" and mode == "normal":
        bot_reply = (
            "Hi! I'm Optimus, your battery health assistant.\n"
            "You can say **'check battery soh'** to start an SOH check, or "
            "ask any general battery question."
        )
        bot_reply = format_bot_text(bot_reply)

    elif simple_intent == "goodbye":
        bot_reply = "Goodbye! If you need another SOH check later, just come back. 😊"
        bot_reply = format_bot_text(bot_reply)
        mode = "normal"
        soh_data = {}

    elif simple_intent == "thanks":
        bot_reply = "You're welcome! Happy to help. ⚡"
        bot_reply = format_bot_text(bot_reply)

    elif simple_intent == "cancel":
        bot_reply = (
            "Okay, I've cancelled the current SOH check.\n"
            "You can start a new one any time by saying **'check battery soh'**."
        )
        bot_reply = format_bot_text(bot_reply)
        mode = "normal"
        soh_data = {}

    elif simple_intent == "restart":
        # Restart the SOH flow from threshold question
        soh_data = {}
        mode = "await_threshold"
        bot_reply = (
            "No problem, let's restart the SOH check.\n"
            "The default SOH threshold for a 'healthy' battery is **0.8 (80%)**.\n"
            "If you'd like a different threshold, enter a number like `0.75`.\n"
            "Or type `default` to keep 0.8."
        )
        bot_reply = format_bot_text(bot_reply)

    # If no simple intent handled it, run the state machine
    if bot_reply is None:
        # ---------------- STATE: NORMAL ----------------
        if mode == "normal":
            # Start SOH check
            if any(
                key in lower_msg
                for key in [
                    "check battery soh",
                    "predict soh",
                    "battery health",
                    "soh check",
                ]
            ):
                mode = "await_threshold"
                soh_data = {}
                bot_reply = (
                    "Great, let's check your battery SOH.\n\n"
                    "The default SOH threshold for a 'healthy' battery is **0.8 (80%)**.\n"
                    "If you want to use a different threshold, enter a number like `0.75`.\n"
                    "Or type `default` (or just press enter) to keep 0.8."
                )
                bot_reply = format_bot_text(bot_reply)
            else:
                # General chat → Gemini
                prompt = (
                    "You are a helpful battery and energy storage assistant.\n"
                    "Answer the following user question clearly and concisely:\n\n"
                    f"User: {user_msg}"
                )
                reply_text = gemini_reply(prompt)
                bot_reply = format_bot_text(reply_text)

        # ---------------- STATE: AWAIT_THRESHOLD ----------------
        elif mode == "await_threshold":
            text = lower_msg.strip()
            if text in ["", "default", "keep", "no", "skip"]:
                threshold = 0.8
            else:
                try:
                    threshold = float(user_msg)
                except ValueError:
                    bot_reply = (
                        "Please enter a numeric threshold like `0.8` or `0.75`,\n"
                        "or type `default` to keep 0.8."
                    )
                    bot_reply = format_bot_text(bot_reply)
                    # stay in await_threshold
                    history.append(("bot", bot_reply))
                    session["history"] = history
                    session["mode"] = mode
                    session["soh_data"] = soh_data
                    return render_template("chat.html", history=history)

            soh_data["threshold"] = threshold
            mode = "await_voltages"
            bot_reply = (
                f"Got it. I'll use **{threshold:.2f}** as the SOH threshold.\n\n"
                "Now please enter your cell voltages.\n"
                "You can enter **between 1 and 21** voltages separated by spaces.\n"
                "- If you enter all 21, I'll use the most accurate model.\n"
                "- If you enter fewer than 21, I'll use a simplified, less accurate model."
            )
            bot_reply = format_bot_text(bot_reply)

        # ---------------- STATE: AWAIT_VOLTAGES ----------------
        elif mode == "await_voltages":
            try:
                parts = user_msg.split()
                U_values = [float(x) for x in parts]

                if len(U_values) == 0 or len(U_values) > 21:
                    raise ValueError

                soh_data["U_values"] = U_values
                mode = "await_soc"

                if len(U_values) == 21:
                    msg = (
                        "Great, I got all **21 voltages**.\n"
                        "Now please enter **SOC** as a number (e.g., `50` for 50%)."
                    )
                else:
                    msg = (
                        f"Thanks, I received **{len(U_values)}** voltages.\n"
                        "I'll use a simplified model based on their average, "
                        "so this prediction will be **less accurate**.\n\n"
                        "Now please enter **SOC** as a number (e.g., `50` for 50%)."
                    )

                bot_reply = format_bot_text(msg)

            except ValueError:
                msg = (
                    "I couldn't read that correctly.\n"
                    "Please enter **between 1 and 21 numeric voltages** separated by spaces."
                )
                bot_reply = format_bot_text(msg)

        # ---------------- STATE: AWAIT_SOC ----------------
        elif mode == "await_soc":
            try:
                soc = float(user_msg)
                soh_data["soc"] = soc
                mode = "await_soe"
                msg = "Got it. Now please enter **SOE** as a number."
                bot_reply = format_bot_text(msg)
            except ValueError:
                msg = "Please enter SOC as a numeric value (e.g., `50` for 50%)."
                bot_reply = format_bot_text(msg)

        # ---------------- STATE: AWAIT_SOE ----------------
        elif mode == "await_soe":
            try:
                soe = float(user_msg)
                soh_data["soe"] = soe

                U_values = soh_data.get("U_values", [])
                soc = soh_data.get("soc")
                threshold = soh_data.get("threshold", 0.8)

                soh, used_fallback = predict_soh(U_values, soc, soe)
                status = "Healthy" if soh >= threshold else "Has a Problem"

                # Explanation prompt
                if used_fallback:
                    explanation_prompt = (
                        f"The predicted battery SOH is {soh:.4f} and the status is '{status}'. "
                        "The prediction was made using a simplified model that only had access "
                        "to the **average** of a partial set of voltages plus SOC and SOE, "
                        "so it is less accurate than the full model. "
                        "Explain what this means in simple terms for a non-expert user, and "
                        "mention that the result is approximate because not all voltages were provided."
                    )
                else:
                    explanation_prompt = (
                        f"The predicted battery SOH is {soh:.4f} and the status is '{status}'. "
                        "The prediction was made using all 21 cell voltages plus SOC and SOE. "
                        "Explain what this means in simple terms for a non-expert user."
                    )

                explanation = gemini_reply(explanation_prompt)

                msg = (
                    f"**Predicted SOH:** {soh:.4f}\n"
                    f"**Battery Status:** {status}\n"
                    f"**Threshold Used:** {threshold:.2f}\n\n"
                    f"{explanation}"
                )

                bot_reply = format_bot_text(msg)

                # Reset back to normal chat mode
                mode = "normal"
                soh_data = {}

            except ValueError:
                msg = "Please enter SOE as a numeric value."
                bot_reply = format_bot_text(msg)

        else:
            # Safety fallback
            mode = "normal"
            soh_data = {}
            msg = (
                "Something went wrong with the conversation state, "
                "so I reset the SOH check.\n"
                "You can start again by saying **'check battery soh'**."
            )
            bot_reply = format_bot_text(msg)

    # Store state back in session
    if bot_reply:
        history.append(("bot", bot_reply))

    session["history"] = history
    session["mode"] = mode
    session["soh_data"] = soh_data

    return render_template("chat.html", history=history)


# ---------------------------
# Main entrypoint
# ---------------------------

if __name__ == "__main__":
    print("\nFlask server running!")
    print("Manual SOH form -> http://127.0.0.1:5000/")
    print("Chatbot UI      -> http://127.0.0.1:5000/chatbot\n")

    app.run(debug=True, use_reloader=False)
