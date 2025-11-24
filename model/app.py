import os
import re  # 👈 for formatting helper
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, session
from dotenv import load_dotenv
from google import genai
from google.genai import types

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# ==========================
#   LOAD ENV + GEMINI CLIENT
# ==========================

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-2.0-flash"

# ==========================
#   TRAIN MODEL ON STARTUP
# ==========================

print("[SOH MODEL] Training model...")

data = pd.read_feather("PulseBat.feather")
data = data.sort_values(by="SOC", ascending=True)

model_data = data[['Qn', 'Q', 'SOC', 'SOE'] + [f'U{i}' for i in range(1, 22)] + ['SOH']]
X = model_data[[f"U{i}" for i in range(1, 22)] + ["SOC", "SOE"]]
Y = model_data["SOH"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# --- Outlier removal ---
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

model = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linreg", LinearRegression())
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, Y, cv=cv, scoring='r2')
print("[SOH MODEL] CV mean R²:", np.mean(cv_scores))

model.fit(X_train_clean, Y_train_clean)
print("[SOH MODEL] Ready.\n")


def predict_soh(U_values, soc, soe):
    """U_values: list of 21 floats; soc, soe: floats"""
    if len(U_values) != 21:
        raise ValueError(f"Expected 21 voltages, got {len(U_values)}")
    vec = np.array(U_values + [soc, soe]).reshape(1, -1)
    return float(model.predict(vec)[0])


def format_bot_text(text: str) -> str:
    """
    Convert simple markdown (**bold**) and newlines to HTML
    so they render nicely in the chat UI.
    """
    if not text:
        return ""
    # **bold** -> <b>bold</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # new lines -> <br>
    text = text.replace("\n", "<br>")
    return text


# ==========================
#        FLASK APP
# ==========================

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret"  # needed for session


@app.route("/", methods=["GET", "POST"])
def index():
    """Original form-based UI (optional, keep as backup)."""
    prediction = None
    status = None
    error = None

    if request.method == "POST":
        try:
            threshold = float(request.form.get("threshold", "0.8"))
            soc = float(request.form.get("soc", "0"))
            soe = float(request.form.get("soe", "0"))

            U_values = []
            for i in range(1, 22):
                val = float(request.form.get(f"U{i}", "0"))
                U_values.append(val)

            soh = predict_soh(U_values, soc, soe)
            prediction = round(soh, 4)
            status = "Healthy" if soh >= threshold else "Has a Problem"

        except ValueError:
            error = "Please enter valid numeric values."

    return render_template(
        "index.html",
        prediction=prediction,
        status=status,
        error=error
    )


# ==========================
#        CHATBOT ROUTE
# ==========================

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    # simple session-based state
    if "history" not in session:
        session["history"] = []
    if "mode" not in session:
        session["mode"] = "normal"   # normal / await_voltages / await_soc / await_soe
    if "soh_data" not in session:
        session["soh_data"] = {}

    history = session["history"]
    mode = session["mode"]
    soh_data = session["soh_data"]

    bot_reply = None

    if request.method == "POST":
        action = request.form.get("action", "send")

        # --- Handle "New chat" button ---
        if action == "reset":
            history = []
            mode = "normal"
            soh_data = {}
            bot_reply = format_bot_text("Starting a new conversation. How can I help you?")
            history.append(("bot", bot_reply))

            session["history"] = history
            session["mode"] = mode
            session["soh_data"] = soh_data

            return render_template("chat.html", history=history)

        # Otherwise, normal send
        user_msg = request.form.get("message", "").strip()
        if user_msg:
            history.append(("user", user_msg))

            # --------- STATE MACHINE ----------
            # 1) If in normal mode and user asks for prediction
            if mode == "normal" and (
                "predict soh" in user_msg.lower()
                or "battery health" in user_msg.lower()
            ):
                mode = "await_voltages"
                soh_data = {}
                bot_reply = (
                    "Okay, let's check your battery health.\n"
                    "Please send 21 cell voltages (U1–U21) in one line, "
                    "separated by spaces."
                )
                bot_reply = format_bot_text(bot_reply)

            # 2) Collect voltages
            elif mode == "await_voltages":
                try:
                    parts = user_msg.split()
                    U_values = [float(x) for x in parts]
                    if len(U_values) != 21:
                        raise ValueError
                    soh_data["U_values"] = U_values
                    mode = "await_soc"
                    bot_reply = "Got the voltages. Now enter SOC (as a number)."
                    bot_reply = format_bot_text(bot_reply)
                except ValueError:
                    bot_reply = (
                        "I couldn't read that as 21 numbers.\n"
                        "Please enter exactly 21 numeric voltages separated by spaces."
                    )
                    bot_reply = format_bot_text(bot_reply)

            # 3) Collect SOC
            elif mode == "await_soc":
                try:
                    soc = float(user_msg)
                    soh_data["soc"] = soc
                    mode = "await_soe"
                    bot_reply = "Thanks. Now enter SOE (as a number)."
                    bot_reply = format_bot_text(bot_reply)
                except ValueError:
                    bot_reply = "Please enter SOC as a numeric value."
                    bot_reply = format_bot_text(bot_reply)

            # 4) Collect SOE and run prediction
            elif mode == "await_soe":
                try:
                    soe = float(user_msg)
                    soh_data["soe"] = soe

                    U_values = soh_data["U_values"]
                    soc = soh_data["soc"]

                    soh = predict_soh(U_values, soc, soe)
                    status = "Healthy" if soh >= 0.8 else "Has a Problem"

                    # Ask Gemini to explain the result
                    explanation_prompt = (
                        f"The predicted battery SOH is {soh:.4f} and the status is '{status}'. "
                        "Explain what this means in simple terms for a non-expert user."
                    )
                    contents = [
                        types.Content(
                            role="user",
                            parts=[types.Part(text=explanation_prompt)]
                        )
                    ]

                    chunks = []
                    for chunk in client.models.generate_content_stream(
                        model=GEMINI_MODEL,
                        contents=contents,
                        config=types.GenerateContentConfig()
                    ):
                        if chunk.text:
                            chunks.append(chunk.text)
                    explanation = "".join(chunks)

                    bot_reply_raw = (
                        f"Predicted SOH: {soh:.4f}\n"
                        f"Battery Status: {status}\n\n"
                        f"{explanation}"
                    )
                    bot_reply = format_bot_text(bot_reply_raw)

                    # reset state back to normal
                    mode = "normal"
                    soh_data = {}

                except ValueError:
                    bot_reply = "Please enter SOE as a numeric value."
                    bot_reply = format_bot_text(bot_reply)

            # 5) Normal chat mode → forward to Gemini
            elif mode == "normal":
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part(text=user_msg)]
                    )
                ]
                chunks = []
                for chunk in client.models.generate_content_stream(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig()
                ):
                    if chunk.text:
                        chunks.append(chunk.text)
                bot_reply = format_bot_text("".join(chunks))

            # ---------------------------------

            if bot_reply:
                history.append(("bot", bot_reply))

            # store back in session
            session["history"] = history
            session["mode"] = mode
            session["soh_data"] = soh_data

    return render_template("chat.html", history=history)


if __name__ == "__main__":
    print("\nFlask server running!")
    print("Manual SOH form → http://127.0.0.1:5000/")
    print("Chatbot UI      → http://127.0.0.1:5000/chatbot\n")

    app.run(debug=True, use_reloader=False)

