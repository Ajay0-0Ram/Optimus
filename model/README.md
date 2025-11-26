**Prerequisites:**  
Python 3.10 or higher
VS Code (recommended) with the Python extension

**Installation (VS Code):**  
Open the project folder in VS Code.

**Create a virtual environment (recommended):**  
**Windows (PowerShell) - paste each one by one into terminal**  
python -m venv .venv  
.venv\Scripts\Activate.ps1

**macOS / Linux - paste each one by one into terminal**  
python3 -m venv .venv  
source .venv/bin/activate

**Select the virtual environment in VS Code:**  
Ctrl+Shift+P → “Python: Select Interpreter” → choose .venv.

**Install Dependencies:**  
pip install pandas numpy scikit-learn matplotlib pyarrow flask python-dotenv google-genai  
pyarrow is required to read the Feather file format.

**How to Run:**  
Make sure main.py, app.py and PulseBat.feather are in the same directory, then:
python main.py -> regular prediction model through the terminal
python app.py -> flask generated chatbot and prediction form

You will see:
Cross-validation R² scores (leak check)
Test set metrics (R², MSE, MAE)
Interactive CLI for SOH prediction

**Troubleshooting**  
ModuleNotFoundError → Re-activate your venv and reinstall dependencies.
Feather read error → Ensure PulseBat.feather is in the same folder as main.py.
Wrong interpreter → Ctrl+Shift+P → “Python: Select Interpreter” → pick .venv.

**What the Script Does:**  
Load Data (PulseBat.feather
Sort by SOC (ascending) to keep train/test consistent.
Split data (80/20) without shuffling.
Outlier Removal on train set using MAD (robust).
Feature Scaling + Polynomial Features (degree=2).
Cross-Validation (5-fold) to check for leakage.
Model Training and evaluation.
Interactive User Prediction for real-time SOH estimation.

How To Use Input Feature:

1. When asked to enter SOH threshold enter a number (0.8 for 80% etc), if no value given it will automatically default to 0.8
2. Next input the 21 voltage readings you want to test (can paste all 21 values on 1 line seperated by space or tab)
3. Next enter the given SOC and SOE valuer associated with the voltage readings
4. Output will show you the predicted SOH of your battery and if the status is healthy or not
