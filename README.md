<div align="center">
📊 Customer Lead Conversion Prediction
From customer enquiries to data-informed lead insights
An end-to-end Python machine learning portfolio project for estimating lead conversion likelihood, comparing classification models, and exploring predictions through an interactive Streamlit dashboard.
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-Not%20specified-lightgrey)
Explore the dashboard · View model results · Run locally
</div>
> [!IMPORTANT]
> **Portfolio demonstration:** All 2,000 training records are **synthetically generated**. The model has **not** been validated using actual customer or real-estate CRM data. Its predictions are illustrative and should **not** guide real sales decisions.
✨ Highlights
End-to-end ML workflow: data generation, exploratory analysis, preprocessing, model training, and evaluation.
Model comparison: Logistic Regression versus Random Forest, including stratified 5-fold cross-validation.
Interactive dashboard: enter a customer's details, view a predicted conversion class, and inspect the model's estimated probability.
Reproducible scripts: regenerate the dataset and retrain the saved model locally.
Transparent evaluation: includes performance metrics and explicit limitations.
🖥️ Dashboard preview
1. Lead information and model overview
![Streamlit dashboard showing the project title, model information, and customer input form](screenshots/dashboard-overview.png)
2. Prediction results and submitted customer details
![Streamlit dashboard showing an unlikely-to-convert prediction, a 36.19 percent estimated conversion probability, and customer input summary](screenshots/prediction-results.png)
Example shown: a 25-year-old customer in Chennai, interested in an apartment with a ₹1,00,00,000 budget, sourced through the website, with 3 calls, 4 follow-ups, no site visit, and a 6-hour response time. The saved model estimated 36.19% conversion probability and classified the lead as unlikely to convert at its default 0.50 threshold. This is an illustrative prediction, not a calibrated real-world probability.
🧩 Problem statement
Sales teams often manage many enquiries and need a consistent way to review lead characteristics. This project demonstrates how a classification model could use structured CRM-style fields to estimate whether a lead may convert.
Input features: age, location, budget, property type, lead source, calls, follow-ups, site visit, and response time.
Target: `converted` (`1` = converted; `0` = not converted).
> **Prediction-time caveat:** Calls, follow-ups, and site visits may happen *after* a lead is created. The current model is best understood as a demonstration using a snapshot of lead activity, **not** as a validated new-lead scoring system. Any real deployment would need a clearly defined prediction time and leakage-safe features.
🛠️ Technology stack
Area	Tools
Programming	Python
Data processing	Pandas, NumPy
Visualization / EDA	Matplotlib, Seaborn
Machine learning	Scikit-learn
Model serialization	Joblib
Web interface	Streamlit
Version control	Git, GitHub
📈 Model evaluation
Two classification algorithms were compared on the 2,000-row synthetic dataset. A stratified 80/20 split provided 1,600 training and 400 test records.
Stratified 5-fold cross-validation (training portion)
Model	Mean ROC-AUC	Fold standard deviation
Logistic Regression	0.6862	0.0237
Random Forest	0.6637	0.0162
Held-out split metrics (selected Logistic Regression model)
Metric	Score
Accuracy	71.25%
Precision	65.38%
Recall	36.69%
F1-score	0.4700
ROC-AUC	0.6788
Average precision (reported as PR-AUC in the script)	0.5738
Interpretation: Logistic Regression performed better than Random Forest in the reported comparison, but its 36.69% recall means it missed many converting examples at the default classification threshold.
Evaluation limitations: These scores describe synthetic data, not business performance. Earlier experiments inspected the same test split, so it should not be treated as a fully untouched final benchmark. Probabilities have not been calibrated. The synthetic outcome-generation process also favors certain learnable relationships.
🚀 Getting started
Prerequisites
Python 3 installed
Git (optional, for cloning)
1. Clone the repository
```bash
git clone https://github.com/YOGESH-PANDIYAN/Customer-Lead-Conversion-Prediction.git
cd Customer-Lead-Conversion-Prediction
```
2. Create and activate a virtual environment
Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
python -m pip install -r requirements.txt
```
4. Prepare the dataset and saved model (if needed)
Run these commands from the repository root. They regenerate the synthetic data and retrain the model:
```bash
python src/generate_dataset.py
python src/compare_models.py
```
5. Launch the dashboard
```bash
python -m streamlit run app.py
```
Open http://localhost:8501 in your browser.
6. Reproduce the evaluation
```bash
python src/evaluate_models.py
```
This evaluation script compares models using cross-validation and reports metrics on the held-out split. It does not replace the model used by the dashboard.
📁 Key project files
```text
Customer_Lead_Conversion_Prediction/
├── app.py                       # Streamlit prediction interface
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore
├── screenshots/
│   ├── dashboard-overview.png
│   └── prediction-results.png
├── data/
│   ├── leads.csv                # Initial sample dataset
│   └── leads_large.csv          # Generated 2,000-record dataset
├── models/
│   ├── best_lead_model.pkl      # Saved selected model
│   └── model_comparison.csv     # Model comparison output
├── src/
│   ├── generate_dataset.py
│   ├── compare_models.py
│   ├── evaluate_models.py
│   └── predict.py
└── visualizations/             # Exploratory analysis outputs
```
Additional scripts from the development process may also be present.
🔭 Future improvements
Train and evaluate using authorized, privacy-protected real CRM data.
Define separate prediction points for newly created leads and in-progress leads.
Use a genuinely independent final test set and tune classification thresholds on validation data.
Calibrate probabilities and explain predictions responsibly.
Add CSV batch scoring, input validation, and automated tests.
Deploy a publicly accessible demo using non-sensitive synthetic data.
👤 Author
Yogesh Pandiyan  
GitHub: @YOGESH-PANDIYAN
License
No open-source license has been specified yet. All rights are reserved by default. If you want others to reuse or modify the code, choose and add an appropriate license.
