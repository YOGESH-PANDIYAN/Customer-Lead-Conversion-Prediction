
# Customer Lead Conversion Prediction

An end-to-end Machine Learning project that predicts
the likelihood of customer lead conversion using
historical-style CRM features.

## Project Overview

This project demonstrates how Machine Learning can
support lead prioritization in a real estate sales
workflow.

It includes data analysis, preprocessing, model
training, model comparison, and an interactive
Streamlit dashboard.

**Important:** The dataset is synthetic. Model
performance does not establish real-world predictive
accuracy.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset

The project uses 2,000 synthetically generated
customer lead records.

Features include:

- Age
- Location
- Budget
- Property type
- Lead source
- Number of calls
- Follow-ups
- Site visit
- Response time

Target variable:

- converted: 1 = Converted, 0 = Not converted

## Machine Learning Models

Two classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest

### Model Comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 71.25% | 67.00% |
| Precision | 65.38% | 55.93% |
| Recall | 36.69% | 23.74% |
| F1 Score | 47.00% | 33.33% |
| ROC-AUC | 67.88% | 66.11% |

Logistic Regression achieved the highest ROC-AUC
on the evaluated test split.

These are educational results from synthetic data.
The test split was also used for model selection,
so further validation is needed before making
generalization claims.

## Streamlit Dashboard

The application allows users to:

- Enter customer details
- Predict lead conversion
- View estimated conversion probability
- Review submitted customer information

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Customer-Lead-Conversion-Prediction.git
cd Customer-Lead-Conversion-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate the dataset and train the model:

```bash
python src/generate_dataset.py
python src/compare_models.py
```

Run the dashboard:

```bash
python -m streamlit run app.py
```

## Limitations

- Dataset is synthetic, not actual CRM data.
- Prediction probabilities are not calibrated.
- Model selection used test-set performance.
- Some features may not be available when
  a new lead is first created.
- The application is for demonstration purposes,
  not operational sales decisions.

## Future Improvements

- Train using authorized real-world CRM data.
- Implement cross-validation.
- Add model explainability.
- Introduce database integration.
- Develop batch lead prediction.
- Deploy the application.

## Author

Your Name
```

Replace `YOUR_USERNAME` and `Your Name` with your own details when you create your GitHub repository.

---

## Step 10.2 — Generate requirements.txt

This file records the Python libraries needed to run your application.

In the VS Code terminal, make sure your virtual environment is activated.

Run:

```powershell
python -m pip freeze > requirements.txt
```

This creates `requirements.txt` automatically.

Open it and check that it includes the packages required by your project, including `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `streamlit`, and `joblib`.

The file may contain many additional dependencies, which is normal for `pip freeze`. For a cleaner portfolio, we can later replace it with a curated dependency list.

## Step 10.3 — Create .gitignore

Create a new file named `.gitignore` in the project root.

Add:

<CodeBlock language="gitignore" editable>
# Python virtual environment
venv/
.venv/

# Python cache
__pycache__/
*.py[cod]

# Jupyter checkpoints
.ipynb_checkpoints/

# IDE settings
.vscode/
.idea/

# Environment variables and secrets
.env
.env.*
!.env.example
.streamlit/secrets.toml

# Operating system files
.DS_Store
Thumbs.db

# Local logs
*.log
