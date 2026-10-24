# 🌾 AgriPredict AI — Crop Yield Prediction Dashboard

An interactive Streamlit dashboard for exploring crop-yield data and generating predictions with a trained machine-learning model.

<p align="center">
  <strong>Python · Streamlit · Pandas · NumPy · Scikit-learn</strong>
</p>

## 🚀 Live Demo

**[Open AgriPredict AI](https://agripredict-ai-epvzhidqy38s4lsuynzu5r.streamlit.app/)**

> If the app shows a model-file error, the model artifact has not yet been added to the deployment or the app's model path needs updating. See [Deployment notes](#-deployment-notes).

## ✨ Features

- Interactive crop-yield prediction interface
- Dataset exploration and summary statistics
- Visual summaries for crop and season data
- Clear validation messages for missing model/data files
- Streamlit-based responsive web interface

## 🧰 Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Joblib for loading the trained model
- Plotly/Matplotlib (if used by the app)

## 📁 Project Structure

```text
AgriPredict-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
└── crop_yield.csv                 # optional dataset; add if redistribution is allowed
```

The trained model (`agripredict_model.pkl`) is not included in this starter package because it is large (~134 MB). See the deployment notes below before deciding how to host it.

## ▶️ Run Locally

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/kumarishreya-ai-ds/AgriPredict-AI-.git
cd AgriPredict-AI-
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\\Scripts\\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deployment notes

Streamlit Community Cloud installs files committed to the GitHub repository; it cannot see files stored only in Google Colab (`/content`).

Your current `app.py` expects these paths:

```text
/content/agripredict_model.pkl
/content/crop_yield.csv
```

For deployment, change the paths to project-relative paths, for example:

```python
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "agripredict_model.pkl"
DATA_PATH = BASE_DIR / "crop_yield.csv"
```

**Important:** the model is about 134 MB. GitHub's regular file limit is 100 MB, so do not upload it as a normal file. Use Git LFS or external model storage, and update `app.py` to download/load it safely at startup. Also confirm your model file can be legally shared publicly before publishing it.

## 🛡️ Data and model

- Do not commit passwords, API keys, personal data, or private datasets.
- Add a data/model license or source attribution if required.
- Predictions are estimates and should not be treated as guaranteed agricultural outcomes.

## 👩‍💻 Author

**Shreya Kumari**  
GitHub: [@kumarishreya-ai-ds](https://github.com/kumarishreya-ai-ds)

## 📄 License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

## 🚀 Live Demo

[Open AgriPredict AI Dashboard](https://agripredict-ai-epvzhidqy38s4lsuynzu5r.streamlit.app/)
