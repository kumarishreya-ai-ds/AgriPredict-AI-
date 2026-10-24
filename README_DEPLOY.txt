AgriPredict AI - Streamlit Community Cloud deployment

1. Replace the repository's app.py with the supplied app.py.
2. Add requirements.txt to the repository root (same folder as app.py).
3. If you have a real dataset, add its CSV to the repository root. The app looks for a CSV with these columns:
   Crop, Crop_Year, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide, Yield
   Common aliases such as year, rainfall, fertiliser, pesticides, and crop_yield are normalized.
4. If you have a trained model, add its .pkl, .pickle, or .joblib file to the repository root.
   Important: only load trusted model files; pickle/joblib files can execute code when loaded.
5. Commit and push changes to GitHub.
6. In Streamlit Community Cloud, open Manage app > Settings (or App settings) and confirm:
   Main file path = app.py
7. Reboot the app. If it still shows old code, use Reboot app and confirm the correct GitHub repository/branch.

The dashboard can start without a dataset or trained model by using clearly labelled synthetic preview data.
Without a trained model, predictions remain disabled. Do not present synthetic preview data as real agricultural results.
