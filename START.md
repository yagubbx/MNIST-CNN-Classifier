# Start

Use Python 3.11. Open this folder in VS Code, then open a PowerShell terminal. Run each command after the previous one finishes.

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip check
```

## 2. Run the experiment

```powershell
# Train CNN and MLP; save weights, test results and plots.
.\.venv\Scripts\python train.py

# Reload the CNN and test all seven handwriting images.
.\.venv\Scripts\python external_test.py

# Plot the learned filters (bonus).
.\.venv\Scripts\python show_filters.py

# Predict one image and save its preprocessing steps.
.\.venv\Scripts\python predict.py custom_images/picture_3.png --stages artifacts/preprocessing
```

Training needs internet for the first MNIST download. Saved weights are included, so skip `train.py` if you only want to test images. New runs overwrite results in `artifacts/`; update REPORT.md if the values change.

## 3. Optional upload page

```powershell
.\.venv\Scripts\python app.py
```

Open http://127.0.0.1:5000, select an image and submit it. Keep the terminal open; press Ctrl+C to stop.

To test the API, use a second terminal in the same folder while the server runs:

```powershell
curl.exe -X POST -F "image=@custom_images/picture_3.png" http://127.0.0.1:5000/predict
```

Upload the project files to GitHub, including `artifacts/` and `custom_images/`. Leave out `.venv/`, `data/` and `__pycache__/`.
