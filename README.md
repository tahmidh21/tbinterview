# TB Risk Intelligence

A TB risk screening/assessment prototype developed for the TB Hotspot Research Project. It utilizes a full-stack architecture to accept patient symptom inputs, process them through a Machine Learning model (RandomForestClassifier), and present an AI-assisted healthcare risk classification via a responsive dashboard. 

**Disclaimer:** Research Prototype — This system is intended for research and demonstration purposes only and is not a validated clinical diagnostic tool. Predictions should not be used for real medical decisions. The training dataset does not contain ground-truth clinical diagnoses, and the risk labels are synthetically derived from symptom counts.

## Architecture

```text
[ CSV Dataset ]
      |
      v
[ train_model.py ] ----> tb_model.pkl, features.json, feature_importance.json
                               |
                               v
[ FastAPI Backend (main.py) ] <---> [ SQLite Database (predictions.db) ]
                               |
                        (REST Endpoints)
                               |
                               v
[ Frontend Dashboard (index.html) ]
```

## Setup & Run

Follow these steps to set up and run the application locally:

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the ML model**
   ```bash
   python model/train_model.py
   ```

3. **Start the FastAPI backend**
   ```bash
   python -m uvicorn backend.main:app --reload --port 8000
   ```

4. **Serve the frontend** (in a separate terminal)
   ```bash
   python -m http.server 3000
   ```
   *Navigate to `http://localhost:3000/frontend/index.html` in your browser.*

## API Reference

- **`GET /`**
  Returns a simple health check status.
- **`GET /history`**
  Returns the 20 most recent prediction assessments logged in the database.
- **`GET /feature-importance`**
  Returns the machine learning model's feature importance scores, sorted descending.
- **`POST /predict`**
  Receives a patient symptom profile and returns a risk prediction.

**Example `/predict` Request:**
```json
{
  "gender": 0,
  "fever_for_two_weeks": 0,
  "coughing_blood": 0,
  "sputum_mixed_with_blood": 0,
  "night_sweats": 0,
  "chest_pain": 0,
  "back_pain_in_certain_parts": 0,
  "shortness_of_breath": 0,
  "weight_loss": 0,
  "body_feels_tired": 0,
  "lumps_that_appear_around_the_armpits_and_neck": 0,
  "cough_and_phlegm_continuously_for_two_weeks_to_four_weeks": 0,
  "swollen_lymph_nodes": 0,
  "loss_of_appetite": 0
}
```

**Example `/predict` Response:**
```json
{
  "risk_label": 0,
  "risk_name": "Low",
  "confidence": 0.965
}
```

## Model Details

The core model is a **RandomForestClassifier** (`n_estimators=200`). 

Because the source dataset lacked a ground-truth diagnosis column, the target `risk_label` was derived programmatically based on total symptom count:
- 0 to 4 symptoms = **Low** (0)
- 5 to 8 symptoms = **Medium** (1)
- 9+ symptoms = **High** (2)

### Training Evaluation
- **Class Imbalance:** 149 Low (15%), 741 Medium (74%), 110 High (11%).
- **Cross-validation accuracy:** `0.8370 (+/- 0.0121)`
- **Baseline Comparison:** `Baseline LogisticRegression accuracy: 1.0000 vs RandomForest accuracy: 0.8500`

**Confusion Matrix (Test Set):**
```text
Predicted ->      Low   Medium     High
---------------------------------------
       Low  |      15       15        0
    Medium  |       0      148        0
      High  |       0       15        7
```

### Limitations
- **Synthetic/Imbalanced Data:** The dataset is heavily skewed toward Medium risk cases. As seen in the confusion matrix, the model achieves perfect recall (1.00) on Medium cases but struggles with the extremes, only identifying 50% of Low cases and 32% of High cases (often classifying them as Medium).
- **Overfitting Risk:** The LogisticRegression baseline achieving 1.0000 accuracy suggests the programmatic symptom-count derivation logic is highly linear, making the problem trivially separable.
- **Not for Medical Use:** This model is entirely synthetic and must not be deployed in a clinical setting.

## Testing

To run the backend test suite, use pytest:
```bash
pytest backend/test_main.py -v
```

## Project Structure

```text
tb-risk-predictor/
├── backend/
│   ├── main.py
│   ├── test_main.py
│   └── predictions.db
├── frontend/
│   └── index.html
├── model/
│   ├── Tb disease symptoms.csv
│   ├── train_model.py
│   ├── tb_model.pkl
│   ├── features.json
│   ├── feature_importance.json
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```
