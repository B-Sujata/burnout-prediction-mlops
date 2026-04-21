# 🧠 AI-Based Student Burnout Prediction System

> **Production-level end-to-end ML + MLOps project using a real Kaggle student stress dataset.**  
> Predicts burnout risk (0–100) across psychological, physical, academic, and social dimensions.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Real Dataset](#-real-dataset)
- [Architecture](#-architecture)
- [Feature Engineering](#-feature-engineering)
- [Burnout Score Formula](#-burnout-score-formula)
- [Models & Results](#-models--results)
- [MLOps Stack](#-mlops-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Prediction Examples](#-prediction-examples)
- [Visualizations](#-visualizations)

---

## 🎯 Project Overview

This system trains regression models on **1,100 real student records** to predict burnout risk on a continuous 0–100 scale, enabling early intervention. A key design choice: the original `stress_level` label (0/1/2) is used only for **validation** — the model learns to predict a rich continuous score rather than a flat 3-class label.

| Dimension | Details |
|---|---|
| Dataset | Real Kaggle dataset — 1,100 students, 20 features, 0 missing values |
| Target | Engineered `burnout_score` (0–100, continuous regression) |
| Best model | Linear Regression (R²=1.000) · Gradient Boosting (R²=0.996) |
| Validation | Burnout score means: Low=16.4 · Moderate=42.8 · High=80.7 |
| MLOps | DVC + MLflow + GitHub Actions CI/CD |

---

## 📊 Real Dataset

**Source:** [Kaggle — Student Stress Factors: A Comprehensive Analysis](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis)  
**File:** `StressLevelDataset.csv`

| Category | Features | Scale |
|---|---|---|
| **Psychological** | anxiety_level, self_esteem, mental_health_history, depression | 0–21, 0–30, 0–1, 0–27 |
| **Physical** | headache, blood_pressure, breathing_problem | 0–5, 1–3, 0–5 |
| **Environmental** | noise_level, living_conditions, safety, basic_needs | 0–5 each |
| **Academic** | academic_performance, study_load, teacher_student_relationship, future_career_concerns | 0–5 each |
| **Social** | social_support, peer_pressure, extracurricular_activities, bullying | 0–3, 0–5, 0–5, 0–5 |
| **Label** | stress_level (0=Low, 1=Moderate, 2=High) — used for validation only | 0–2 |

Dataset is **clean out of the box**: zero missing values, zero duplicates, balanced classes (373/358/369).

---

## 🏗️ Architecture

```
StressLevelDataset.csv
        │
        ▼
┌─────────────────┐
│  Stage 1        │  Validate + log dataset
│  Data Ingestion │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 2        │  Handle edge cases, standardise columns
│  Preprocessing  │
└────────┬────────┘
         │
         ▼
┌──────────────────────┐
│  Stage 3 & 4         │  5 composite indices → burnout_score (0–100)
│  Feature Engineering │  + 5 interaction features
└────────┬─────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  Stage 5        │────►│  MLflow      │
│  Model Training │     │  Experiment  │
│  LR / RF / GB   │     │  Tracking    │
└────────┬────────┘     └──────────────┘
         │
         ▼
┌─────────────────┐
│  Stage 6        │  MAE / RMSE / R² / CV R²
│  Evaluation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────────┐
│  Stage 9        │     │  5 Plots saved to         │
│  Visualization  │────►│  results/                 │
└────────┬────────┘     │  · Burnout distribution   │
         │              │  · Correlation heatmap    │
         ▼              │  · Feature importance     │
┌─────────────────┐     │  · Model comparison       │
│  Prediction     │     │  · Stress level validation│
│  Pipeline       │     └──────────────────────────┘
└─────────────────┘
```

---

## ⚙️ Feature Engineering

Each of the 20 raw features is normalised to [0, 1] using its known scale, then combined into 5 composite indices. Features where **higher = better** (e.g. self_esteem, sleep_quality) are **inverted** before weighting.

### Composite Indices

| Index | Key Inputs | Direction |
|---|---|---|
| **Emotional Strain** | anxiety + depression + mental_health_history + ¬self_esteem | Higher = worse |
| **Physical Stress** | headache + blood_pressure + breathing_problem | Higher = worse |
| **Academic Pressure** | study_load + future_career_concerns + ¬academic_performance + ¬teacher_relationship | Higher = worse |
| **Social Stress** | peer_pressure + bullying + noise_level + ¬living_conditions + ¬safety + ¬basic_needs | Higher = worse |
| **Recovery Index** | sleep_quality + social_support + extracurricular_activities | Higher = protective ↓ |

### Interaction Features

| Feature | Formula |
|---|---|
| `stress_sleep_ratio` | `anxiety_level / (sleep_quality + ε)` |
| `academic_burden` | `(study_load + future_career_concerns) / 10` |
| `isolation_index` | `poor_social_support × peer_pressure` |
| `environment_quality` | `(living_conditions + safety + basic_needs) / 15` |
| `mental_load` | `(norm(anxiety) + norm(depression)) / 2` |

---

## 🔢 Burnout Score Formula

```
burnout_raw =  0.30 × emotional_strain
             + 0.15 × physical_stress
             + 0.25 × academic_pressure
             + 0.20 × social_stress
             − 0.10 × recovery_index     ← protective factor

burnout_score = (raw − min) / (max − min) × 100   [Min-Max to 0–100]
```

### Risk Thresholds

| Score | Risk | Recommendation |
|---|---|---|
| 0 – 33 | 🟢 Low | Maintain current lifestyle |
| 34 – 66 | 🟡 Moderate | Improve sleep, activity, seek support |
| 67 – 100 | 🔴 High | Immediate counselling recommended |

---

## 🤖 Models & Results

All models are wrapped in `sklearn.Pipeline` with `StandardScaler`.

| Model | MAE | RMSE | R² | CV R² |
|---|---|---|---|---|
| **Linear Regression** | 0.003 | 0.003 | **1.000** | 1.000 |
| **Gradient Boosting** | 1.107 | 1.980 | **0.996** | 0.995 |
| **Random Forest** | 1.596 | 2.615 | **0.992** | 0.992 |

> **Why R²=1.0 for Linear Regression?** The burnout score is a weighted linear combination of features — Linear Regression recovers the exact formula perfectly. Gradient Boosting and Random Forest are the more meaningful benchmarks for non-linear generalisation.

---

## 🔧 MLOps Stack

| Component | Tool | Purpose |
|---|---|---|
| Data versioning | **DVC** | Track CSV, processed data, model artefacts |
| Experiment tracking | **MLflow** | Log params, metrics, model per run |
| Pipeline runner | `run_pipeline.py` / `dvc repro` | Reproducible end-to-end execution |
| CI/CD | **GitHub Actions** | 4-job pipeline on every push |
| Config management | **YAML** (`configs/config.yaml`) | All weights, paths, hyperparams |
| Logging | Custom `logger.py` | File + console logs in `logs/project.log` |

---

## 📁 Project Structure

```
burnout-prediction-mlops/
│
├── .github/workflows/ci.yml         # 4-job GitHub Actions CI/CD
├── configs/
│   ├── config.yaml                  # All pipeline configuration
│   └── advisor_config.yaml          # LLM advisor — provider, model, weights
├── params.yaml                      # DVC pipeline parameters
├── dvc.yaml                         # 6-stage DVC pipeline
├── run_pipeline.py                  # Master runner (all stages)
├── requirements.txt
│
├── data/
│   ├── raw/student_data.csv         # Real Kaggle dataset (place here)
│   └── processed/
│       ├── processed_data.csv       # After Stage 2
│       └── engineered_data.csv      # After Stage 3+4 (32 columns)
│
├── src/
│   ├── data/
│   │   ├── ingest_data.py           # Stage 1 — validate & log real CSV
│   │   ├── preprocess_data.py       # Stage 2 — clean, standardise
│   │   └── feature_engineering.py  # Stage 3+4 — 5 composites + burnout score
│   ├── models/
│   │   ├── train_model.py           # Stage 5 — LR / RF / GB + MLflow
│   │   ├── evaluate_model.py        # Stage 6 — MAE / RMSE / R²
│   │   └── predict_model.py         # Stage 8 — predict + optional LLM advice
│   ├── advisor/                     # ← GenAI / LangChain layer
│   │   ├── burnout_advisor.py       # Driver detection + LangChain chain
│   │   └── prompt_templates.py     # Structured ChatPromptTemplate
│   ├── visualization/
│   │   └── visualize_results.py     # Stage 9 — 5 plots
│   └── utils/
│       ├── logger.py
│       └── config_loader.py
│
├── models/
│   ├── burnout_model.pkl            # Best model (auto-selected)
│   ├── linearregression_model.pkl
│   ├── randomforest_model.pkl
│   ├── gradientboosting_model.pkl
│   └── feature_names.json           # Saved feature order for prediction
│
├── results/
│   ├── burnout_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── stress_level_validation.png  # ← NEW: validates engineered score
│   ├── metrics_summary.json
│   └── evaluation_report.json
│
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py             # 20+ pipeline unit tests
│   └── test_advisor.py              # 20 advisor unit tests (no API key needed)
│
└── logs/project.log                 # Auto-generated pipeline log
```

---

## 🤖 GenAI — LLM Burnout Advisor (LangChain)

After the ML model predicts a burnout score, the **LLM Burnout Advisor** generates a personalised, empathetic, section-based action plan — powered by LangChain and your choice of **Anthropic Claude** or **OpenAI GPT-4o**.

### How it works

```
predict_burnout(features, use_advisor=True)
        │
        ├── 1. ML model  →  burnout_score + 5 composite indices
        │
        ├── 2. identify_stress_drivers()
        │        Normalises all 20 features to [0,1]
        │        Inverts protective features (sleep, self_esteem, social_support …)
        │        Ranks by stress contribution → Top 3 drivers surfaced
        │
        ├── 3. LangChain chain: ChatPromptTemplate | LLM
        │        System prompt: expert counsellor persona
        │        Human prompt:  score + profile + drivers + composite breakdown
        │
        └── 4. Structured 5-section advice response:
               ① Understanding Your Situation
               ② Your Top Stress Drivers
               ③ Immediate Actions (This Week)
               ④ Short-Term Goals (Next Month)
               ⑤ When to Seek Help
```

### Setup

**Step 1 — Choose your provider** in `configs/advisor_config.yaml`:
```yaml
advisor:
  provider: "anthropic"       # or "openai"
  anthropic_model: "claude-sonnet-4-6"
  openai_model: "gpt-4o"
  temperature: 0.7
  max_tokens: 1024
  top_stress_drivers: 3
```

**Step 2 — Set your API key:**
```bash
# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI (GPT-4o)
export OPENAI_API_KEY="sk-..."
```

**Step 3 — Run with the `--advise` flag:**
```bash
# Interactive CLI
python src/models/predict_model.py --interactive --advise

# JSON input
python src/models/predict_model.py --json-input '{
  "anxiety_level":15, "self_esteem":8, "mental_health_history":1,
  "depression":14, "headache":3, "blood_pressure":2, "sleep_quality":2,
  "breathing_problem":2, "noise_level":3, "living_conditions":2,
  "safety":2, "basic_needs":2, "academic_performance":2, "study_load":4,
  "teacher_student_relationship":2, "future_career_concerns":4,
  "social_support":1, "peer_pressure":4, "extracurricular_activities":1,
  "bullying":3
}' --advise
```

**Docker with advisor:**
```bash
docker run --rm \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e PREDICT_INPUT='{...}' \
  -e USE_ADVISOR=true \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  burnout-prediction:latest predict-json
```

### Sample output

```
🟡 BURNOUT SCORE : 61.4 / 100
   RISK LEVEL    : Moderate

─────────────────────────────────────────────────────
  🤖 PERSONALISED AI ADVICE
─────────────────────────────────────────────────────

**Understanding Your Situation**
Your assessment reveals a student under considerable academic and emotional
pressure, with sleep and social support acting as your two most critical
unmet recovery needs. While you haven't reached crisis point, the trajectory
is concerning if left unaddressed.

**Your Top Stress Drivers**
1. Sleep Quality (score: 2/5) — Chronic poor sleep amplifies every other
   stressor. At this level, your cognitive performance, emotional regulation,
   and immune function are all compromised.
2. Future Career Concerns (score: 4/5) — High career anxiety is creating a
   background hum of worry that makes it hard to be present in your studies.
3. Peer Pressure (score: 4/5) — Feeling pressure to keep up socially or
   academically is adding a layer of stress that's difficult to switch off.

**Immediate Actions (This Week)**
1. Set a non-negotiable 10:30pm phone-down rule for 7 days — track how
   your mood shifts.
2. Book one appointment with your university counsellor this week, even
   just to talk.
3. Write down your top 3 career fears and identify one small action per fear.

**Short-Term Goals (Next Month)**
1. Build a consistent sleep schedule (same wake time daily, including weekends).
2. Join one low-pressure social group or study circle to rebuild positive
   peer connection.
3. Speak to a careers advisor about your post-graduation concerns — external
   perspective reduces catastrophising.

**When to Seek Help**
Your score of 61 is in the Moderate range. Professional support is
recommended but not urgent. If your score rises, or you experience
persistent low mood, please speak with a mental health professional promptly.

  📊 Top Stress Drivers Identified:
     • Sleep Quality: 2  (contribution: 60%)
     • Future Career Concerns: 4  (contribution: 80%)
     • Peer Pressure: 4  (contribution: 80%)
```

### Graceful fallback

If no API key is set or the LLM call fails for any reason, the advisor automatically falls back to a static message — **the prediction pipeline never breaks**.

### Files added

| File | Purpose |
|---|---|
| `src/advisor/burnout_advisor.py` | Main advisor class — driver detection, LangChain chain, fallback |
| `src/advisor/prompt_templates.py` | Structured `ChatPromptTemplate` (system + human messages) |
| `configs/advisor_config.yaml` | Provider, model, temperature, feature weights for driver ranking |
| `tests/test_advisor.py` | 20 unit tests — all run without an API key |

---

## 🐳 Docker

The project ships a **multi-stage Dockerfile** and a **docker-compose.yml** with named services for every stage of the pipeline.

### Build the image

```bash
docker build -t burnout-prediction:latest .
```

### Run with Docker Compose

```bash
# Run the full pipeline (all 6 stages)
docker-compose --profile pipeline up

# Run a single stage
docker-compose --profile train up train

# Launch MLflow UI → http://localhost:5000
docker-compose --profile mlflow up mlflow

# Interactive prediction CLI
docker-compose --profile predict run --rm predict

# JSON prediction (non-interactive)
PREDICT_INPUT='{"anxiety_level":18,"self_esteem":5,...}' \
  docker-compose --profile predict-json run --rm predict-json

# Run test suite inside container
docker-compose --profile test run --rm test
```

### Run with plain Docker

```bash
# Full pipeline (mounts host data/models/results so artefacts persist)
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  burnout-prediction:latest pipeline

# JSON prediction
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e PREDICT_INPUT='{"anxiety_level":18,"self_esteem":5,"mental_health_history":1,
    "depression":22,"headache":4,"blood_pressure":3,"sleep_quality":1,
    "breathing_problem":4,"noise_level":4,"living_conditions":1,"safety":1,
    "basic_needs":1,"academic_performance":1,"study_load":5,
    "teacher_student_relationship":1,"future_career_concerns":5,
    "social_support":0,"peer_pressure":5,"extracurricular_activities":0,"bullying":5}' \
  burnout-prediction:latest predict-json

# Drop into a shell for debugging
docker run --rm -it \
  -v $(pwd)/data:/app/data \
  burnout-prediction:latest bash
```

### Available container commands

| Command | Description |
|---|---|
| `pipeline` | Run all 6 pipeline stages (default) |
| `ingest` | Stage 1 — validate & log dataset |
| `preprocess` | Stage 2 — clean & standardise |
| `engineer` | Stage 3+4 — build features + burnout score |
| `train` | Stage 5 — train all models + MLflow tracking |
| `evaluate` | Stage 6 — MAE / RMSE / R² report |
| `visualize` | Stage 9 — generate all 5 plots |
| `predict` | Interactive CLI prediction |
| `predict-json` | Non-interactive JSON prediction (set `PREDICT_INPUT`) |
| `mlflow` | Launch MLflow UI on port 5000 |
| `test` | Run pytest test suite |
| `bash` | Shell access for debugging |

### CI/CD Docker jobs

The GitHub Actions pipeline now has **5 jobs**:

```
lint → unit-tests ─┐
                   ├─► ml-pipeline → docker-build-test → docker-push (main only)
                   │       ↓
                   │  Upload artefacts
                   └── (parallel)
```

On every push to `main`, the image is built, tested end-to-end, and pushed to **GitHub Container Registry (GHCR)**:
```
ghcr.io/<your-username>/<repo>/burnout-prediction:latest
ghcr.io/<your-username>/<repo>/burnout-prediction:sha-<commit>
```

---

## 🚀 How to Run

### 1. Setup

```bash
git clone https://github.com/your-username/burnout-prediction-mlops.git
cd burnout-prediction-mlops
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add the Dataset

Download `StressLevelDataset.csv` from [Kaggle](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis) and place it at:
```
data/raw/student_data.csv
```

### 3. Run the Full Pipeline

```bash
# Option A — Python runner (recommended)
python run_pipeline.py

# Option B — DVC (reproducible)
dvc init && dvc repro

# Option C — Run specific stages only
python run_pipeline.py --stages preprocess engineer train evaluate
```

### 4. View MLflow UI

```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Run Unit Tests

```bash
pytest tests/ -v --tb=short
```

---

## 🔮 Prediction Examples

**Interactive CLI:**
```bash
python src/models/predict_model.py --interactive
```

**JSON input:**
```bash
# High-risk student
python src/models/predict_model.py --json-input '{
  "anxiety_level":18, "self_esteem":5, "mental_health_history":1, "depression":22,
  "headache":4, "blood_pressure":3, "sleep_quality":1, "breathing_problem":4,
  "noise_level":4, "living_conditions":1, "safety":1, "basic_needs":1,
  "academic_performance":1, "study_load":5, "teacher_student_relationship":1,
  "future_career_concerns":5, "social_support":0, "peer_pressure":5,
  "extracurricular_activities":0, "bullying":5
}'
# → 🔴 BURNOUT SCORE: 100.0 / 100  |  RISK LEVEL: High

# Low-risk student
python src/models/predict_model.py --json-input '{
  "anxiety_level":2, "self_esteem":28, "mental_health_history":0, "depression":2,
  "headache":0, "blood_pressure":1, "sleep_quality":5, "breathing_problem":0,
  "noise_level":1, "living_conditions":5, "safety":5, "basic_needs":5,
  "academic_performance":5, "study_load":1, "teacher_student_relationship":5,
  "future_career_concerns":0, "social_support":3, "peer_pressure":0,
  "extracurricular_activities":5, "bullying":0
}'
# → 🟢 BURNOUT SCORE: 0.0 / 100  |  RISK LEVEL: Low
```

---

## 📈 Visualizations

After running the pipeline, 5 plots are saved to `results/`:

| Plot | Description |
|---|---|
| `burnout_distribution.png` | Score histogram + risk category pie chart |
| `correlation_heatmap.png` | Feature correlation matrix (all 21 key columns) |
| `feature_importance.png` | Top 18 features — Random Forest & Gradient Boosting |
| `model_comparison.png` | MAE / RMSE / R² side-by-side bar charts |
| `stress_level_validation.png` | **Box plots** confirming engineered score aligns with real labels |

---

## 🌍 Impact

- **Student Wellbeing** — Early identification of at-risk students for counsellor outreach
- **Institutional Analytics** — Identify which factors most contribute to burnout across cohorts
- **Research** — Extensible pipeline for adding new features or datasets
- **Portfolio** — Demonstrates production ML engineering with real-world data

---

*B.Tech IT Final Year Project — AI-Based Student Burnout Prediction with MLOps*  
*Dataset: Kaggle StressLevelDataset · Models: Linear Regression, Random Forest, Gradient Boosting*
