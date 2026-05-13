# Reddit Personality Fingerprinter

An end-to-end NLP pipeline that estimates Big-Five personality traits from a Reddit user's public comment history, deployed as an interactive Streamlit web app.

---

## What It Does

1. Fetches a Reddit user's comment history via the Reddit API (PRAW)
2. Preprocesses and cleans text with spaCy
3. Extracts 50+ psycholinguistic features (LIWC-style: pronoun usage, hedging, sentiment, complexity, emotion words, etc.)
4. Predicts Big-Five scores (OCEAN) using trained LightGBM models
5. Displays a radar chart, trait scores, and a downloadable PDF report

---

## Big Five Traits Predicted

| Trait | Abbreviation | What it captures |
|---|---|---|
| Openness | O | Curiosity, creativity, broad interests |
| Conscientiousness | C | Discipline, organisation, goal-orientation |
| Extraversion | E | Sociability, assertiveness, talkativeness |
| Agreeableness | A | Cooperativeness, empathy, warmth |
| Neuroticism | N | Emotional instability, anxiety, moodiness |

---

## Project Structure

```
reddit-personality-fingerprinter/
├── app.py                        # Streamlit web app entry point
├── config.py                     # API keys, thresholds, model paths
├── requirements.txt
├── README.md
│
├── src/
│   ├── scraper/
│   │   ├── __init__.py
│   │   └── reddit_scraper.py     # PRAW-based comment fetcher
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── preprocessor.py       # spaCy cleaning pipeline
│   │   ├── psycholinguistic.py   # 50+ feature extractors
│   │   └── feature_pipeline.py   # Orchestrates feature extraction
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py            # LightGBM training + CV
│   │   ├── predictor.py          # Load model + predict
│   │   └── evaluator.py          # Reliability metrics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualiser.py         # Radar chart + bar plots
│       └── report_generator.py   # PDF report via reportlab
│
├── tests/
│   ├── test_scraper.py
│   ├── test_features.py
│   └── test_predictor.py
│
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory analysis of feature distributions
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
└── outputs/                      # Generated reports land here
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/yourusername/reddit-personality-fingerprinter.git
cd reddit-personality-fingerprinter
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure credentials

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=PersonalityFingerprinter/1.0
```

Get Reddit API credentials at: https://www.reddit.com/prefs/apps

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Or use the CLI

```bash
python -m src.models.predictor --username spez --min_comments 100
```

---

## Training Your Own Models

The repo ships with pre-trained placeholder models. To train on your own labelled data:

```bash
python -m src.models.trainer \
    --data data/labelled_users.csv \
    --output models/ \
    --cv_folds 5
```

Expected CSV format:
```
username, O_score, C_score, E_score, A_score, N_score
```

---

## Feature Overview

See `src/features/psycholinguistic.py` for full implementation. Key feature groups:

| Group | Examples |
|---|---|
| Pronoun usage | I-rate, we-rate, you-rate, they-rate |
| Hedging | maybe, perhaps, possibly, I think |
| Certainty | definitely, always, never, absolutely |
| Sentiment | VADER compound, positive ratio, negative ratio |
| Complexity | avg sentence length, type-token ratio, Flesch score |
| Social | question marks, exclamation marks, mentions |
| Emotion | NRC anger, fear, joy, trust word counts |
| Time orientation | past tense ratio, future tense ratio |
| Cognitive | cause words, insight words, discrepancy words |

---

## Limitations & Ethics

- Scores are *probabilistic estimates*, not clinical diagnoses
- Accuracy depends on comment volume (< 50 comments = low reliability)
- Reddit comments may not reflect full personality
- Never use for hiring, screening, or profiling without consent
- All predictions are for **public** comment histories only

---

## References

- Harrison, Thurgood, Boivie & Pfarrer (2019). *Measuring CEO Personality Using Machine Learning*. Strategic Management Journal.
- Mairesse et al. (2007). *Using Linguistic Cues for the Automatic Recognition of Personality in Conversation*.
- Pennebaker, J.W. et al. LIWC framework.

---

## License

MIT
