<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge" alt="Status: Complete">
</p>

<h1 align="center">🤖 Artificial Intelligence Projects</h1>
<h3 align="center">DecodeLabs — From Rules to Learning to Personalisation</h3>

<p align="center">
  A three-part journey through foundational AI — building a deterministic rule-based chatbot,
  training a supervised classifier on real data, and engineering a pattern-alignment
  recommendation engine. Each project is a self-contained, fully-commented Python pipeline
  designed to demonstrate the <b>why</b> behind the math, not just the <b>how</b>.
</p>

---

## 📑 Table of Contents

| # | Project | Track | Core Concept |
|:-:|:--------|:------|:-------------|
| P1 | [Rule-Based AI Chatbot](#-project-1--rule-based-ai-chatbot) | Deterministic Logic | Hard-coded if-else control flow, IPO model, zero hallucination |
| P2 | [Data Classification Using AI](#-project-2--data-classification-using-ai) | Supervised Learning | StandardScaler, train/test split, K-Nearest Neighbours |
| P3 | [AI Recommendation Logic](#-project-3--ai-recommendation-logic) | Pattern Alignment | Jaccard Similarity, TF-IDF weighted Cosine Similarity |

---

## 🗺️ The Learning Arc

```
 P1: DETERMINISTIC          P2: PREDICTIVE              P3: PERSONALISED
 ┌─────────────────┐        ┌─────────────────┐         ┌─────────────────┐
 │  Hard-coded     │   ──▸  │  Learns from    │    ──▸  │  Matches user   │
 │  rules produce  │        │  data patterns  │         │  vectors to     │
 │  exact output   │        │  via KNN        │         │  item vectors   │
 │                 │        │                 │         │                 │
 │  Zero risk.     │        │  Generalises.   │         │  Recommends.    │
 │  Full control.  │        │  Scales.        │         │  Personalises.  │
 └─────────────────┘        └─────────────────┘         └─────────────────┘
     if input == "hi":          μ = mean(feat)              J(A,B) = |A∩B|
         print("hello")         x̂ = (x - μ) / σ              |A∪B|
                                 ŷ = KNN.predict(x̂)        cos(θ) = A·B
                                                            ─────────────
                                                             ‖A‖ × ‖B‖
```

---

## 🚀 Quick Start

### Prerequisites

```bash
python --version    # 3.10 or higher
pip --version       # latest recommended
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-projects.git
cd ai-projects

# Install dependencies
pip install -r requirements.txt
```

### Run Any Project

```bash
# Project 1 — Rule-Based Chatbot (interactive terminal loop)
python logic_engine_chatbot.py

# Project 2 — Data Classification (full ML pipeline)
python data_classifier_pipeline.py

# Project 3 — Recommendation Engine (similarity comparison)
python recommendation_engine.py
```

---

## 📁 Repository Structure

```
ai-projects/
│
├── logic_engine_chatbot.py          # P1: Rule-based chatbot (IPO model)
├── data_classifier_pipeline.py      # P2: KNN classification pipeline
├── recommendation_engine.py         # P3: Jaccard + TF-IDF Cosine engine
│
├── requirements.txt                 # All dependencies
├── README.md                        # This file
│
├── assets/
│   ├── p1_chatbot_demo.gif          # P1 terminal demo
│   ├── p2_confusion_matrix.png      # P2 confusion matrix visual
│   └── p3_similarity_heatmap.png    # P3 cosine similarity heatmap
│
└── docs/
    ├── P1_ARCHITECTURE.md           # P1 detailed design notes
    ├── P2_MATH_DERIVATIONS.md       # P2 full mathematical derivations
    └── P3_TFIDF_EXPLAINER.md        # P3 TF-IDF deep-dive
```

---

## 🤖 Project 1 — Rule-Based AI Chatbot

> **Track:** Deterministic Logic
> **File:** `logic_engine_chatbot.py`
> **Dependencies:** None (stdlib only)

### What It Does

A white-box chatbot that follows the **Input-Process-Output (IPO)** model. Every response is hard-coded — same input always produces the same output. Zero hallucination risk, zero inference, zero model.

### Architecture

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│   INPUT     │ ──▸ │  PROCESS   │ ──▸ │   OUTPUT   │
│ (Phase 1)   │     │ (Phase 2)  │     │ (Phase 3)  │
│             │     │             │     │             │
│ • Read raw  │     │ • Greeting  │     │ • Format   │
│   stdin     │     │   detect    │     │   bordered  │
│ • Lowercase │     │ • Calc cmd  │     │   speech    │
│ • Strip ws  │     │ • Reverse   │     │   bubble    │
│ • Check     │     │   cmd       │     │ • Print to  │
│   exit cmds │     │ • Intent    │     │   stdout    │
│             │     │   catalogue │     │             │
│             │     │ • Fallback  │     │             │
└────────────┘     └────────────┘     └────────────┘
```

### Supported Intents

| Trigger | Response Type |
|:--------|:-------------|
| `hello`, `hi`, `good morning`… | Time-of-day aware greeting |
| `who are you`, `your name`… | Identity (bot description) |
| `who made you`, `your creator`… | DecodeLabs attribution |
| `what can you do`, `help`… | Capability list |
| `what time`, `what date`… | Live system clock |
| `how are you`… | Status response |
| `tell me a joke`… | Hard-coded joke |
| `thank you`, `thanks`… | Acknowledgement |
| `calc 2+3*4`… | Safe arithmetic evaluation |
| `reverse <text>`… | String reversal |
| `bye`, `exit`, `quit`… | Graceful shutdown |

### Key Design Decisions

- **Priority-ordered if-elif-else chain** — greetings checked first, then commands, then intents, then fallback
- **`eval()` is never called on raw input** — calculator expressions are regex-whitelisted to digits + operators only
- **Modular functions** — `sanitise_input()`, `detect_greeting()`, `process_input()`, `display_response()` are independently testable
- **Graceful degradation** — handles `EOFError`, `KeyboardInterrupt`, empty input, and unknown commands

### Sample Session

```
🧑 You  ▸  hello
  ┌──────────────────────────────────────────────────┐
  │ 🤖 LogicEngine v1.0:                              │
  │                                                    │
  │ Good afternoon! 🌤️  I'm LogicEngine v1.0.         │
  │ What can I do for you?                             │
  └──────────────────────────────────────────────────┘

🧑 You  ▸  calc 2+3*4
  ┌──────────────────────────────────────────────────┐
  │ 🤖 LogicEngine v1.0:                              │
  │ 🧮 Result: 2+3*4 = 14                             │
  └──────────────────────────────────────────────────┘

🧑 You  ▸  bye
  ┌──────────────────────────────────────────────────┐
  │ 🤖 LogicEngine v1.0:                              │
  │ Goodbye! 👋 It was nice chatting with you.         │
  └──────────────────────────────────────────────────┘

  📊 Session ended after 2 turn(s). Goodbye!
```

---

## 📊 Project 2 — Data Classification Using AI

> **Track:** Supervised Learning
> **File:** `data_classifier_pipeline.py`
> **Dependencies:** `numpy`, `scikit-learn`

### What It Does

Trains a **K-Nearest Neighbours** classifier on Fisher's Iris dataset (1936). The StandardScaler and train/test split are implemented **from scratch** to demonstrate understanding of the underlying math — no black-box shortcuts.

### Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  STEP 1      │ ▸▸│  STEP 3      │ ▸▸│  STEP 2      │ ▸▸│  STEP 4      │
│  Load Data   │   │  Shuffle &   │   │  Fit Scaler  │   │  Train KNN   │
│  (150 × 4)   │   │  Split       │   │  on TRAIN    │   │  Predict     │
│              │   │  80/20       │   │  only        │   │  Evaluate    │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

> **Note:** Split (Step 3) runs BEFORE scaling (Step 2) to prevent data leakage — the scaler is fitted only on training data, then applied to both sets.

### The Math

**StandardScaler (Z-Score Normalisation):**

```
                    x_ij − μ_j
x_scaled(i,j) = ────────────────
                       σ_j
```

- `μ_j` = mean of feature j (from training set only)
- `σ_j` = std deviation of feature j (from training set only)
- Result: every feature has **μ ≈ 0, σ ≈ 1** (verified in output ✅)

**Why?** Without scaling, petal length (1–7 cm) dominates petal width (0.1–2.5 cm) in Euclidean distance calculations.

**KNN Euclidean Distance:**

```
d(a, b) = √( Σ(a_j − b_j)² )   for j = 1..4
```

For each test sample → find 5 nearest training samples → majority vote on their labels.

### Results

```
📊 ACCURACY: 100.0% (30/30 test samples)

              precision    recall  f1-score   support
       setosa       1.00      1.00      1.00        13
   versicolor       1.00      1.00      1.00         6
    virginica       1.00      1.00      1.00        11
```

### Sample Inference

```
🌸 Raw input       : [5.1, 3.5, 1.4, 0.2]
📐 Scaled input    : [-0.9265, 1.084, -1.4195, -1.3747]
🏷️  Predicted class : setosa (label 0)
📊 Probabilities   : setosa 100% | versicolor 0% | virginica 0%
```

---

## 🎯 Project 3 — AI Recommendation Logic

> **Track:** Pattern Alignment
> **File:** `recommendation_engine.py`
> **Dependencies:** `numpy`

### What It Does

Builds a digital matching engine that ranks 10 items for each of 4 users using **two approaches** — then shows side-by-side how TF-IDF reshuffles rankings by rewarding specific tag matches and penalising generic ones.

### Architecture

```
┌─────────────────────┐      ┌─────────────────────────────┐
│  APPROACH 1          │      │  APPROACH 2                  │
│  Jaccard Similarity  │      │  TF-IDF Cosine Similarity   │
│                      │      │                              │
│  |A ∩ B|             │      │  IDF(t) = log(N/df(t)) + 1  │
│  ───────             │      │  cos(θ) = A⃗·B⃗ / ‖A⃗‖‖B⃗‖   │
│  |A ∪ B|             │      │                              │
│                      │      │  • Penalises generic tags    │
│  • Binary overlap    │      │  • Rewards rare tag matches  │
│  • All tags equal    │      │  • Directional alignment     │
└─────────────────────┘      └─────────────────────────────┘
```

### Unified Vocabulary Index

All 39 unique tags are mapped to the **same integer index** for both users and items. If `"sci-fi"` is index 34 in a user vector, it's index 34 in every item vector — no alignment failure.

```
Index  Tag                           IDF Weight   Specificity
─────  ────────────────────────────  ──────────── ────────────
    9  cyberpunk                        2.6094      🔵 HIGH
   34  sci-fi                           1.6931      🔴 LOW
   37  thriller                         1.6931      🔴 LOW
```

### The Math

**Jaccard Similarity:**
```
J(A, B) = |A ∩ B| / |A ∪ B|

User: {sci-fi, space, dystopia, thriller}        — 4 tags
Item: {sci-fi, dystopia, thriller, philosophical} — 4 tags
Intersection: {sci-fi, dystopia, thriller}        — 3 tags
Union: {sci-fi, space, dystopia, thriller, philosophical} — 5 tags
J = 3/5 = 0.600
```

**TF-IDF Cosine Similarity:**
```
IDF(t) = log(N / df(t)) + 1

"cyberpunk" appears in 1 of 10 items → IDF = log(10/1) + 1 = 3.303  (HIGH)
"sci-fi"   appears in 4 of 10 items → IDF = log(10/4) + 1 = 1.693  (LOW)

User vec (TF-IDF):  [0, 0, 0, 3.303, 0, 0, 0, 1.693, 0, ...]
Item vec (TF-IDF):  [0, 0, 0, 3.303, 0, 0, 0, 1.693, 0, ...]

cos(θ) = (A⃗ · B⃗) / (‖A⃗‖ × ‖B⃗‖)
```

### Ranking Comparison (Alice — sci-fi fan)

| Rank | Item | Jaccard | TF-IDF | Δ |
|:----:|:-----|:-------:|:------:|:--|
| 1 | Blade Runner 2049 | 0.6250 | **0.7403** | ↑ 0.1153 |
| 2 | Interstellar | 0.3000 | **0.3770** | ↑ 0.0770 |
| 3 | The Martian | 0.2000 | **0.2676** | ↑ 0.0676 |
| 4 | Breaking Bad | 0.2000 | **0.2266** | ↑ 0.0266 |
| 5 | Inception | 0.2000 | 0.1950 | ↓ 0.0050 |

**Why the shift?** Alice's `cyberpunk` tag (IDF=2.61) aligns with Blade Runner's `cyberpunk` — a rare, highly specific match that Jaccard treats the same as `sci-fi` (which appears in 4 items).

---

## 📚 Mathematical Reference

### Formulas Used Across All Projects

| Formula | Project | Purpose |
|:--------|:-------:|:--------|
| `x̂ = (x − μ) / σ` | P2 | Z-score normalisation |
| `d(a,b) = √(Σ(a_j − b_j)²)` | P2 | Euclidean distance |
| `J(A,B) = \|A∩B\| / \|A∪B\|` | P3 | Set overlap similarity |
| `IDF(t) = log(N / df(t)) + 1` | P3 | Inverse document frequency |
| `cos(θ) = A⃗·B⃗ / (‖A⃗‖ × ‖B⃗‖)` | P3 | Directional vector similarity |

---

## 🧪 Testing

```bash
# Verify all scripts run without errors
python logic_engine_chatbot.py < test_inputs.txt
python data_classifier_pipeline.py
python recommendation_engine.py

# Quick syntax check
python -m py_compile logic_engine_chatbot.py
python -m py_compile data_classifier_pipeline.py
python -m py_compile recommendation_engine.py
```

---

## 🛠️ Extending the Projects

### P1: Add a New Intent

```python
# In INTENT_CATALOGUE, add one new entry:
"weather": {
    "trigger_phrases": ["weather", "forecast", "temperature"],
    "response": "I don't have live weather data, but I hope it's sunny! ☀️"
},
```

### P2: Swap the Classifier

```python
# Replace KNeighborsClassifier with any sklearn classifier:
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron

# Same .fit() / .predict() API — just swap the model object
model = DecisionTreeClassifier(random_state=42)
```

### P3: Add New Users or Items

```python
# Add to the USERS or ITEMS dicts — the vocabulary auto-expands
USERS["Eve"] = ["horror", "supernatural", "psychological"]

ITEMS["Stranger Things"] = {
    "tags": ["sci-fi", "horror", "supernatural", "coming-of-age", "friendship"],
    "description": "Kids face supernatural forces in a small town.",
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Fisher's Iris Dataset** (1936) — the most famous dataset in pattern recognition
- **DecodeLabs** — project framework and educational curriculum
- **scikit-learn** — for the KNN implementation and dataset loading utilities
- **NumPy** — for all vector and matrix operations

---

<p align="center">
  <b>Built with 🧠 by DecodeLabs</b><br>
  <sub>From hard-coded rules → supervised learning → personalised recommendations</sub>
</p>
