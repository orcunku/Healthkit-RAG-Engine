#  On-Device Bio-Temporal HealthKit RAG Engine

An edge-computed **Retrieval-Augmented Generation (RAG)** pipeline designed for natural language semantic search over private, time-series biometric streams and user health logs—engineered around zero-cloud data exfiltration and sub-10ms retrieval latency.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://healthkit-rag-engine-9bdz8a8m8evyoe9hhqmxqy.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  Overview & Architectural Intent

Personal biometric data (Heart Rate Variability, Sleep Analysis, Resting Heart Rate, and Workouts) represents some of the most sensitive user context on mobile devices. Traditional LLM architectures require streaming this private data to remote cloud servers to process complex health inquiries.

This project implements an **on-device vector retrieval engine** that combines quantitative HealthKit metric streams with qualitative user journal entries into **bio-temporal context chunks**. Queries are evaluated entirely in local memory with zero network transit, demonstrating how personal health context can be queried instantly and privately on the edge.

---

##  Key Architectural Features

* **Sub-10ms Contextual Vector Search:** Utilizes lightweight cosine-similarity vector quantization over bio-temporal embeddings to return relevant health records instantly.
* **Bio-Temporal Context Chunking:** Fuses discrete time-series metrics (`HKQuantityTypeIdentifierHeartRate`, sleep stage scores, workout metrics) with qualitative journal notes into unified temporal search candidates.
* **Zero-Cloud Data Exfiltration:** All vector math and index calculations execute transiently in RAM (<15MB footprint), ensuring strict client-side data isolation.
* **Confidence Threshold Gating:** Implemented precision confidence boundaries (default `0.60`) to reject out-of-scope or irrelevant queries gracefully without hallucinating biometric results.

---

## Pipeline Architecture

[ Natural Language Query ]  (e.g., "Late night work session high stress low HRV")
│
▼
┌─────────────────────────────────────────┐
│ On-Device Bio-Vector Embedder          │
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│ Local Cosine Similarity Matching       │ ◄── [ In-Memory HealthKit Context Chunks ]
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│ Confidence Threshold Gate (≥ 0.60)      │ ──► (Rejected if Out-of-Scope)
└─────────────────────────────────────────┘
│ (Accepted)
▼
┌─────────────────────────────────────────┐
│ Bio-Metric Analytics & Context Display  │
└─────────────────────────────────────────┘
│
▼
[ Output: Resolved Day Snapshot, HRV/Sleep Metrics & Qualitative Journal Insight ]


---

## Quickstart & Local Setup

### Prerequisites
* Python 3.10 or higher
* Git

### Installation & Execution

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/orcunku/HealthKit-BioTemporal-RAG-Engine.git](https://github.com/orcunku/HealthKit-BioTemporal-RAG-Engine.git)
   cd HealthKit-BioTemporal-RAG-Engine
Install Dependencies:

Bash
pip install -r requirements.txt
Launch the Application:python -m streamlit run app.py


Test Scenarios
Scenario                                   Input Query                                                Expected Result
Exact Metric & Context Match"             Late night work session high stress low HRV"                Matches 2026-09-02 with ~81% confidence,                                                                                           surfacing 5.2h sleep, 38ms HRV, and the                                                                                                   associated work stress journal entry.


Complex Correlation Query                  "How did my sleep quality correlate with my HRV recovery?"  Correctly resolves high recovery                                                                                                          days (2026-09-03 / 2026-09-01) with sleep scores $>85$.


Out-of-Scope Rejection                     "What is the stock price of Apple today"                     Fails Threshold Gate (<10% match                                                                                                          score.                                                                                                                                    Displays graceful fallback alert.


Tech StackLanguage: Python 3.10+Framework: Streamlit (UI & State Management)Vector Math: NumPy, Pandas, Plotly ExpressData Schema: Synthetic HealthKit Time-Series Data (healthkit_data.json)Environment: GitHub Codespaces / VS Code


📄 Legal Disclaimer:This repository is an independent research prototype created solely for educational and portfolio demonstration purposes. It is not affiliated with, endorsed by, sponsored by, or associated with Apple Inc. All product names, trademarks, and registered trademarks—including "Apple," "HealthKit," and "CoreML"—are the property of their respective owners.


📜 LicenseDistributed under the MIT License. See LICENSE for more information.

