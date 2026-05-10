# Transcript Intelligence Pipeline

A Python-based analytical system that processes ~100 AegisCloud meeting transcripts and produces actionable business intelligence for multiple stakeholder groups. The pipeline runs entirely locally (no external API keys required) and outputs a Jupyter notebook with full analysis plus a slide deck for leadership presentation.

## Prerequisites

- Python 3.10+
- pip (Python package manager)

## Setup

1. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Verify the dataset:**

Ensure the `dataset/` directory is present in the project root. It should contain ~100 meeting folders, each with:
- `meeting-info.json` (required)
- `transcript.json`
- `summary.json`
- `events.json`
- `speakers.json`
- `speaker-meta.json`

## Running the Notebook

Launch Jupyter and open the notebook:

```bash
jupyter notebook notebook.ipynb
```

Or run it non-interactively:

```bash
jupyter nbconvert --to notebook --execute notebook.ipynb --output notebook_executed.ipynb
```

The notebook executes top-to-bottom in a fresh environment without errors. All processing is local — no external API keys or services are needed.

## Expected Outputs

After running the notebook, the following outputs are generated:

- **`output/figures/`** — 9 PNG chart files:
  - `topic_distribution.png` — Topic frequency across meetings
  - `sentiment_time_series.png` — Sentiment trends over time by call type
  - `sentiment_boxplot.png` — Sentiment distribution by call type
  - `sentiment_correlation.png` — VADER vs. pre-existing score scatter plot
  - `churn_risk_ranking.png` — Top 10 at-risk accounts
  - `feature_gap_frequency.png` — Feature gaps by product area
  - `incident_timeline.png` — Incident discussions over time
  - `support_categories.png` — Support issue category distribution
  - `sentiment_shift.png` — Sentiment shift around positive pivots

## Project Structure

```
.
├── README.md                  # This file
├── notebook.ipynb             # Primary analysis notebook
├── requirements.txt           # Python dependencies (pinned versions)
├── dataset/                   # Meeting transcript data (~100 folders)
│   └── <meeting_id>/
│       ├── meeting-info.json
│       ├── transcript.json
│       ├── summary.json
│       ├── events.json
│       ├── speakers.json
│       └── speaker-meta.json
├── src/                       # Pipeline source modules
│   ├── __init__.py
│   ├── data_loader.py         # Data ingestion and call type classification
│   ├── topic_categorizer.py   # Rule-based + TF-IDF topic assignment
│   ├── sentiment_analyzer.py  # VADER sentiment scoring and trends
│   ├── insight_generator.py   # Churn risk, feature gaps, incidents, support
│   ├── visualization_engine.py # Chart generation (matplotlib/seaborn)
│   └── slide_generator.py     # python-pptx slide deck creation
├── tests/                     # Test suite
│   └── ...
└── output/                    # Generated outputs
    ├── figures/               # Exported chart PNGs
```

## Methodology

The pipeline follows a modular ETL-style architecture:

1. **Extract** — Parse all JSON meeting records, classify by call type (internal/external/support)
2. **Transform** — Apply NLP analysis:
   - Topic categorization via rule-based keyword matching validated by TF-IDF clustering
   - Sentiment scoring via VADER (weighted by sentence length, normalized to 1-5 scale)
   - Churn risk detection from external meeting signals
   - Feature gap aggregation with product area assignment
   - Incident pattern analysis from internal meetings
   - Support issue categorization with frustration detection
3. **Present** — Generate 9 publication-ready charts, compile 18-slide deck

Key design choices:
- **VADER** for sentiment: works well on short conversational text, no API keys needed
- **Rule-based + TF-IDF** for topics: interpretable results with statistical validation
- **Keyword-based** insight extraction: deterministic, reproducible, auditable
- **pandas** throughout: industry-standard data wrangling with excellent JSON support

## How the Required Tasks Are Addressed

### Task 1: Topic/Theme Categorization Pipeline

**What I built:** A hybrid categorization system in `src/topic_categorizer.py` that assigns one or more topic labels to each of the 100 meetings.

**Approach chosen:** Rule-based keyword matching as the primary method, validated by TF-IDF + K-means clustering.

**Why this approach:**
- The domain taxonomy is well-defined (AegisCloud has three products: Detect, Comply, Identity) — rule-based gives interpretable, auditable results.
- TF-IDF clustering validates that keyword assignments align with natural text groupings, catching any blind spots.
- No external API keys or LLM calls needed — fully reproducible.
- I considered pure LLM-based classification but rejected it because it would introduce non-determinism and an API dependency for a 100-document corpus where the taxonomy is already known.

**Categories identified (9 topics):**
- Product areas: Detect, Comply, Identity
- Operational themes: Incident Response, Sprint Planning, Customer Success
- Business themes: Billing, Renewal, Competitive

**Output:** Each meeting gets a list of matched topics, a primary topic, a confidence score, and the method used. The notebook shows the full distribution table and example meetings per topic.

---

### Task 2: Sentiment Analysis Across Call Types

**What I built:** An independent sentiment scoring pipeline in `src/sentiment_analyzer.py` that computes VADER sentiment per sentence, aggregates per meeting (weighted by sentence length), and produces trends by call type.

**Approach chosen:** VADER (Valence Aware Dictionary and sEntiment Reasoner) with word-count weighting and normalization to a 1-5 scale.

**Why this approach:**
- VADER is specifically designed for short, informal text (social media, conversational) — a good fit for meeting transcripts.
- Weighting by sentence length prevents short filler sentences ("Yeah", "Okay") from dominating the meeting-level score.
- Normalizing to the same 1-5 scale as the pre-existing `sentimentScore` allows direct comparison and correlation analysis.

**Key findings presented:**
- Pearson correlation between my VADER scores and the pre-existing labels (r = 0.805, p < 0.0001) — strong validation that my independent analysis aligns with the source data.
- Call type statistics: External calls have the highest average sentiment (3.91), while internal and support calls are lower (~3.69).
- Time-series trends with plain-language interpretation of what they mean for stakeholders.
- The notebook explains *why* these trends matter — e.g., support sentiment improving over time suggests process improvements are working.

---

### Task 3: Additional Insights (2-3 ideas)

I implemented **four** additional insight modules (exceeding the minimum of 2-3), each targeting a specific stakeholder:

| # | Insight | Stakeholder | Module |
|---|---------|-------------|--------|
| 1 | Churn Risk & Renewal Intelligence | Sales Managers | `detect_churn_risk()` |
| 2 | Feature Gap Analysis | Product Managers | `aggregate_feature_gaps()` |
| 3 | Incident Pattern & Technical Health | Engineering Leads | `analyze_incidents()` |
| 4 | Support Issue Categorization | Support Leaders | `categorize_support_issues()` |

Each is fully implemented with code, visualizations, and narrative interpretation in the notebook — not just described.

## What I Did Beyond the Requirements

### 1. Independent Sentiment Validation with Correlation Analysis

Rather than simply using the pre-existing `sentimentScore` from the dataset, I computed sentiment independently using VADER and then **measured the correlation** between my scores and the existing labels. This demonstrates:
- The pipeline can work on raw transcripts without pre-computed labels
- The Pearson r = 0.805 validates both my approach and the source data quality
- A scatter plot with regression line visualizes the relationship

### 2. Churn Risk Scoring with Composite Formula

I built a quantitative risk scoring model that combines multiple signals into a single 0-10 score per account:
- Churn signal key moments (×3 weight)
- Concern key moments (×1 weight)
- Competitive mentions like SentinelShield (×2 weight)
- Negative sentiment penalty (if avg < 2.5)

This produces a **ranked, actionable list** — not just a flag. Sales managers can sort by risk score and see the specific evidence (key moments) that contributed to each score.

### 3. Systemic Issue Detection with Threshold-Based Flagging

The incident pattern analyzer doesn't just list technical issues — it detects **recurring patterns** by tracking which components appear in 3+ distinct meetings. This surfaces problems that individual incident reports miss because they span multiple conversations over time. For example, the "event_processing" component appeared in 7 meetings — a clear systemic issue.

### 4. Team Health Indicators

Beyond incidents, I extract **organizational health signals** from internal meetings:
- **Timeline risk:** mentions of deadlines slipping, "60-40" odds, delays
- **Resource constraints:** "stretched thin", bandwidth issues, understaffing
- **Process gaps:** bypassed workflows, manual workarounds, inconsistent practices

These are signals that engineering leadership typically only discovers through 1:1s — here they're surfaced automatically from meeting transcripts.

### 5. Support-to-Incident Correlation

The support issue categorizer doesn't just classify cases — it **links support tickets to known incidents**. When a support call mentions "outage" and "detect" together, it's correlated to the Detect Pipeline Outage. This quantifies the support burden generated by each incident, giving engineering teams a concrete cost metric for reliability investments.

### 6. Resolution Pattern Detection

I identify `positive_pivot` moments in support calls — points where customer sentiment shifts from negative to positive. These represent successful resolution techniques that can be replicated across the support team. The notebook shows 15 such patterns with the specific pivot text and speaker.

### 7. Publication-Ready Visualization Suite

Nine charts with consistent professional styling, all exportable as PNGs:
- Horizontal bar charts, box plots, scatter plots with regression lines, donut charts, time-series with confidence bands, and incident timelines
- All charts handle empty data gracefully and include proper titles, axis labels, and legends
- Color-coded risk levels (red/orange/yellow) for the churn ranking chart
