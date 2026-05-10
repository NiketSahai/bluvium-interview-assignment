"""Sentiment Analyzer module for the Transcript Intelligence Pipeline.

Computes VADER sentiment scores per sentence, aggregates per meeting (weighted by
sentence length), normalizes to a [1, 5] scale for comparison with pre-existing
sentimentScore, and produces call-type statistics and time-series data.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Module-level VADER analyzer instance (stateless, safe to reuse)
_vader = SentimentIntensityAnalyzer()


def compute_sentence_sentiment(sentence: str) -> dict:
    """
    Compute VADER sentiment for a single sentence.

    Args:
        sentence: Text string to analyze.

    Returns:
        Dictionary with keys: compound, pos, neg, neu
    """
    if not sentence or not isinstance(sentence, str):
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    scores = _vader.polarity_scores(sentence)
    return {
        "compound": scores["compound"],
        "pos": scores["pos"],
        "neg": scores["neg"],
        "neu": scores["neu"],
    }


def normalize_vader_to_5point(compound: float) -> float:
    """
    Map VADER compound [-1, 1] to [1, 5] scale using linear mapping.

    The mapping is: normalized = (compound + 1) * 2 + 1
    This gives: -1 -> 1.0, 0 -> 3.0, 1 -> 5.0

    Args:
        compound: VADER compound score in [-1, 1].

    Returns:
        Normalized score in [1, 5].
    """
    return (compound + 1) * 2 + 1


def _compute_meeting_sentiment(meeting_sentences: pd.DataFrame) -> dict:
    """
    Compute aggregated VADER sentiment for a single meeting.

    Weights each sentence's compound score by the number of words in the sentence.

    Args:
        meeting_sentences: DataFrame with 'sentence' column for one meeting.

    Returns:
        Dictionary with vader_compound, vader_pos, vader_neg, vader_neu (all weighted).
    """
    if meeting_sentences.empty:
        return {
            "vader_compound": 0.0,
            "vader_pos": 0.0,
            "vader_neg": 0.0,
            "vader_neu": 1.0,
        }

    compounds = []
    positives = []
    negatives = []
    neutrals = []
    weights = []

    for _, row in meeting_sentences.iterrows():
        sentence = row.get("sentence", "")
        if not sentence or not isinstance(sentence, str):
            continue

        sentiment = compute_sentence_sentiment(sentence)
        word_count = len(sentence.split())
        if word_count == 0:
            continue

        compounds.append(sentiment["compound"])
        positives.append(sentiment["pos"])
        negatives.append(sentiment["neg"])
        neutrals.append(sentiment["neu"])
        weights.append(word_count)

    if not weights:
        return {
            "vader_compound": 0.0,
            "vader_pos": 0.0,
            "vader_neg": 0.0,
            "vader_neu": 1.0,
        }

    total_weight = sum(weights)
    weighted_compound = sum(c * w for c, w in zip(compounds, weights)) / total_weight
    weighted_pos = sum(p * w for p, w in zip(positives, weights)) / total_weight
    weighted_neg = sum(n * w for n, w in zip(negatives, weights)) / total_weight
    weighted_neu = sum(n * w for n, w in zip(neutrals, weights)) / total_weight

    return {
        "vader_compound": weighted_compound,
        "vader_pos": weighted_pos,
        "vader_neg": weighted_neg,
        "vader_neu": weighted_neu,
    }


def _interpret_trends(time_series_df: pd.DataFrame) -> str:
    """
    Generate a plain-language explanation of sentiment trends.

    Args:
        time_series_df: DataFrame with date, call_type, avg_sentiment columns.

    Returns:
        String with trend interpretation.
    """
    if time_series_df.empty:
        return "Insufficient data to identify sentiment trends."

    interpretations = []

    for call_type in time_series_df["call_type"].unique():
        ct_data = time_series_df[time_series_df["call_type"] == call_type].sort_values(
            "date"
        )
        if len(ct_data) < 2:
            continue

        scores = ct_data["avg_sentiment"].values
        overall_mean = np.mean(scores)

        # Simple trend: compare first half to second half
        mid = len(scores) // 2
        first_half_mean = np.mean(scores[:mid]) if mid > 0 else overall_mean
        second_half_mean = np.mean(scores[mid:]) if mid < len(scores) else overall_mean

        diff = second_half_mean - first_half_mean

        if abs(diff) < 0.05:
            trend_desc = "remained stable"
        elif diff > 0:
            trend_desc = f"improved (+{diff:.2f})"
        else:
            trend_desc = f"declined ({diff:.2f})"

        # Interpret the level
        if overall_mean >= 3.5:
            level_desc = "generally positive"
        elif overall_mean >= 2.5:
            level_desc = "neutral"
        else:
            level_desc = "generally negative"

        interpretations.append(
            f"- {call_type.capitalize()} calls: Sentiment {trend_desc} over the period. "
            f"Average sentiment is {level_desc} ({overall_mean:.2f}/5.0)."
        )

    if not interpretations:
        return "Insufficient data points to identify meaningful trends."

    header = "Sentiment Trend Summary:\n"
    return header + "\n".join(interpretations)


def analyze_sentiment(
    transcripts_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    meetings_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute VADER sentiment and compare with existing scores.

    Processes all transcript sentences, aggregates per meeting weighted by sentence
    length, normalizes to [1, 5] scale, computes Pearson correlation with pre-existing
    scores, and produces call-type statistics and time-series data.

    Args:
        transcripts_df: DataFrame with columns: meeting_id, index, speaker_name,
                        sentence, sentiment_type, time, end_time, confidence
        summaries_df: DataFrame with columns: meeting_id, summary_text,
                      overall_sentiment, sentiment_score, topics
        meetings_df: DataFrame with columns: meeting_id, title, organizer,
                     start_time, end_time, duration, call_type, participant_count,
                     external_domains

    Returns:
        Dictionary with keys:
            - 'meeting_sentiment': DataFrame with meeting_id, vader_compound,
              vader_pos, vader_neg, vader_neu, normalized_score, existing_score,
              call_type
            - 'call_type_stats': DataFrame with call_type, mean, median, std
            - 'time_series': DataFrame with date, call_type, avg_sentiment
            - 'correlation': dict with pearson_r, p_value
            - 'trend_interpretation': str with plain-language trend explanation
    """
    # Step 1: Compute per-meeting VADER sentiment (weighted by sentence length)
    meeting_ids = meetings_df["meeting_id"].unique()
    meeting_sentiments = []

    for meeting_id in meeting_ids:
        meeting_sentences = transcripts_df[
            transcripts_df["meeting_id"] == meeting_id
        ]
        sentiment = _compute_meeting_sentiment(meeting_sentences)
        sentiment["meeting_id"] = meeting_id
        meeting_sentiments.append(sentiment)

    meeting_sentiment_df = pd.DataFrame(meeting_sentiments)

    if meeting_sentiment_df.empty:
        logger.warning("No meeting sentiment data computed.")
        empty_meeting = pd.DataFrame(
            columns=[
                "meeting_id", "vader_compound", "vader_pos", "vader_neg",
                "vader_neu", "normalized_score", "existing_score", "call_type",
            ]
        )
        empty_stats = pd.DataFrame(columns=["call_type", "mean", "median", "std"])
        empty_ts = pd.DataFrame(columns=["date", "call_type", "avg_sentiment"])
        return {
            "meeting_sentiment": empty_meeting,
            "call_type_stats": empty_stats,
            "time_series": empty_ts,
            "correlation": {"pearson_r": None, "p_value": None},
            "trend_interpretation": "No data available for trend analysis.",
        }

    # Step 2: Normalize VADER compound to [1, 5] scale
    meeting_sentiment_df["normalized_score"] = meeting_sentiment_df[
        "vader_compound"
    ].apply(normalize_vader_to_5point)

    # Step 3: Merge with existing sentiment scores from summaries
    if not summaries_df.empty and "sentiment_score" in summaries_df.columns:
        score_map = summaries_df.set_index("meeting_id")["sentiment_score"].to_dict()
        meeting_sentiment_df["existing_score"] = meeting_sentiment_df[
            "meeting_id"
        ].map(score_map)
    else:
        meeting_sentiment_df["existing_score"] = np.nan

    # Step 4: Merge with call_type from meetings
    if not meetings_df.empty and "call_type" in meetings_df.columns:
        call_type_map = meetings_df.set_index("meeting_id")["call_type"].to_dict()
        meeting_sentiment_df["call_type"] = meeting_sentiment_df["meeting_id"].map(
            call_type_map
        )
    else:
        meeting_sentiment_df["call_type"] = "unknown"

    # Step 5: Compute Pearson correlation between VADER-derived and pre-existing scores
    valid_pairs = meeting_sentiment_df.dropna(
        subset=["normalized_score", "existing_score"]
    )
    if len(valid_pairs) >= 3:
        pearson_r, p_value = stats.pearsonr(
            valid_pairs["normalized_score"], valid_pairs["existing_score"]
        )
        correlation = {"pearson_r": float(pearson_r), "p_value": float(p_value)}
    else:
        correlation = {"pearson_r": None, "p_value": None}
        logger.warning(
            "Insufficient paired data points (%d) for Pearson correlation.",
            len(valid_pairs),
        )

    # Step 6: Aggregate by call type - mean, median, standard deviation
    call_type_stats_rows = []
    for call_type in meeting_sentiment_df["call_type"].dropna().unique():
        ct_scores = meeting_sentiment_df[
            meeting_sentiment_df["call_type"] == call_type
        ]["normalized_score"]
        if len(ct_scores) > 0:
            call_type_stats_rows.append(
                {
                    "call_type": call_type,
                    "mean": float(ct_scores.mean()),
                    "median": float(ct_scores.median()),
                    "std": float(ct_scores.std()) if len(ct_scores) > 1 else 0.0,
                }
            )

    call_type_stats_df = pd.DataFrame(call_type_stats_rows)

    # Step 7: Produce time-series DataFrame (date, call_type, avg_sentiment)
    # Parse start_time from meetings_df to get the date
    if not meetings_df.empty and "start_time" in meetings_df.columns:
        meetings_with_date = meetings_df[["meeting_id", "start_time", "call_type"]].copy()
        meetings_with_date["date"] = pd.to_datetime(
            meetings_with_date["start_time"], errors="coerce"
        ).dt.date

        # Merge sentiment scores with meeting dates
        ts_data = meeting_sentiment_df[["meeting_id", "normalized_score"]].merge(
            meetings_with_date[["meeting_id", "date", "call_type"]],
            on="meeting_id",
            how="inner",
            suffixes=("_sent", "_meet"),
        )

        # Use call_type from the merge (from meetings_df)
        if "call_type_meet" in ts_data.columns:
            ts_data["call_type"] = ts_data["call_type_meet"]
            ts_data.drop(columns=["call_type_sent", "call_type_meet"], inplace=True, errors="ignore")

        # Drop rows with missing dates
        ts_data = ts_data.dropna(subset=["date"])

        if not ts_data.empty:
            # Group by week and call_type for smoother trends
            ts_data["date"] = pd.to_datetime(ts_data["date"])
            ts_data["week"] = ts_data["date"].dt.to_period("W").apply(
                lambda r: r.start_time
            )

            time_series_df = (
                ts_data.groupby(["week", "call_type"])["normalized_score"]
                .mean()
                .reset_index()
            )
            time_series_df.columns = ["date", "call_type", "avg_sentiment"]
            time_series_df = time_series_df.sort_values(
                ["call_type", "date"]
            ).reset_index(drop=True)
        else:
            time_series_df = pd.DataFrame(
                columns=["date", "call_type", "avg_sentiment"]
            )
    else:
        time_series_df = pd.DataFrame(columns=["date", "call_type", "avg_sentiment"])

    # Step 8: Generate trend interpretation
    trend_interpretation = _interpret_trends(time_series_df)

    logger.info(
        "Sentiment analysis complete: %d meetings analyzed, correlation r=%.3f (p=%.4f)",
        len(meeting_sentiment_df),
        correlation["pearson_r"] if correlation["pearson_r"] is not None else 0.0,
        correlation["p_value"] if correlation["p_value"] is not None else 1.0,
    )

    return {
        "meeting_sentiment": meeting_sentiment_df,
        "call_type_stats": call_type_stats_df,
        "time_series": time_series_df,
        "correlation": correlation,
        "trend_interpretation": trend_interpretation,
    }
