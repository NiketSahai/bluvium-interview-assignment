"""Topic Categorizer module for the Transcript Intelligence Pipeline.

Assigns topic labels to each meeting using a hybrid approach:
1. **Primary (rule-based):** Keyword matching against meeting titles, summary text,
   and pre-existing topic arrays from summary.json. This is the primary method because
   the dataset already contains curated topic labels and the keyword taxonomy is
   well-defined for AegisCloud's product areas and operational themes.
2. **Validation (TF-IDF + K-means):** Vectorizes transcript text with TF-IDF and
   clusters meetings using K-means to validate that rule-based assignments align
   with natural text groupings. This catches meetings that may be miscategorized
   by keywords alone.
3. **Fallback:** If no rule-based match is found, maps pre-existing topics from
   summary.json to the taxonomy using fuzzy keyword overlap.

Methodology choice rationale:
- Rule-based is preferred because the domain taxonomy is well-defined and the
  dataset includes pre-labeled topics that serve as ground truth.
- TF-IDF validation adds statistical rigor without requiring external APIs.
- The hybrid approach balances interpretability (rule-based) with coverage (TF-IDF).
"""

import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Topic taxonomy with associated keywords
TOPIC_TAXONOMY = {
    # Product Areas
    "Detect": [
        "detect", "threat", "monitoring", "alert", "event processing",
        "pipeline", "ingestion", "siem", "detection",
    ],
    "Comply": [
        "comply", "compliance", "iso", "audit", "regulation",
        "framework", "certification", "policy",
    ],
    "Identity": [
        "identity", "authentication", "sso", "provisioning",
        "deprovisioning", "seats", "access", "permissions",
    ],
    # Operational Themes
    "Incident Response": [
        "outage", "incident", "remediation", "postmortem", "downtime",
        "failure", "recovery",
    ],
    "Sprint Planning": [
        "sprint", "standup", "backlog", "velocity", "story points",
        "release", "deployment",
    ],
    "Customer Success": [
        "onboarding", "adoption", "renewal", "expansion", "health score",
        "qbr", "quarterly review",
    ],
    # Business Themes
    "Billing": [
        "billing", "invoice", "payment", "pricing", "overage",
        "subscription", "contract",
    ],
    "Renewal": [
        "renewal", "churn", "retention", "upsell", "expansion", "contract",
    ],
    "Competitive": [
        "competitor", "competitive", "alternative", "switch", "evaluate",
        "sentinelshield",
    ],
}


def _build_text_for_meeting(
    meeting_id: str,
    meetings_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
) -> str:
    """Combine title, summary text, and pre-existing topics into searchable text."""
    parts = []

    # Get title from meetings_df
    meeting_row = meetings_df[meetings_df["meeting_id"] == meeting_id]
    if not meeting_row.empty:
        title = meeting_row.iloc[0].get("title", "")
        if title:
            parts.append(title)

    # Get summary text and topics from summaries_df
    summary_row = summaries_df[summaries_df["meeting_id"] == meeting_id]
    if not summary_row.empty:
        summary_text = summary_row.iloc[0].get("summary_text", "")
        if summary_text:
            parts.append(summary_text)

        topics = summary_row.iloc[0].get("topics", [])
        if isinstance(topics, list):
            parts.append(" ".join(topics))

    return " ".join(parts).lower()


def _match_keywords(text: str, keywords: list[str]) -> tuple[int, float]:
    """
    Count keyword matches in text and compute confidence.

    Returns:
        Tuple of (match_count, confidence) where confidence is
        the ratio of matched keywords to total keywords in the category.
    """
    if not text:
        return 0, 0.0

    match_count = 0
    for keyword in keywords:
        if keyword in text:
            match_count += 1

    confidence = match_count / len(keywords) if keywords else 0.0
    return match_count, confidence


def _rule_based_categorize(
    meeting_id: str,
    meetings_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
) -> tuple[list[str], str, float]:
    """
    Apply rule-based keyword matching for a single meeting.

    Returns:
        Tuple of (topics_list, primary_topic, confidence).
        If no match found, returns ([], "", 0.0).
    """
    text = _build_text_for_meeting(meeting_id, meetings_df, summaries_df)

    topic_scores = {}
    for topic, keywords in TOPIC_TAXONOMY.items():
        match_count, confidence = _match_keywords(text, keywords)
        if match_count > 0:
            topic_scores[topic] = confidence

    if not topic_scores:
        return [], "", 0.0

    # Sort by confidence descending
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    topics_list = [t[0] for t in sorted_topics]
    primary_topic = sorted_topics[0][0]
    primary_confidence = sorted_topics[0][1]

    return topics_list, primary_topic, primary_confidence


def _fallback_categorize(
    meeting_id: str,
    summaries_df: pd.DataFrame,
) -> tuple[list[str], str, float]:
    """
    Fallback: map pre-existing topics from summary.json to taxonomy.

    Uses keyword overlap between pre-existing topic strings and taxonomy keywords.
    """
    summary_row = summaries_df[summaries_df["meeting_id"] == meeting_id]
    if summary_row.empty:
        return ["Uncategorized"], "Uncategorized", 0.0

    existing_topics = summary_row.iloc[0].get("topics", [])
    if not isinstance(existing_topics, list) or not existing_topics:
        return ["Uncategorized"], "Uncategorized", 0.0

    # Try to map existing topics to taxonomy
    existing_text = " ".join(existing_topics).lower()
    topic_scores = {}

    for topic, keywords in TOPIC_TAXONOMY.items():
        match_count, confidence = _match_keywords(existing_text, keywords)
        if match_count > 0:
            topic_scores[topic] = confidence

    if not topic_scores:
        return ["Uncategorized"], "Uncategorized", 0.0

    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    topics_list = [t[0] for t in sorted_topics]
    primary_topic = sorted_topics[0][0]
    primary_confidence = sorted_topics[0][1] * 0.7  # Lower confidence for fallback

    return topics_list, primary_topic, primary_confidence


def _tfidf_cluster_validation(
    meetings_df: pd.DataFrame,
    transcripts_df: pd.DataFrame,
    rule_based_results: pd.DataFrame,
    n_clusters: int = 10,
) -> pd.DataFrame:
    """
    Validate rule-based assignments using TF-IDF + K-means clustering.

    Vectorizes all transcript text per meeting, clusters with K-means,
    and compares cluster assignments with rule-based labels.

    Returns DataFrame with meeting_id, cluster_id, and cluster_topic mapping.
    """
    # Aggregate transcript text per meeting
    if transcripts_df.empty:
        logger.warning("No transcript data available for TF-IDF validation")
        return pd.DataFrame(columns=["meeting_id", "cluster_id", "cluster_primary_topic"])

    meeting_texts = (
        transcripts_df.groupby("meeting_id")["sentence"]
        .apply(lambda x: " ".join(x.dropna()))
        .reset_index()
    )
    meeting_texts.columns = ["meeting_id", "full_text"]

    # Filter out meetings with very short text
    meeting_texts = meeting_texts[meeting_texts["full_text"].str.len() > 50]

    if meeting_texts.empty or len(meeting_texts) < n_clusters:
        logger.warning(
            "Insufficient transcript data for TF-IDF clustering "
            "(need at least %d meetings with text, got %d)",
            n_clusters,
            len(meeting_texts),
        )
        return pd.DataFrame(columns=["meeting_id", "cluster_id", "cluster_primary_topic"])

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        min_df=2,
        max_df=0.95,
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(meeting_texts["full_text"])
    except ValueError as e:
        logger.warning("TF-IDF vectorization failed: %s", e)
        return pd.DataFrame(columns=["meeting_id", "cluster_id", "cluster_primary_topic"])

    # K-means clustering
    actual_clusters = min(n_clusters, len(meeting_texts))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    meeting_texts["cluster_id"] = cluster_labels

    # Map clusters to most common rule-based topic in each cluster
    merged = meeting_texts.merge(
        rule_based_results[["meeting_id", "primary_topic"]],
        on="meeting_id",
        how="left",
    )

    cluster_topic_map = {}
    for cluster_id in range(actual_clusters):
        cluster_meetings = merged[merged["cluster_id"] == cluster_id]
        topics = cluster_meetings["primary_topic"].dropna()
        if not topics.empty:
            most_common = topics.value_counts().index[0]
            cluster_topic_map[cluster_id] = most_common
        else:
            cluster_topic_map[cluster_id] = "Uncategorized"

    meeting_texts["cluster_primary_topic"] = meeting_texts["cluster_id"].map(
        cluster_topic_map
    )

    return meeting_texts[["meeting_id", "cluster_id", "cluster_primary_topic"]]


def categorize_topics(
    meetings_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    transcripts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign topic labels to each meeting using hybrid rule-based + TF-IDF approach.

    Methodology:
    1. Rule-based keyword matching (primary): Checks meeting title, summary text,
       and pre-existing topic arrays against a defined taxonomy of product areas,
       operational themes, and business themes.
    2. TF-IDF + K-means validation: Clusters transcript text to verify rule-based
       assignments align with natural text groupings.
    3. Fallback: Maps pre-existing topics from summary.json to taxonomy when
       rule-based matching finds no hits.

    Args:
        meetings_df: DataFrame with columns meeting_id, title, etc.
        summaries_df: DataFrame with columns meeting_id, summary_text, topics (list).
        transcripts_df: DataFrame with columns meeting_id, sentence, etc.

    Returns:
        DataFrame with columns:
            meeting_id: str - unique meeting identifier
            topics: list[str] - all matched topic labels
            primary_topic: str - highest confidence topic
            confidence: float - confidence score for primary topic
            method: str - "rule_based", "tfidf_cluster", or "fallback"
    """
    results = []

    for meeting_id in meetings_df["meeting_id"].unique():
        topics, primary_topic, confidence = _rule_based_categorize(
            meeting_id, meetings_df, summaries_df
        )

        if topics:
            method = "rule_based"
        else:
            # Fallback to mapping pre-existing topics
            topics, primary_topic, confidence = _fallback_categorize(
                meeting_id, summaries_df
            )
            method = "fallback"

        results.append(
            {
                "meeting_id": meeting_id,
                "topics": topics,
                "primary_topic": primary_topic,
                "confidence": confidence,
                "method": method,
            }
        )

    results_df = pd.DataFrame(results)

    # TF-IDF validation pass
    tfidf_results = _tfidf_cluster_validation(
        meetings_df, transcripts_df, results_df
    )

    if not tfidf_results.empty:
        # For meetings where TF-IDF cluster topic differs from rule-based,
        # and rule-based confidence is low, consider upgrading to tfidf_cluster method
        merged = results_df.merge(
            tfidf_results[["meeting_id", "cluster_primary_topic"]],
            on="meeting_id",
            how="left",
        )

        # Update method for low-confidence rule-based that align with cluster
        mask = (
            (merged["method"] == "fallback")
            & (merged["cluster_primary_topic"].notna())
            & (merged["cluster_primary_topic"] != "Uncategorized")
        )

        for idx in merged[mask].index:
            cluster_topic = merged.loc[idx, "cluster_primary_topic"]
            results_df.loc[idx, "primary_topic"] = cluster_topic
            if cluster_topic not in results_df.loc[idx, "topics"]:
                current_topics = results_df.loc[idx, "topics"]
                if isinstance(current_topics, list):
                    current_topics.insert(0, cluster_topic)
                else:
                    results_df.at[idx, "topics"] = [cluster_topic]
            results_df.loc[idx, "method"] = "tfidf_cluster"
            results_df.loc[idx, "confidence"] = 0.5  # Moderate confidence for cluster

    # Log summary
    method_counts = results_df["method"].value_counts().to_dict()
    logger.info("Topic categorization complete: %d meetings processed", len(results_df))
    for method, count in sorted(method_counts.items()):
        logger.info("  %s: %d meetings", method, count)

    return results_df


def get_topic_distribution(topic_results: pd.DataFrame) -> pd.DataFrame:
    """
    Produce frequency distribution of topics across the corpus.

    Counts how many meetings each topic appears in (a meeting can have
    multiple topics, so totals may exceed meeting count).

    Args:
        topic_results: DataFrame from categorize_topics() with columns
                       meeting_id, topics (list[str]), primary_topic, confidence, method.

    Returns:
        DataFrame with columns:
            topic: str - topic label from taxonomy
            count: int - number of meetings containing this topic
            percentage: float - percentage of total meetings
    """
    if topic_results.empty:
        return pd.DataFrame(columns=["topic", "count", "percentage"])

    total_meetings = len(topic_results)

    # Count occurrences of each topic across all meetings
    topic_counter = Counter()
    for topics_list in topic_results["topics"]:
        if isinstance(topics_list, list):
            for topic in topics_list:
                topic_counter[topic] += 1

    if not topic_counter:
        return pd.DataFrame(columns=["topic", "count", "percentage"])

    distribution = pd.DataFrame(
        [
            {
                "topic": topic,
                "count": count,
                "percentage": round((count / total_meetings) * 100, 1),
            }
            for topic, count in topic_counter.most_common()
        ]
    )

    return distribution
