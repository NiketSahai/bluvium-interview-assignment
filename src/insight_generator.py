"""Insight Generator module for the Transcript Intelligence Pipeline.

Orchestrates domain-specific insight extraction sub-modules:
- Churn Risk Detector: Identifies at-risk accounts from external meetings
- Feature Gap Aggregator: Extracts and consolidates feature gaps across meetings
- Incident Pattern Analyzer: Identifies incident patterns from internal meetings
- Support Issue Categorizer: Categorizes support interactions and identifies patterns
"""

import logging
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# Constants and Keyword Maps
# ============================================================================

PRODUCT_AREA_KEYWORDS = {
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
    "Platform": [
        "billing", "api", "integration", "dashboard", "reporting",
        "performance", "uptime", "infrastructure",
    ],
}

SUPPORT_ISSUE_CATEGORIES = {
    "billing": [
        "billing", "invoice", "payment", "pricing", "overage",
        "subscription", "contract",
    ],
    "technical": [
        "error", "bug", "crash", "timeout", "latency",
        "performance", "outage", "down",
    ],
    "provisioning": [
        "provisioning", "deprovisioning", "seats", "accounts",
        "migration", "setup",
    ],
    "compliance": [
        "compliance", "audit", "certification", "regulation",
        "policy", "framework",
    ],
    "access/permissions": [
        "access", "permission", "login", "sso", "authentication",
        "password", "locked",
    ],
    "performance": [
        "slow", "latency", "timeout", "degraded", "performance",
    ],
}

# Component keywords for incident pattern analysis
COMPONENT_KEYWORDS = {
    "event_processing": ["event processing", "pipeline", "ingestion", "event pipeline"],
    "detect": ["detect", "detection", "threat monitoring", "siem"],
    "identity": ["identity", "sso", "authentication", "provisioning", "deprovisioning"],
    "comply": ["comply", "compliance", "audit", "certification"],
    "billing": ["billing", "invoice", "payment", "subscription"],
    "api": ["api", "endpoint", "integration", "webhook"],
    "database": ["database", "db", "query", "migration", "schema"],
    "infrastructure": ["infrastructure", "server", "node", "cluster", "deployment"],
    "networking": ["network", "dns", "load balancer", "latency", "timeout"],
    "ui": ["dashboard", "ui", "frontend", "interface", "console"],
}

# Team health indicator keywords
TIMELINE_RISK_KEYWORDS = [
    "deadline", "behind", "slip", "delay", "risk", "60-40",
    "timeline", "overdue", "late", "pushed back",
]
RESOURCE_CONSTRAINT_KEYWORDS = [
    "stretched", "bandwidth", "capacity", "overloaded", "thin",
    "understaffed", "overwhelmed", "burnout", "short-staffed",
]
PROCESS_GAP_KEYWORDS = [
    "bypass", "workaround", "manual", "inconsistent", "broken",
    "ad-hoc", "no process", "undocumented",
]

# Negative sentiment threshold for support frustration
FRUSTRATION_THRESHOLD = 3.0

# Competitor names to look for
COMPETITOR_NAMES = [
    "sentinelshield", "sentinel shield", "crowdstrike", "splunk",
    "palo alto", "fortinet", "darktrace",
]



# ============================================================================
# Sub-module 4a: Churn Risk Detector
# ============================================================================


def detect_churn_risk(
    meetings_df: pd.DataFrame,
    key_moments_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify at-risk accounts from external meetings.

    Filters external call type meetings for churn_signal and competitive_threat
    key moments, computes a risk score per account domain, and produces a ranked
    list with evidence.

    Risk scoring formula:
        - Base: count of churn_signal key moments × 3
        - Add: count of concern key moments × 1
        - Add: negative sentiment penalty (if avg sentiment < 2.5, add (2.5 - score) × 2)
        - Add: competitive mention bonus × 2
        - Normalize to 0-10 scale (cap at 10)

    Args:
        meetings_df: DataFrame with meeting metadata including call_type and external_domains.
        key_moments_df: DataFrame with key moments (meeting_id, time, text, type, speaker).
        sentiment_df: DataFrame with meeting sentiment (meeting_id, normalized_score, call_type).

    Returns:
        DataFrame with columns:
            account_domain, risk_score, churn_signals_count,
            avg_sentiment, competitive_mentions, evidence (list of key moments)
    """
    if meetings_df.empty or key_moments_df.empty:
        return pd.DataFrame(columns=[
            "account_domain", "risk_score", "churn_signals_count",
            "avg_sentiment", "competitive_mentions", "evidence",
        ])

    # Filter to external meetings only
    external_meetings = meetings_df[meetings_df["call_type"] == "external"].copy()
    if external_meetings.empty:
        return pd.DataFrame(columns=[
            "account_domain", "risk_score", "churn_signals_count",
            "avg_sentiment", "competitive_mentions", "evidence",
        ])

    external_meeting_ids = set(external_meetings["meeting_id"].tolist())

    # Get key moments for external meetings
    external_moments = key_moments_df[
        key_moments_df["meeting_id"].isin(external_meeting_ids)
    ].copy()

    # Build mapping: meeting_id -> external domains
    meeting_to_domains = {}
    for _, row in external_meetings.iterrows():
        domains = row.get("external_domains", [])
        if isinstance(domains, list) and domains:
            meeting_to_domains[row["meeting_id"]] = domains

    # Build mapping: meeting_id -> sentiment score
    meeting_to_sentiment = {}
    if not sentiment_df.empty and "normalized_score" in sentiment_df.columns:
        for _, row in sentiment_df.iterrows():
            meeting_to_sentiment[row["meeting_id"]] = row["normalized_score"]

    # Aggregate signals per account domain
    account_data = {}  # domain -> {churn_signals, concerns, competitive, sentiment_scores, evidence, meeting_ids}

    for _, moment in external_moments.iterrows():
        meeting_id = moment["meeting_id"]
        domains = meeting_to_domains.get(meeting_id, [])

        for domain in domains:
            if domain not in account_data:
                account_data[domain] = {
                    "churn_signals": 0,
                    "concerns": 0,
                    "competitive_mentions": [],
                    "sentiment_scores": [],
                    "evidence": [],
                    "meeting_ids": set(),
                }

            moment_type = moment.get("type", "")
            moment_text = moment.get("text", "")

            if moment_type == "churn_signal":
                account_data[domain]["churn_signals"] += 1
                account_data[domain]["evidence"].append({
                    "meeting_id": meeting_id,
                    "type": moment_type,
                    "text": moment_text,
                    "speaker": moment.get("speaker", ""),
                    "time": moment.get("time", 0.0),
                })
            elif moment_type == "concern":
                account_data[domain]["concerns"] += 1
                account_data[domain]["evidence"].append({
                    "meeting_id": meeting_id,
                    "type": moment_type,
                    "text": moment_text,
                    "speaker": moment.get("speaker", ""),
                    "time": moment.get("time", 0.0),
                })
            elif moment_type == "competitive_threat":
                account_data[domain]["competitive_mentions"].append(moment_text)
                account_data[domain]["evidence"].append({
                    "meeting_id": meeting_id,
                    "type": moment_type,
                    "text": moment_text,
                    "speaker": moment.get("speaker", ""),
                    "time": moment.get("time", 0.0),
                })

            # Check for competitor mentions in text
            text_lower = moment_text.lower()
            for competitor in COMPETITOR_NAMES:
                if competitor in text_lower:
                    if moment_text not in account_data[domain]["competitive_mentions"]:
                        account_data[domain]["competitive_mentions"].append(moment_text)

            account_data[domain]["meeting_ids"].add(meeting_id)

    # Add sentiment data per domain
    for domain, data in account_data.items():
        for meeting_id in data["meeting_ids"]:
            if meeting_id in meeting_to_sentiment:
                data["sentiment_scores"].append(meeting_to_sentiment[meeting_id])

    # Compute risk scores
    results = []
    for domain, data in account_data.items():
        churn_signals = data["churn_signals"]
        concerns = data["concerns"]
        competitive_count = len(data["competitive_mentions"])

        # Average sentiment for this account
        if data["sentiment_scores"]:
            avg_sentiment = np.mean(data["sentiment_scores"])
        else:
            avg_sentiment = 3.0  # neutral default

        # Risk score calculation
        raw_score = (churn_signals * 3) + (concerns * 1) + (competitive_count * 2)

        # Sentiment penalty: if avg sentiment < 2.5, add (2.5 - score) × 2
        if avg_sentiment < 2.5:
            raw_score += (2.5 - avg_sentiment) * 2

        # Normalize to 0-10 (cap at 10)
        risk_score = min(raw_score, 10.0)

        results.append({
            "account_domain": domain,
            "risk_score": round(risk_score, 2),
            "churn_signals_count": churn_signals,
            "avg_sentiment": round(avg_sentiment, 2),
            "competitive_mentions": data["competitive_mentions"],
            "evidence": data["evidence"],
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values("risk_score", ascending=False).reset_index(drop=True)

    logger.info(
        "Churn risk analysis complete: %d accounts assessed, %d with non-zero risk",
        len(result_df),
        len(result_df[result_df["risk_score"] > 0]) if not result_df.empty else 0,
    )

    return result_df


# ============================================================================
# Sub-module 4b: Feature Gap Aggregator
# ============================================================================


def _assign_product_area(text: str) -> str:
    """
    Assign a product area based on keyword matching in the text.

    Args:
        text: The key moment text or meeting title to classify.

    Returns:
        Product area string: 'Detect', 'Comply', 'Identity', or 'Platform'.
    """
    text_lower = text.lower()
    scores = {}

    for area, keywords in PRODUCT_AREA_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[area] = score

    # Return the area with the highest score, default to 'Platform' if no match
    best_area = max(scores, key=scores.get)
    if scores[best_area] == 0:
        return "Platform"  # default fallback
    return best_area


def _are_similar(text1: str, text2: str, threshold: float = 0.6) -> bool:
    """
    Check if two text descriptions are similar enough to consolidate.

    Uses SequenceMatcher ratio for fuzzy string comparison.

    Args:
        text1: First text string.
        text2: Second text string.
        threshold: Similarity threshold (0-1). Default 0.6.

    Returns:
        True if texts are similar above threshold.
    """
    if not text1 or not text2:
        return False
    ratio = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return ratio >= threshold


def _get_stakeholder_type(call_type: str) -> str:
    """
    Map call_type to stakeholder type.

    Args:
        call_type: Meeting call type ('internal', 'external', 'support').

    Returns:
        Stakeholder type string.
    """
    mapping = {
        "external": "customer",
        "internal": "internal",
        "support": "support",
    }
    return mapping.get(call_type, "unknown")


def aggregate_feature_gaps(
    key_moments_df: pd.DataFrame,
    meetings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract and consolidate feature gaps across all meetings.

    Extracts all feature_gap key moments, assigns product area via keyword mapping,
    groups by description similarity, consolidates duplicates, and ranks by frequency
    of mention across distinct meetings.

    Args:
        key_moments_df: DataFrame with key moments (meeting_id, time, text, type, speaker).
        meetings_df: DataFrame with meeting metadata (meeting_id, call_type, title).

    Returns:
        DataFrame with columns:
            feature_description, product_area, mention_count,
            source_meetings (list), stakeholder_types (list), verbatim_contexts (list)
    """
    if key_moments_df.empty or meetings_df.empty:
        return pd.DataFrame(columns=[
            "feature_description", "product_area", "mention_count",
            "source_meetings", "stakeholder_types", "verbatim_contexts",
        ])

    # Filter for feature_gap key moments
    feature_gaps = key_moments_df[key_moments_df["type"] == "feature_gap"].copy()

    if feature_gaps.empty:
        return pd.DataFrame(columns=[
            "feature_description", "product_area", "mention_count",
            "source_meetings", "stakeholder_types", "verbatim_contexts",
        ])

    # Build meeting_id -> call_type mapping
    meeting_call_types = meetings_df.set_index("meeting_id")["call_type"].to_dict()

    # Build meeting_id -> title mapping for additional context
    meeting_titles = meetings_df.set_index("meeting_id")["title"].to_dict()

    # Extract individual feature gap records
    gap_records = []
    for _, row in feature_gaps.iterrows():
        meeting_id = row["meeting_id"]
        text = row.get("text", "")
        call_type = meeting_call_types.get(meeting_id, "unknown")
        title = meeting_titles.get(meeting_id, "")

        # Assign product area using both the moment text and meeting title
        combined_text = f"{text} {title}"
        product_area = _assign_product_area(combined_text)
        stakeholder_type = _get_stakeholder_type(call_type)

        gap_records.append({
            "text": text,
            "meeting_id": meeting_id,
            "product_area": product_area,
            "stakeholder_type": stakeholder_type,
        })

    # Consolidate similar descriptions
    consolidated = []  # list of {description, product_area, meetings, stakeholders, verbatims}

    for record in gap_records:
        text = record["text"]
        meeting_id = record["meeting_id"]
        product_area = record["product_area"]
        stakeholder_type = record["stakeholder_type"]

        # Try to find an existing consolidated entry that matches
        matched = False
        for entry in consolidated:
            if _are_similar(text, entry["description"]):
                if meeting_id not in entry["source_meetings"]:
                    entry["source_meetings"].append(meeting_id)
                if stakeholder_type not in entry["stakeholder_types"]:
                    entry["stakeholder_types"].append(stakeholder_type)
                entry["verbatim_contexts"].append(text)
                entry["mention_count"] += 1
                matched = True
                break

        if not matched:
            consolidated.append({
                "description": text,
                "product_area": product_area,
                "mention_count": 1,
                "source_meetings": [meeting_id],
                "stakeholder_types": [stakeholder_type],
                "verbatim_contexts": [text],
            })

    # Build result DataFrame
    results = []
    for entry in consolidated:
        results.append({
            "feature_description": entry["description"],
            "product_area": entry["product_area"],
            "mention_count": entry["mention_count"],
            "source_meetings": entry["source_meetings"],
            "stakeholder_types": entry["stakeholder_types"],
            "verbatim_contexts": entry["verbatim_contexts"],
        })

    result_df = pd.DataFrame(results)

    # Rank by frequency of mention across distinct meetings
    if not result_df.empty:
        result_df["distinct_meeting_count"] = result_df["source_meetings"].apply(len)
        result_df = result_df.sort_values(
            ["distinct_meeting_count", "mention_count"],
            ascending=[False, False],
        ).reset_index(drop=True)
        result_df.drop(columns=["distinct_meeting_count"], inplace=True)

    logger.info(
        "Feature gap analysis complete: %d unique gaps from %d total mentions",
        len(result_df),
        result_df["mention_count"].sum() if not result_df.empty else 0,
    )

    return result_df


# ============================================================================
# Sub-module 4c: Incident Pattern Analyzer
# ============================================================================


def _extract_component(text: str) -> str:
    """
    Extract the affected system/component from a technical issue description.

    Uses keyword matching against known component names.

    Args:
        text: The key moment text describing a technical issue.

    Returns:
        Component name string, or 'other' if no match found.
    """
    text_lower = text.lower()

    scores = {}
    for component, keywords in COMPONENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[component] = score

    best_component = max(scores, key=scores.get)
    if scores[best_component] == 0:
        return "other"
    return best_component


def _detect_team_health_signals(text: str) -> dict:
    """
    Detect team health indicator signals in text.

    Args:
        text: Text to analyze for health signals.

    Returns:
        Dict with keys 'timeline_risk', 'resource_constraints', 'process_gaps'
        each containing a boolean.
    """
    text_lower = text.lower()
    return {
        "timeline_risk": any(kw in text_lower for kw in TIMELINE_RISK_KEYWORDS),
        "resource_constraints": any(kw in text_lower for kw in RESOURCE_CONSTRAINT_KEYWORDS),
        "process_gaps": any(kw in text_lower for kw in PROCESS_GAP_KEYWORDS),
    }


def analyze_incidents(
    meetings_df: pd.DataFrame,
    key_moments_df: pd.DataFrame,
    action_items_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Identify incident patterns from internal meetings.

    Filters internal call type meetings for technical_issue key moments, groups by
    affected system/component, detects recurring patterns (3+ meetings = systemic),
    produces incident timeline, team health indicators, and action items by owner.

    Args:
        meetings_df: DataFrame with meeting metadata (meeting_id, call_type, start_time, title).
        key_moments_df: DataFrame with key moments (meeting_id, time, text, type, speaker).
        action_items_df: DataFrame with action items (meeting_id, assignee, action_text).

    Returns:
        Dict with keys:
            - 'incidents': DataFrame grouped by system/component
            - 'systemic_issues': DataFrame of issues in 3+ meetings
            - 'timeline': DataFrame of incident discussions over time
            - 'team_health': DataFrame with risk/resource/process signals
            - 'action_items_by_owner': DataFrame grouped by assignee
    """
    empty_result = {
        "incidents": pd.DataFrame(columns=[
            "component", "issue_count", "meeting_count", "meetings", "sample_issues",
        ]),
        "systemic_issues": pd.DataFrame(columns=[
            "component", "issue_count", "meeting_count", "meetings", "is_systemic",
        ]),
        "timeline": pd.DataFrame(columns=[
            "meeting_id", "date", "component", "text", "speaker",
        ]),
        "team_health": pd.DataFrame(columns=[
            "meeting_id", "date", "timeline_risk", "resource_constraints", "process_gaps", "text",
        ]),
        "action_items_by_owner": pd.DataFrame(columns=[
            "assignee", "action_count", "actions",
        ]),
    }

    if meetings_df.empty or key_moments_df.empty:
        return empty_result

    # Filter to internal meetings only
    internal_meetings = meetings_df[meetings_df["call_type"] == "internal"].copy()
    if internal_meetings.empty:
        return empty_result

    internal_meeting_ids = set(internal_meetings["meeting_id"].tolist())

    # Get key moments for internal meetings
    internal_moments = key_moments_df[
        key_moments_df["meeting_id"].isin(internal_meeting_ids)
    ].copy()

    # Filter for technical_issue type
    tech_issues = internal_moments[internal_moments["type"] == "technical_issue"].copy()

    # Build meeting_id -> start_time mapping
    meeting_dates = {}
    for _, row in internal_meetings.iterrows():
        start_time = row.get("start_time", "")
        if start_time:
            try:
                meeting_dates[row["meeting_id"]] = pd.to_datetime(start_time)
            except (ValueError, TypeError):
                meeting_dates[row["meeting_id"]] = pd.NaT

    # ---- Incidents grouped by component ----
    component_data = {}  # component -> {issues: [], meetings: set()}

    for _, row in tech_issues.iterrows():
        text = row.get("text", "")
        meeting_id = row["meeting_id"]
        component = _extract_component(text)

        if component not in component_data:
            component_data[component] = {"issues": [], "meetings": set()}

        component_data[component]["issues"].append({
            "text": text,
            "meeting_id": meeting_id,
            "speaker": row.get("speaker", ""),
            "time": row.get("time", 0.0),
        })
        component_data[component]["meetings"].add(meeting_id)

    incidents_rows = []
    for component, data in component_data.items():
        incidents_rows.append({
            "component": component,
            "issue_count": len(data["issues"]),
            "meeting_count": len(data["meetings"]),
            "meetings": list(data["meetings"]),
            "sample_issues": [i["text"] for i in data["issues"][:5]],
        })

    incidents_df = pd.DataFrame(incidents_rows)
    if not incidents_df.empty:
        incidents_df = incidents_df.sort_values("issue_count", ascending=False).reset_index(drop=True)

    # ---- Systemic issues (3+ meetings) ----
    systemic_rows = []
    for component, data in component_data.items():
        is_systemic = len(data["meetings"]) >= 3
        systemic_rows.append({
            "component": component,
            "issue_count": len(data["issues"]),
            "meeting_count": len(data["meetings"]),
            "meetings": list(data["meetings"]),
            "is_systemic": is_systemic,
        })

    systemic_df = pd.DataFrame(systemic_rows)
    if not systemic_df.empty:
        systemic_df = systemic_df[systemic_df["is_systemic"]].reset_index(drop=True)

    # ---- Timeline of incident discussions ----
    timeline_rows = []
    for _, row in tech_issues.iterrows():
        meeting_id = row["meeting_id"]
        date = meeting_dates.get(meeting_id, pd.NaT)
        component = _extract_component(row.get("text", ""))

        timeline_rows.append({
            "meeting_id": meeting_id,
            "date": date,
            "component": component,
            "text": row.get("text", ""),
            "speaker": row.get("speaker", ""),
        })

    timeline_df = pd.DataFrame(timeline_rows)
    if not timeline_df.empty:
        timeline_df = timeline_df.sort_values("date").reset_index(drop=True)

    # ---- Team health indicators ----
    # Analyze ALL internal key moments (not just technical_issue) for health signals
    health_rows = []
    for _, row in internal_moments.iterrows():
        text = row.get("text", "")
        meeting_id = row["meeting_id"]
        signals = _detect_team_health_signals(text)

        if any(signals.values()):
            date = meeting_dates.get(meeting_id, pd.NaT)
            health_rows.append({
                "meeting_id": meeting_id,
                "date": date,
                "timeline_risk": signals["timeline_risk"],
                "resource_constraints": signals["resource_constraints"],
                "process_gaps": signals["process_gaps"],
                "text": text,
            })

    team_health_df = pd.DataFrame(health_rows)
    if not team_health_df.empty:
        team_health_df = team_health_df.sort_values("date").reset_index(drop=True)

    # ---- Action items by owner ----
    if not action_items_df.empty:
        internal_actions = action_items_df[
            action_items_df["meeting_id"].isin(internal_meeting_ids)
        ].copy()

        if not internal_actions.empty:
            owner_groups = internal_actions.groupby("assignee").agg(
                action_count=("action_text", "count"),
                actions=("action_text", list),
            ).reset_index()
            owner_groups = owner_groups.sort_values("action_count", ascending=False).reset_index(drop=True)
        else:
            owner_groups = pd.DataFrame(columns=["assignee", "action_count", "actions"])
    else:
        owner_groups = pd.DataFrame(columns=["assignee", "action_count", "actions"])

    logger.info(
        "Incident analysis complete: %d components, %d systemic issues, %d health signals",
        len(incidents_df),
        len(systemic_df),
        len(team_health_df),
    )

    return {
        "incidents": incidents_df,
        "systemic_issues": systemic_df,
        "timeline": timeline_df,
        "team_health": team_health_df,
        "action_items_by_owner": owner_groups,
    }


# ============================================================================
# Sub-module 4d: Support Issue Categorizer
# ============================================================================


def _categorize_issue(text: str) -> str:
    """
    Categorize a support issue based on keyword matching.

    Args:
        text: Combined text from meeting title and summary to classify.

    Returns:
        Category string from SUPPORT_ISSUE_CATEGORIES, or 'other' if no match.
    """
    text_lower = text.lower()
    scores = {}

    for category, keywords in SUPPORT_ISSUE_CATEGORIES.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[category] = score

    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "other"
    return best_category


def categorize_support_issues(
    meetings_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    key_moments_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Categorize support interactions and identify patterns.

    Filters support call type meetings, categorizes by issue type via keyword
    matching, identifies high-frustration meetings, detects resolution patterns,
    and links support cases to known incidents.

    Args:
        meetings_df: DataFrame with meeting metadata (meeting_id, call_type, title).
        summaries_df: DataFrame with summaries (meeting_id, summary_text, sentiment_score).
        key_moments_df: DataFrame with key moments (meeting_id, time, text, type, speaker).

    Returns:
        Dict with keys:
            - 'issue_categories': DataFrame with category, count, percentage
            - 'frustration_cases': DataFrame of high-frustration meetings
            - 'resolution_patterns': DataFrame of positive_pivot cases
            - 'incident_correlation': DataFrame linking support to incidents
    """
    empty_result = {
        "issue_categories": pd.DataFrame(columns=["category", "count", "percentage"]),
        "frustration_cases": pd.DataFrame(columns=[
            "meeting_id", "title", "sentiment_score", "frustration_indicators",
        ]),
        "resolution_patterns": pd.DataFrame(columns=[
            "meeting_id", "title", "pivot_text", "speaker",
        ]),
        "incident_correlation": pd.DataFrame(columns=[
            "meeting_id", "title", "related_incident", "evidence",
        ]),
    }

    if meetings_df.empty:
        return empty_result

    # Filter to support meetings only
    support_meetings = meetings_df[meetings_df["call_type"] == "support"].copy()
    if support_meetings.empty:
        return empty_result

    support_meeting_ids = set(support_meetings["meeting_id"].tolist())

    # Build meeting_id -> title mapping
    meeting_titles = support_meetings.set_index("meeting_id")["title"].to_dict()

    # Build meeting_id -> summary_text mapping
    meeting_summaries = {}
    meeting_sentiment_scores = {}
    if not summaries_df.empty:
        for _, row in summaries_df.iterrows():
            if row["meeting_id"] in support_meeting_ids:
                meeting_summaries[row["meeting_id"]] = row.get("summary_text", "")
                meeting_sentiment_scores[row["meeting_id"]] = row.get("sentiment_score", 3.0)

    # ---- Issue categorization ----
    meeting_categories = {}
    for meeting_id in support_meeting_ids:
        title = meeting_titles.get(meeting_id, "")
        summary = meeting_summaries.get(meeting_id, "")
        combined_text = f"{title} {summary}"
        category = _categorize_issue(combined_text)
        meeting_categories[meeting_id] = category

    # Count categories
    category_counts = {}
    for category in meeting_categories.values():
        category_counts[category] = category_counts.get(category, 0) + 1

    total_support = len(support_meeting_ids)
    category_rows = []
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_support * 100) if total_support > 0 else 0
        category_rows.append({
            "category": category,
            "count": count,
            "percentage": round(percentage, 1),
        })

    issue_categories_df = pd.DataFrame(category_rows)

    # ---- High-frustration meetings ----
    # Use sentiment_score from summaries (1-5 scale); threshold is 3.0
    frustration_rows = []
    for meeting_id in support_meeting_ids:
        sentiment_score = meeting_sentiment_scores.get(meeting_id, 3.0)
        if sentiment_score < FRUSTRATION_THRESHOLD:
            title = meeting_titles.get(meeting_id, "")
            # Extract frustration indicators from key moments
            meeting_moments = key_moments_df[
                key_moments_df["meeting_id"] == meeting_id
            ] if not key_moments_df.empty else pd.DataFrame()

            indicators = []
            if not meeting_moments.empty:
                concerns = meeting_moments[meeting_moments["type"] == "concern"]
                for _, m in concerns.iterrows():
                    indicators.append(m.get("text", ""))

            frustration_rows.append({
                "meeting_id": meeting_id,
                "title": title,
                "sentiment_score": sentiment_score,
                "frustration_indicators": indicators,
            })

    frustration_df = pd.DataFrame(frustration_rows)
    if not frustration_df.empty:
        frustration_df = frustration_df.sort_values("sentiment_score").reset_index(drop=True)

    # ---- Resolution patterns (positive_pivot key moments) ----
    resolution_rows = []
    if not key_moments_df.empty:
        support_moments = key_moments_df[
            key_moments_df["meeting_id"].isin(support_meeting_ids)
        ]
        pivots = support_moments[support_moments["type"] == "positive_pivot"]

        for _, row in pivots.iterrows():
            meeting_id = row["meeting_id"]
            resolution_rows.append({
                "meeting_id": meeting_id,
                "title": meeting_titles.get(meeting_id, ""),
                "pivot_text": row.get("text", ""),
                "speaker": row.get("speaker", ""),
            })

    resolution_df = pd.DataFrame(resolution_rows)

    # ---- Incident correlation ----
    # Link support cases to known incidents by looking for incident-related keywords
    incident_keywords = [
        "outage", "incident", "down", "degraded", "disruption",
        "failure", "cascading", "pipeline",
    ]

    correlation_rows = []
    for meeting_id in support_meeting_ids:
        title = meeting_titles.get(meeting_id, "")
        summary = meeting_summaries.get(meeting_id, "")
        combined = f"{title} {summary}".lower()

        # Check if this support case mentions incident-related terms
        matched_keywords = [kw for kw in incident_keywords if kw in combined]
        if matched_keywords:
            # Try to identify the specific incident
            related_incident = "unknown"
            if "detect" in combined and ("outage" in combined or "pipeline" in combined):
                related_incident = "Detect Pipeline Outage"
            elif "identity" in combined and ("outage" in combined or "down" in combined):
                related_incident = "Identity Service Disruption"
            elif "billing" in combined and "outage" in combined:
                related_incident = "Billing System Outage"

            correlation_rows.append({
                "meeting_id": meeting_id,
                "title": title,
                "related_incident": related_incident,
                "evidence": matched_keywords,
            })

    incident_correlation_df = pd.DataFrame(correlation_rows)

    logger.info(
        "Support issue analysis complete: %d meetings categorized, %d frustration cases, %d resolutions",
        total_support,
        len(frustration_df),
        len(resolution_df),
    )

    return {
        "issue_categories": issue_categories_df,
        "frustration_cases": frustration_df,
        "resolution_patterns": resolution_df,
        "incident_correlation": incident_correlation_df,
    }
