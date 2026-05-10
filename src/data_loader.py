"""Data Loader module for the Transcript Intelligence Pipeline.

Parses all meeting folders from the dataset directory, builds unified DataFrames,
and classifies meetings by call type (internal, support, external).
"""

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def classify_call_type(meeting_info: dict) -> str:
    """
    Classify a meeting as 'internal', 'support', or 'external'.

    Rules (applied in priority order):
        - support: title matches pattern 'Support Case #XXXX' (regex: Support Case\\s*#?\\d+)
        - internal: all emails end with @aegiscloud.com
        - external: mix of @aegiscloud.com and other domains (not support)

    Args:
        meeting_info: Dictionary from meeting-info.json containing at minimum
                      'title' and 'allEmails' keys.

    Returns:
        One of 'support', 'internal', or 'external'.
    """
    title = meeting_info.get("title", "")
    emails = meeting_info.get("allEmails", [])

    # Support: title matches "Support Case #XXXX" pattern
    if re.search(r"Support Case\s*#?\d+", title, re.IGNORECASE):
        return "support"

    # Internal: all participants have @aegiscloud.com
    if emails and all(
        email.lower().endswith("@aegiscloud.com") for email in emails
    ):
        return "internal"

    # External: mix of internal and external domains
    return "external"


def _extract_external_domains(emails: list[str]) -> list[str]:
    """Extract unique non-aegiscloud domains from email list."""
    domains = set()
    for email in emails:
        if "@" in email:
            domain = email.split("@")[1].lower()
            if domain != "aegiscloud.com":
                domains.add(domain)
    return sorted(domains)


def _parse_meeting_folder(folder_path: Path) -> dict | None:
    """
    Parse a single meeting folder and return structured data.

    Returns None if meeting-info.json is missing or contains invalid JSON,
    since that file is required for meeting identification.
    """
    meeting_id = folder_path.name
    result = {"meeting_id": meeting_id}

    # meeting-info.json is required
    info_path = folder_path / "meeting-info.json"
    if not info_path.exists():
        logger.warning(
            "Missing meeting-info.json in folder '%s', skipping meeting",
            meeting_id,
        )
        return None

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            meeting_info = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(
            "Malformed JSON in meeting-info.json for '%s': %s, skipping meeting",
            meeting_id,
            e,
        )
        return None

    result["meeting_info"] = meeting_info

    # Parse optional files - log warnings but continue
    optional_files = {
        "transcript": "transcript.json",
        "summary": "summary.json",
        "events": "events.json",
        "speakers": "speakers.json",
        "speaker_meta": "speaker-meta.json",
    }

    for key, filename in optional_files.items():
        file_path = folder_path / filename
        if not file_path.exists():
            logger.warning(
                "Missing %s in folder '%s', continuing without it",
                filename,
                meeting_id,
            )
            result[key] = None
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                result[key] = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(
                "Malformed JSON in %s for '%s': %s, skipping this file",
                filename,
                meeting_id,
                e,
            )
            result[key] = None

    return result


def _build_meetings_df(parsed_meetings: list[dict]) -> pd.DataFrame:
    """Build the meetings DataFrame from parsed meeting data."""
    rows = []
    for m in parsed_meetings:
        info = m["meeting_info"]
        emails = info.get("allEmails", [])
        call_type = classify_call_type(info)
        external_domains = _extract_external_domains(emails)

        rows.append(
            {
                "meeting_id": m["meeting_id"],
                "title": info.get("title", ""),
                "organizer": info.get("organizerEmail", ""),
                "start_time": info.get("startTime", ""),
                "end_time": info.get("endTime", ""),
                "duration": info.get("duration", 0.0),
                "call_type": call_type,
                "participant_count": len(emails),
                "external_domains": external_domains,
            }
        )

    return pd.DataFrame(rows)


def _build_transcripts_df(parsed_meetings: list[dict]) -> pd.DataFrame:
    """Build the transcripts DataFrame from parsed meeting data."""
    rows = []
    for m in parsed_meetings:
        transcript = m.get("transcript")
        if transcript is None:
            continue

        # transcript.json has a "data" key containing the list of sentences
        sentences = transcript.get("data", []) if isinstance(transcript, dict) else []

        for sentence in sentences:
            rows.append(
                {
                    "meeting_id": m["meeting_id"],
                    "index": sentence.get("index", 0),
                    "speaker_name": sentence.get("speaker_name", ""),
                    "sentence": sentence.get("sentence", ""),
                    "sentiment_type": sentence.get("sentimentType", ""),
                    "time": sentence.get("time", 0.0),
                    "end_time": sentence.get("endTime", 0.0),
                    "confidence": sentence.get("averageConfidence", 0.0),
                }
            )

    return pd.DataFrame(rows)


def _build_summaries_df(parsed_meetings: list[dict]) -> pd.DataFrame:
    """Build the summaries DataFrame from parsed meeting data."""
    rows = []
    for m in parsed_meetings:
        summary = m.get("summary")
        if summary is None:
            continue

        rows.append(
            {
                "meeting_id": m["meeting_id"],
                "summary_text": summary.get("summary", ""),
                "overall_sentiment": summary.get("overallSentiment", ""),
                "sentiment_score": summary.get("sentimentScore", 0.0),
                "topics": summary.get("topics", []),
            }
        )

    return pd.DataFrame(rows)


def _build_key_moments_df(parsed_meetings: list[dict]) -> pd.DataFrame:
    """Build the key_moments DataFrame from parsed meeting data."""
    rows = []
    for m in parsed_meetings:
        summary = m.get("summary")
        if summary is None:
            continue

        key_moments = summary.get("keyMoments", [])
        for moment in key_moments:
            rows.append(
                {
                    "meeting_id": m["meeting_id"],
                    "time": moment.get("time", 0.0),
                    "text": moment.get("text", ""),
                    "type": moment.get("type", ""),
                    "speaker": moment.get("speaker", ""),
                }
            )

    return pd.DataFrame(rows)


def _build_action_items_df(parsed_meetings: list[dict]) -> pd.DataFrame:
    """Build the action_items DataFrame from parsed meeting data."""
    rows = []
    for m in parsed_meetings:
        summary = m.get("summary")
        if summary is None:
            continue

        action_items = summary.get("actionItems", [])
        for item in action_items:
            # Action items are strings like "Person Name: Action text"
            if ":" in item:
                assignee, action_text = item.split(":", 1)
                assignee = assignee.strip()
                action_text = action_text.strip()
            else:
                assignee = ""
                action_text = item.strip()

            rows.append(
                {
                    "meeting_id": m["meeting_id"],
                    "assignee": assignee,
                    "action_text": action_text,
                }
            )

    return pd.DataFrame(rows)


def _build_events_df(parsed_meetings: list[dict]) -> pd.DataFrame:
    """Build the events DataFrame from parsed meeting data."""
    rows = []
    for m in parsed_meetings:
        events = m.get("events")
        if events is None:
            continue

        if not isinstance(events, list):
            continue

        for event in events:
            rows.append(
                {
                    "meeting_id": m["meeting_id"],
                    "participant": event.get("participantName", ""),
                    "timestamp": event.get("timestamp", 0),
                    "event_type": event.get("type", ""),
                    "time_offset": event.get("time", 0.0),
                }
            )

    return pd.DataFrame(rows)


def _build_speakers_df(parsed_meetings: list[dict]) -> pd.DataFrame:
    """Build the speakers DataFrame from parsed meeting data."""
    rows = []
    for m in parsed_meetings:
        speakers = m.get("speakers")
        if speakers is None:
            continue

        if not isinstance(speakers, list):
            continue

        for speaker in speakers:
            rows.append(
                {
                    "meeting_id": m["meeting_id"],
                    "speaker_name": speaker.get("speakerName", ""),
                    "start_time": speaker.get("timestamp", 0.0),
                    "end_time": speaker.get("endTimeTs", 0.0),
                }
            )

    return pd.DataFrame(rows)


def load_dataset(dataset_path: str) -> dict[str, pd.DataFrame]:
    """
    Load all meeting records from dataset directory.

    Parses each meeting folder, builds unified DataFrames, classifies call types,
    and reports summary statistics. Malformed or missing files are logged as
    warnings and skipped gracefully.

    Args:
        dataset_path: Path to the dataset directory containing meeting folders.

    Returns:
        Dictionary with keys:
            - 'meetings': DataFrame with meeting metadata + call_type
            - 'transcripts': DataFrame with sentence-level transcript data
            - 'summaries': DataFrame with summary text, sentiment, topics
            - 'key_moments': DataFrame with all key moments across meetings
            - 'action_items': DataFrame with action items across meetings
            - 'events': DataFrame with join/leave events
            - 'speakers': DataFrame with speaker timing data
    """
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}"
        )

    if not dataset_dir.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: {dataset_path}"
        )

    # Collect all meeting folders, skipping non-directory entries
    meeting_folders = sorted(
        [
            entry
            for entry in dataset_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        ]
    )

    total_folders = len(meeting_folders)
    parsed_meetings = []
    skipped_count = 0

    for folder in meeting_folders:
        parsed = _parse_meeting_folder(folder)
        if parsed is None:
            skipped_count += 1
        else:
            parsed_meetings.append(parsed)

    # Build all DataFrames
    meetings_df = _build_meetings_df(parsed_meetings)
    transcripts_df = _build_transcripts_df(parsed_meetings)
    summaries_df = _build_summaries_df(parsed_meetings)
    key_moments_df = _build_key_moments_df(parsed_meetings)
    action_items_df = _build_action_items_df(parsed_meetings)
    events_df = _build_events_df(parsed_meetings)
    speakers_df = _build_speakers_df(parsed_meetings)

    # Report summary statistics
    loaded_count = len(parsed_meetings)
    call_type_counts = (
        meetings_df["call_type"].value_counts().to_dict()
        if not meetings_df.empty
        else {}
    )

    logger.info(
        "Dataset loaded: %d meetings from %d folders (%d skipped)",
        loaded_count,
        total_folders,
        skipped_count,
    )
    for call_type, count in sorted(call_type_counts.items()):
        logger.info("  %s: %d meetings", call_type, count)

    return {
        "meetings": meetings_df,
        "transcripts": transcripts_df,
        "summaries": summaries_df,
        "key_moments": key_moments_df,
        "action_items": action_items_df,
        "events": events_df,
        "speakers": speakers_df,
    }
