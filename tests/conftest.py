"""Shared test fixtures for the Transcript Intelligence Pipeline."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def sample_meeting_info():
    """Generate a sample meeting-info.json dict."""
    return {
        "meetingId": "TEST_MEETING_001",
        "title": "Detect Outage - Remediation Plan Review",
        "organizerEmail": "megan.lawson@aegiscloud.com",
        "host": "megan.lawson@aegiscloud.com",
        "startTime": "2026-03-16T09:30:00.000Z",
        "endTime": "2026-03-16T10:05:12.000Z",
        "duration": 35.2,
        "allEmails": [
            "megan.lawson@aegiscloud.com",
            "raj.kapoor@aegiscloud.com",
            "brian.cho@aegiscloud.com",
        ],
        "invitees": [
            "megan.lawson@aegiscloud.com",
            "raj.kapoor@aegiscloud.com",
            "brian.cho@aegiscloud.com",
        ],
    }


@pytest.fixture
def sample_transcript():
    """Generate a sample transcript.json dict with a few sentences."""
    return {
        "data": [
            {
                "sentence": "Alright, I think we're all on.",
                "speaker_name": "Megan Lawson",
                "sentimentType": "neutral",
                "speaker_id": 0,
                "time": 7.4,
                "endTime": 12.6,
                "averageConfidence": 0.97,
                "index": 0,
            },
            {
                "sentence": "Yeah, I'm here. Audio's good.",
                "speaker_name": "Raj Kapoor",
                "sentimentType": "neutral",
                "speaker_id": 1,
                "time": 13.4,
                "endTime": 14.9,
                "averageConfidence": 0.93,
                "index": 1,
            },
            {
                "sentence": "This is really frustrating, we need to fix this now.",
                "speaker_name": "Brian Cho",
                "sentimentType": "negative",
                "speaker_id": 2,
                "time": 16.2,
                "endTime": 21.2,
                "averageConfidence": 0.95,
                "index": 2,
            },
            {
                "sentence": "Great news, the deployment was successful!",
                "speaker_name": "Megan Lawson",
                "sentimentType": "positive",
                "speaker_id": 0,
                "time": 22.6,
                "endTime": 27.0,
                "averageConfidence": 0.98,
                "index": 3,
            },
        ]
    }


@pytest.fixture
def sample_summary():
    """Generate a sample summary.json dict with topics, key moments, sentiment."""
    return {
        "summary": "Team discussed the ongoing Detect outage remediation plan.",
        "actionItems": [
            "Megan Lawson: Draft updated customer communication within the hour",
            "Raj Kapoor: Send evening status update once rollout phase one is complete",
        ],
        "topics": [
            "outage remediation",
            "incident response",
            "customer communication",
        ],
        "overallSentiment": "mixed-negative",
        "sentimentScore": 2.4,
        "keyMoments": [
            {
                "time": 95.0,
                "text": "112 open support tickets tied to the outage",
                "type": "churn_signal",
                "speaker": "Brian Cho",
            },
            {
                "time": 310.0,
                "text": "Customers had zero threat monitoring data for six hours",
                "type": "technical_issue",
                "speaker": "Raj Kapoor",
            },
            {
                "time": 370.0,
                "text": "No affirmative statements about security posture during outage",
                "type": "concern",
                "speaker": "Megan Lawson",
            },
            {
                "time": 225.0,
                "text": "Conservative timeline approach to rebuild credibility",
                "type": "positive_pivot",
                "speaker": "Raj Kapoor",
            },
        ],
        "meetingId": "TEST_MEETING_001",
    }


@pytest.fixture
def sample_events():
    """Generate a sample events.json list."""
    return [
        {
            "participantName": "Megan Lawson",
            "timestamp": 1773653406000,
            "type": "Join",
            "time": 6.0,
        },
        {
            "participantName": "Raj Kapoor",
            "timestamp": 1773653417000,
            "type": "Join",
            "time": 17.0,
        },
        {
            "participantName": "Megan Lawson",
            "timestamp": 1773655488000,
            "type": "Leave",
            "time": 2088.0,
        },
    ]


@pytest.fixture
def sample_speakers():
    """Generate a sample speakers.json list."""
    return [
        {"speakerName": "Megan Lawson", "timestamp": 7.4, "endTimeTs": 12.6},
        {"speakerName": "Raj Kapoor", "timestamp": 13.4, "endTimeTs": 14.9},
        {"speakerName": "Brian Cho", "timestamp": 16.2, "endTimeTs": 21.2},
    ]


@pytest.fixture
def sample_speaker_meta():
    """Generate a sample speaker-meta.json dict."""
    return {"0": "Megan Lawson", "1": "Raj Kapoor", "2": "Brian Cho"}


@pytest.fixture
def tmp_dataset_valid(
    tmp_path, sample_meeting_info, sample_transcript, sample_summary,
    sample_events, sample_speakers, sample_speaker_meta
):
    """Create a temporary dataset directory with one valid meeting folder."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    meeting_dir = dataset_dir / "TEST_MEETING_001"
    meeting_dir.mkdir()

    (meeting_dir / "meeting-info.json").write_text(
        json.dumps(sample_meeting_info, indent=2)
    )
    (meeting_dir / "transcript.json").write_text(
        json.dumps(sample_transcript, indent=2)
    )
    (meeting_dir / "summary.json").write_text(
        json.dumps(sample_summary, indent=2)
    )
    (meeting_dir / "events.json").write_text(
        json.dumps(sample_events, indent=2)
    )
    (meeting_dir / "speakers.json").write_text(
        json.dumps(sample_speakers, indent=2)
    )
    (meeting_dir / "speaker-meta.json").write_text(
        json.dumps(sample_speaker_meta, indent=2)
    )

    return dataset_dir


@pytest.fixture
def tmp_dataset_mixed(tmp_path, sample_meeting_info, sample_transcript,
                      sample_summary, sample_events, sample_speakers,
                      sample_speaker_meta):
    """Create a temporary dataset with a mix of valid and malformed folders."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # Valid meeting folder
    valid_dir = dataset_dir / "VALID_MEETING_001"
    valid_dir.mkdir()
    (valid_dir / "meeting-info.json").write_text(
        json.dumps(sample_meeting_info, indent=2)
    )
    (valid_dir / "transcript.json").write_text(
        json.dumps(sample_transcript, indent=2)
    )
    (valid_dir / "summary.json").write_text(
        json.dumps(sample_summary, indent=2)
    )
    (valid_dir / "events.json").write_text(
        json.dumps(sample_events, indent=2)
    )
    (valid_dir / "speakers.json").write_text(
        json.dumps(sample_speakers, indent=2)
    )
    (valid_dir / "speaker-meta.json").write_text(
        json.dumps(sample_speaker_meta, indent=2)
    )

    # Malformed folder: missing meeting-info.json
    missing_info_dir = dataset_dir / "MALFORMED_NO_INFO"
    missing_info_dir.mkdir()
    (missing_info_dir / "transcript.json").write_text(
        json.dumps(sample_transcript, indent=2)
    )

    # Malformed folder: invalid JSON syntax
    invalid_json_dir = dataset_dir / "MALFORMED_BAD_JSON"
    invalid_json_dir.mkdir()
    (invalid_json_dir / "meeting-info.json").write_text(
        "{invalid json content, missing quotes"
    )

    # Malformed folder: empty directory
    empty_dir = dataset_dir / "MALFORMED_EMPTY"
    empty_dir.mkdir()

    # Second valid meeting (external call)
    valid_dir2 = dataset_dir / "VALID_MEETING_002"
    valid_dir2.mkdir()
    external_info = sample_meeting_info.copy()
    external_info["meetingId"] = "VALID_MEETING_002"
    external_info["title"] = "Quarterly Business Review"
    external_info["allEmails"] = [
        "megan.lawson@aegiscloud.com",
        "john.smith@externalcorp.com",
    ]
    (valid_dir2 / "meeting-info.json").write_text(
        json.dumps(external_info, indent=2)
    )
    (valid_dir2 / "transcript.json").write_text(
        json.dumps(sample_transcript, indent=2)
    )
    (valid_dir2 / "summary.json").write_text(
        json.dumps(sample_summary, indent=2)
    )
    (valid_dir2 / "events.json").write_text(
        json.dumps(sample_events, indent=2)
    )
    (valid_dir2 / "speakers.json").write_text(
        json.dumps(sample_speakers, indent=2)
    )
    (valid_dir2 / "speaker-meta.json").write_text(
        json.dumps(sample_speaker_meta, indent=2)
    )

    return dataset_dir
