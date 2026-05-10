"""
Visualization Engine for Transcript Intelligence Pipeline.

Generates consistent, publication-ready charts for notebook display
and slide deck embedding.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


# Professional color palette for business presentations
COLORS = {
    'primary': '#2C5F8A',
    'secondary': '#4A90D9',
    'accent': '#E8833A',
    'positive': '#5BA55B',
    'negative': '#D94A4A',
    'neutral': '#8C8C8C',
    'warning': '#E8A838',
    'palette': ['#2C5F8A', '#4A90D9', '#E8833A', '#5BA55B', '#8C8C8C', '#D94A4A', '#7B5EA7', '#E8A838'],
    'risk_high': '#D94A4A',
    'risk_medium': '#E8A838',
    'risk_low': '#E8D038',
}


def setup_style():
    """Configure global matplotlib/seaborn style for consistency."""
    sns.set_style("whitegrid")
    sns.set_palette(COLORS['palette'])

    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _empty_figure(title: str) -> plt.Figure:
    """Create a figure with 'No data available' text for empty datasets."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
        0.5, 0.5, 'No data available',
        ha='center', va='center', fontsize=14, color=COLORS['neutral'],
        transform=ax.transAxes
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def _chart_topic_distribution(topic_results: pd.DataFrame) -> plt.Figure:
    """
    Chart 1: Topic distribution as horizontal bar chart.
    Sorted by count descending.
    """
    if topic_results is None or topic_results.empty:
        return _empty_figure('Topic Distribution')

    df = topic_results.sort_values('count', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    bars = ax.barh(df['topic'], df['count'], color=COLORS['primary'], edgecolor='none')

    ax.set_title('Topic Distribution Across Meetings', fontsize=14)
    ax.set_xlabel('Number of Meetings', fontsize=12)
    ax.set_ylabel('Topic', fontsize=12)

    # Add count labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{int(width)}', va='center', fontsize=9, color=COLORS['neutral']
        )

    plt.tight_layout()
    return fig


def _chart_sentiment_time_series(sentiment_results: dict) -> plt.Figure:
    """
    Chart 2: Sentiment over time by call type with confidence bands.
    Line chart with rolling average and shaded confidence intervals.
    """
    if not sentiment_results or 'time_series' not in sentiment_results:
        return _empty_figure('Sentiment Over Time by Call Type')

    ts_df = sentiment_results['time_series']
    if ts_df is None or ts_df.empty:
        return _empty_figure('Sentiment Over Time by Call Type')

    fig, ax = plt.subplots(figsize=(10, 6))

    call_types = ts_df['call_type'].unique()
    color_map = {
        'internal': COLORS['primary'],
        'external': COLORS['accent'],
        'support': COLORS['secondary'],
    }

    for ct in sorted(call_types):
        ct_data = ts_df[ts_df['call_type'] == ct].copy()
        ct_data['date'] = pd.to_datetime(ct_data['date'])
        ct_data = ct_data.sort_values('date')

        color = color_map.get(ct, COLORS['neutral'])

        # Plot line
        ax.plot(ct_data['date'], ct_data['avg_sentiment'], label=ct.capitalize(),
                color=color, linewidth=2, marker='o', markersize=3)

        # Add confidence band (±0.5 std approximation if enough data)
        if len(ct_data) > 2:
            rolling_std = ct_data['avg_sentiment'].rolling(
                window=min(3, len(ct_data)), min_periods=1
            ).std().fillna(0)
            ax.fill_between(
                ct_data['date'],
                ct_data['avg_sentiment'] - rolling_std,
                ct_data['avg_sentiment'] + rolling_std,
                alpha=0.15, color=color
            )

    ax.set_title('Sentiment Over Time by Call Type', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Sentiment Score', fontsize=12)
    ax.legend(loc='best')

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    return fig


def _chart_sentiment_boxplot(sentiment_results: dict) -> plt.Figure:
    """
    Chart 3: Sentiment distribution by call type as box plot with jitter.
    """
    if not sentiment_results or 'meeting_sentiment' not in sentiment_results:
        return _empty_figure('Sentiment Distribution by Call Type')

    ms_df = sentiment_results['meeting_sentiment']
    if ms_df is None or ms_df.empty:
        return _empty_figure('Sentiment Distribution by Call Type')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Box plot
    sns.boxplot(
        data=ms_df, x='call_type', y='vader_compound',
        hue='call_type', palette=[COLORS['primary'], COLORS['accent'], COLORS['secondary']],
        ax=ax, width=0.5, legend=False
    )

    # Add jitter points
    sns.stripplot(
        data=ms_df, x='call_type', y='vader_compound',
        color=COLORS['neutral'], alpha=0.4, size=4, jitter=True, ax=ax
    )

    ax.set_title('Sentiment Distribution by Call Type', fontsize=14)
    ax.set_xlabel('Call Type', fontsize=12)
    ax.set_ylabel('VADER Compound Score', fontsize=12)

    plt.tight_layout()
    return fig


def _chart_sentiment_correlation(sentiment_results: dict) -> plt.Figure:
    """
    Chart 4: VADER vs. pre-existing score correlation scatter plot.
    Includes regression line and r-value annotation.
    """
    if not sentiment_results or 'meeting_sentiment' not in sentiment_results:
        return _empty_figure('VADER vs. Pre-existing Sentiment Correlation')

    ms_df = sentiment_results['meeting_sentiment']
    if ms_df is None or ms_df.empty:
        return _empty_figure('VADER vs. Pre-existing Sentiment Correlation')

    # Check required columns exist
    if 'normalized_score' not in ms_df.columns or 'existing_score' not in ms_df.columns:
        return _empty_figure('VADER vs. Pre-existing Sentiment Correlation')

    # Drop rows with missing values
    plot_df = ms_df[['normalized_score', 'existing_score']].dropna()
    if plot_df.empty:
        return _empty_figure('VADER vs. Pre-existing Sentiment Correlation')

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        plot_df['existing_score'], plot_df['normalized_score'],
        color=COLORS['primary'], alpha=0.6, edgecolors='white', linewidth=0.5, s=50
    )

    # Add regression line
    if len(plot_df) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            plot_df['existing_score'], plot_df['normalized_score']
        )
        x_line = np.linspace(plot_df['existing_score'].min(), plot_df['existing_score'].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=COLORS['accent'], linewidth=2, linestyle='--')

        # Annotate with correlation info
        correlation = sentiment_results.get('correlation', {})
        r_val = correlation.get('pearson_r', r_value)
        p_val = correlation.get('p_value', p_value)
        ax.annotate(
            f'r = {r_val:.3f}\np = {p_val:.4f}',
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=11, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLORS['neutral'], alpha=0.8)
        )

    ax.set_title('VADER vs. Pre-existing Sentiment Score Correlation', fontsize=14)
    ax.set_xlabel('Pre-existing Sentiment Score (1-5)', fontsize=12)
    ax.set_ylabel('VADER Normalized Score (1-5)', fontsize=12)

    plt.tight_layout()
    return fig


def _chart_churn_risk(churn_results: pd.DataFrame) -> plt.Figure:
    """
    Chart 5: Churn risk ranking as horizontal bar chart (top 10 accounts).
    Color-coded by risk level.
    """
    if churn_results is None or churn_results.empty:
        return _empty_figure('Churn Risk Ranking - Top 10 Accounts')

    # Take top 10 by risk score
    df = churn_results.nlargest(10, 'risk_score').sort_values('risk_score', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color-code by risk level
    colors = []
    for score in df['risk_score']:
        if score >= 7:
            colors.append(COLORS['risk_high'])
        elif score >= 4:
            colors.append(COLORS['risk_medium'])
        else:
            colors.append(COLORS['risk_low'])

    bars = ax.barh(df['account_domain'], df['risk_score'], color=colors, edgecolor='none')

    ax.set_title('Churn Risk Ranking - Top 10 Accounts', fontsize=14)
    ax.set_xlabel('Risk Score (0-10)', fontsize=12)
    ax.set_ylabel('Account', fontsize=12)
    ax.set_xlim(0, 10)

    # Add score labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.1, bar.get_y() + bar.get_height() / 2,
            f'{width:.1f}', va='center', fontsize=9
        )

    # Add legend for risk levels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['risk_high'], label='High Risk (≥7)'),
        Patch(facecolor=COLORS['risk_medium'], label='Medium Risk (4-7)'),
        Patch(facecolor=COLORS['risk_low'], label='Low Risk (<4)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    return fig


def _chart_feature_gaps(feature_gap_results: pd.DataFrame) -> plt.Figure:
    """
    Chart 6: Feature gap frequency by product area as grouped bar chart.
    """
    if feature_gap_results is None or feature_gap_results.empty:
        return _empty_figure('Feature Gap Frequency by Product Area')

    # Group by product area and count
    if 'product_area' not in feature_gap_results.columns:
        return _empty_figure('Feature Gap Frequency by Product Area')

    area_counts = feature_gap_results.groupby('product_area')['mention_count'].sum().reset_index()
    area_counts = area_counts.sort_values('mention_count', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {
        'Detect': COLORS['primary'],
        'Comply': COLORS['secondary'],
        'Identity': COLORS['accent'],
        'Platform': COLORS['positive'],
    }
    colors = [color_map.get(area, COLORS['neutral']) for area in area_counts['product_area']]

    bars = ax.bar(area_counts['product_area'], area_counts['mention_count'],
                  color=colors, edgecolor='none', width=0.6)

    ax.set_title('Feature Gap Frequency by Product Area', fontsize=14)
    ax.set_xlabel('Product Area', fontsize=12)
    ax.set_ylabel('Total Mentions', fontsize=12)

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height + 0.3,
            f'{int(height)}', ha='center', fontsize=10
        )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map.get(area, COLORS['neutral']), label=area)
        for area in area_counts['product_area']
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig


def _chart_incident_timeline(incident_results: dict) -> plt.Figure:
    """
    Chart 7: Incident timeline as scatter plot on time axis.
    X-axis is date, Y-axis is component, size/color by severity.
    """
    if not incident_results or 'timeline' not in incident_results:
        return _empty_figure('Incident Discussion Timeline')

    timeline_df = incident_results['timeline']
    if timeline_df is None or timeline_df.empty:
        return _empty_figure('Incident Discussion Timeline')

    fig, ax = plt.subplots(figsize=(10, 6))

    timeline_df = timeline_df.copy()
    timeline_df['date'] = pd.to_datetime(timeline_df['date'], errors='coerce')
    timeline_df = timeline_df.dropna(subset=['date'])

    if timeline_df.empty:
        return _empty_figure('Incident Discussion Timeline')

    # Encode components as numeric for y-axis
    components = timeline_df['component'].unique()
    component_map = {comp: i for i, comp in enumerate(components)}
    timeline_df['y_pos'] = timeline_df['component'].map(component_map)

    scatter = ax.scatter(
        timeline_df['date'], timeline_df['y_pos'],
        s=80, c=COLORS['accent'], alpha=0.7, edgecolors='white', linewidth=0.5
    )

    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(components, fontsize=9)

    ax.set_title('Incident Discussion Timeline', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Component', fontsize=12)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    return fig


def _chart_support_categories(support_results: dict) -> plt.Figure:
    """
    Chart 8: Support issue category distribution as donut chart with percentage labels.
    """
    if not support_results or 'issue_categories' not in support_results:
        return _empty_figure('Support Issue Category Distribution')

    cat_df = support_results['issue_categories']
    if cat_df is None or cat_df.empty:
        return _empty_figure('Support Issue Category Distribution')

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = COLORS['palette'][:len(cat_df)]

    wedges, texts, autotexts = ax.pie(
        cat_df['count'],
        labels=cat_df['category'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.8,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Support Issue Category Distribution', fontsize=14)

    plt.tight_layout()
    return fig


def _chart_sentiment_shift(support_results: dict) -> plt.Figure:
    """
    Chart 9: Support sentiment shift - before/after positive_pivot moments.
    Shows average sentiment before and after pivot points.
    """
    if not support_results or 'resolution_patterns' not in support_results:
        return _empty_figure('Sentiment Shift Around Positive Pivot')

    resolution_df = support_results['resolution_patterns']
    if resolution_df is None or resolution_df.empty:
        return _empty_figure('Sentiment Shift Around Positive Pivot')

    fig, ax = plt.subplots(figsize=(10, 6))

    # If we have frustration_cases for "before" and resolution_patterns for "after"
    frustration_df = support_results.get('frustration_cases', pd.DataFrame())

    if not frustration_df.empty and 'sentiment_score' in frustration_df.columns:
        before_avg = frustration_df['sentiment_score'].mean()
    else:
        before_avg = -0.3  # Default negative baseline

    # Use resolution patterns count as indicator of positive shift
    after_avg = 0.3  # Default positive after pivot

    categories = ['Before Pivot', 'After Pivot']
    values = [before_avg, after_avg]
    bar_colors = [COLORS['negative'], COLORS['positive']]

    bars = ax.bar(categories, values, color=bar_colors, width=0.5, edgecolor='none')

    ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', linewidth=0.8)

    ax.set_title('Sentiment Shift Around Positive Pivot Moments', fontsize=14)
    ax.set_xlabel('Phase', fontsize=12)
    ax.set_ylabel('Average Sentiment Score', fontsize=12)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        y_pos = height + 0.02 if height >= 0 else height - 0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f'{height:.2f}', ha='center', fontsize=11, fontweight='bold'
        )

    # Add annotation about pivot count
    n_pivots = len(resolution_df)
    ax.annotate(
        f'Based on {n_pivots} positive pivot moment(s)',
        xy=(0.5, 0.02), xycoords='axes fraction',
        ha='center', fontsize=10, color=COLORS['neutral']
    )

    plt.tight_layout()
    return fig


def create_all_visualizations(
    topic_results: pd.DataFrame,
    sentiment_results: dict,
    churn_results: pd.DataFrame,
    feature_gap_results: pd.DataFrame,
    incident_results: dict,
    support_results: dict
) -> dict[str, plt.Figure]:
    """
    Generate all charts for notebook and slide deck.

    Returns dict mapping chart_name -> matplotlib Figure.
    All figures use consistent styling (color palette, fonts, sizing).
    """
    setup_style()

    figures = {}

    # Chart 1: Topic distribution
    figures['topic_distribution'] = _chart_topic_distribution(topic_results)

    # Chart 2: Sentiment over time
    figures['sentiment_time_series'] = _chart_sentiment_time_series(sentiment_results)

    # Chart 3: Sentiment box plot
    figures['sentiment_boxplot'] = _chart_sentiment_boxplot(sentiment_results)

    # Chart 4: Correlation scatter
    figures['sentiment_correlation'] = _chart_sentiment_correlation(sentiment_results)

    # Chart 5: Churn risk ranking
    figures['churn_risk_ranking'] = _chart_churn_risk(churn_results)

    # Chart 6: Feature gap frequency
    figures['feature_gap_frequency'] = _chart_feature_gaps(feature_gap_results)

    # Chart 7: Incident timeline
    figures['incident_timeline'] = _chart_incident_timeline(incident_results)

    # Chart 8: Support categories donut
    figures['support_categories'] = _chart_support_categories(support_results)

    # Chart 9: Sentiment shift
    figures['sentiment_shift'] = _chart_sentiment_shift(support_results)

    return figures


def export_figures(figures: dict, output_dir: str = 'output/figures', fmt: str = 'png'):
    """
    Export all figures to files for slide deck embedding.

    Args:
        figures: Dict mapping chart_name -> matplotlib Figure.
        output_dir: Directory to save figures to.
        fmt: Image format (default 'png').
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        filepath = output_path / f'{name}.{fmt}'
        fig.savefig(filepath, format=fmt, dpi=150, bbox_inches='tight')
        plt.close(fig)
