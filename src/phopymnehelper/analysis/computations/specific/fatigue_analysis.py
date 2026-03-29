import numpy as np
import mne
from scipy import signal
from scipy.stats import pearsonr
import pandas as pd
# from mne.time_frequency import psd_multitaper
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import mne
from scipy.stats import ttest_rel, pearsonr
import warnings
warnings.filterwarnings('ignore')


def compute_fatigue_metrics(raw, duration=60, overlap=0.5):
    """
    Compute comprehensive EEG fatigue metrics from MNE Raw object

    Parameters:
    -----------
    raw : mne.Raw
        MNE Raw object containing EEG data
    duration : float
        Duration of epochs in seconds for analysis (default: 60s)
    overlap : float
        Overlap between epochs (0-1, default: 0.5)

    Returns:
    --------
    dict : Dictionary containing all computed fatigue metrics
    """

    # Basic parameters
    sfreq = raw.info['sfreq']
    n_samples = int(duration * sfreq)
    overlap_samples = int(overlap * n_samples)

    # Create epochs for analysis
    epochs_data = []
    n_total_samples = len(raw.times)

    start_idx = 0
    while start_idx + n_samples <= n_total_samples:
        epoch_data = raw.get_data(start=start_idx, stop=start_idx + n_samples)
        epochs_data.append(epoch_data)
        start_idx += n_samples - overlap_samples

    epochs_data = np.array(epochs_data)
    n_epochs, n_channels, n_timepoints = epochs_data.shape

    # Get channel names and positions
    ch_names = raw.info['ch_names']

    # Define channel groups
    frontal_channels = [ch for ch in ch_names if any(x in ch.upper() for x in ['FZ', 'F3', 'F4', 'FP1', 'FP2'])]
    parietal_channels = [ch for ch in ch_names if any(x in ch.upper() for x in ['PZ', 'P3', 'P4'])]
    occipital_channels = [ch for ch in ch_names if any(x in ch.upper() for x in ['OZ', 'O1', 'O2'])]
    central_channels = [ch for ch in ch_names if any(x in ch.upper() for x in ['CZ', 'C3', 'C4'])]

    # Frequency bands
    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (13, 30),
        'low_alpha': (8, 10),
        'high_alpha': (10, 12),
        'gamma': (30, 45)
    }

    results = {'epoch_times': [], 'metrics': {}}

    for epoch_idx in range(n_epochs):
        epoch_time = (start_idx / sfreq) if epoch_idx == 0 else (epoch_idx * (n_samples - overlap_samples) / sfreq)
        results['epoch_times'].append(epoch_time)

        epoch = epochs_data[epoch_idx]

        # Compute power spectral density for each channel
        freqs, psds = signal.welch(epoch, fs=sfreq, nperseg=int(sfreq*2), axis=-1)

        # Helper function to get power in frequency band
        def get_band_power(psd, freqs, band):
            freq_mask = (freqs >= band[0]) & (freqs <= band[1])
            return np.mean(psd[:, freq_mask], axis=1)

        epoch_metrics = {}

        # 1. Basic spectral power metrics
        for band_name, band_range in freq_bands.items():
            power = get_band_power(psds, freqs, band_range)
            epoch_metrics[f'{band_name}_power_global'] = np.mean(power)

            # Region-specific power
            if frontal_channels:
                frontal_idx = [ch_names.index(ch) for ch in frontal_channels if ch in ch_names]
                epoch_metrics[f'{band_name}_power_frontal'] = np.mean(power[frontal_idx])

            if parietal_channels:
                parietal_idx = [ch_names.index(ch) for ch in parietal_channels if ch in ch_names]
                epoch_metrics[f'{band_name}_power_parietal'] = np.mean(power[parietal_idx])

            if occipital_channels:
                occipital_idx = [ch_names.index(ch) for ch in occipital_channels if ch in ch_names]
                epoch_metrics[f'{band_name}_power_occipital'] = np.mean(power[occipital_idx])

        # 2. Key fatigue ratios
        theta_power = get_band_power(psds, freqs, freq_bands['theta'])
        alpha_power = get_band_power(psds, freqs, freq_bands['alpha'])
        beta_power = get_band_power(psds, freqs, freq_bands['beta'])
        delta_power = get_band_power(psds, freqs, freq_bands['delta'])

        # Theta/Alpha ratio (most important fatigue metric)
        epoch_metrics['theta_alpha_ratio_global'] = np.mean(theta_power / (alpha_power + 1e-10))
        epoch_metrics['alpha_theta_ratio_global'] = np.mean(alpha_power / (theta_power + 1e-10))

        # Region-specific ratios
        if frontal_channels and parietal_channels:
            frontal_idx = [ch_names.index(ch) for ch in frontal_channels if ch in ch_names]
            parietal_idx = [ch_names.index(ch) for ch in parietal_channels if ch in ch_names]

            theta_frontal = np.mean(theta_power[frontal_idx])
            alpha_parietal = np.mean(alpha_power[parietal_idx])

            epoch_metrics['theta_alpha_ratio_regions'] = theta_frontal / (alpha_parietal + 1e-10)
            epoch_metrics['alpha_theta_ratio_regions'] = alpha_parietal / (theta_frontal + 1e-10)

        # 3. Engagement Index: Beta/(Alpha+Theta)
        epoch_metrics['engagement_index'] = np.mean(beta_power / (alpha_power + theta_power + 1e-10))

        # 4. Theta/Beta ratio
        epoch_metrics['theta_beta_ratio'] = np.mean(theta_power / (beta_power + 1e-10))

        # 5. Alpha peak frequency (Individual Alpha Frequency - IAF)
        if occipital_channels:
            occipital_idx = [ch_names.index(ch) for ch in occipital_channels if ch in ch_names]
            alpha_mask = (freqs >= 8) & (freqs <= 12)
            alpha_freqs = freqs[alpha_mask]
            alpha_psds = psds[occipital_idx][:, alpha_mask]

            # Find peak frequency for each occipital channel
            peak_freqs = []
            for ch_idx in range(len(occipital_idx)):
                peak_idx = np.argmax(alpha_psds[ch_idx])
                peak_freqs.append(alpha_freqs[peak_idx])

            epoch_metrics['individual_alpha_frequency'] = np.mean(peak_freqs)

        # 6. Low Alpha/High Alpha ratio
        low_alpha_power = get_band_power(psds, freqs, freq_bands['low_alpha'])
        high_alpha_power = get_band_power(psds, freqs, freq_bands['high_alpha'])
        epoch_metrics['low_high_alpha_ratio'] = np.mean(low_alpha_power / (high_alpha_power + 1e-10))

        # 7. Frontal Midline Theta (specific 6-7 Hz band)
        fmt_mask = (freqs >= 6) & (freqs <= 7)
        if frontal_channels:
            frontal_idx = [ch_names.index(ch) for ch in frontal_channels if ch in ch_names]
            fmt_power = np.mean(psds[frontal_idx][:, fmt_mask])
            epoch_metrics['frontal_midline_theta'] = fmt_power

        # 8. Spectral Entropy (complexity measure)
        def spectral_entropy(psd, freqs):
            # Normalize PSD
            psd_norm = psd / np.sum(psd, axis=1, keepdims=True)
            # Compute entropy
            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-15), axis=1)
            return np.mean(entropy)

        epoch_metrics['spectral_entropy'] = spectral_entropy(psds, freqs)

        # 9. Alpha asymmetry (hemispheric differences)
        left_channels = [ch for ch in ch_names if any(x in ch.upper() for x in ['F3', 'C3', 'P3', 'O1'])]
        right_channels = [ch for ch in ch_names if any(x in ch.upper() for x in ['F4', 'C4', 'P4', 'O2'])]

        if left_channels and right_channels:
            left_idx = [ch_names.index(ch) for ch in left_channels if ch in ch_names]
            right_idx = [ch_names.index(ch) for ch in right_channels if ch in ch_names]

            alpha_left = np.mean(alpha_power[left_idx])
            alpha_right = np.mean(alpha_power[right_idx])

            epoch_metrics['alpha_asymmetry'] = (alpha_right - alpha_left) / (alpha_right + alpha_left + 1e-10)

        # 10. Relative power (normalized by total power)
        total_power = get_band_power(psds, freqs, (0.5, 45))
        for band_name in ['delta', 'theta', 'alpha', 'beta']:
            band_power = get_band_power(psds, freqs, freq_bands[band_name])
            epoch_metrics[f'{band_name}_relative_power'] = np.mean(band_power / (total_power + 1e-10))

        # Store epoch metrics
        for metric_name, value in epoch_metrics.items():
            if metric_name not in results['metrics']:
                results['metrics'][metric_name] = []
            results['metrics'][metric_name].append(value)

    return results

def analyze_fatigue_trends(metrics_dict, time_points):
    """
    Analyze trends in fatigue metrics over time

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of computed metrics from compute_fatigue_metrics
    time_points : list
        Time points corresponding to each epoch

    Returns:
    --------
    dict : Statistical analysis of trends
    """
    trends = {}

    for metric_name, values in metrics_dict.items():
        if len(values) > 1:
            # Compute linear trend
            correlation, p_value = pearsonr(time_points, values)

            # Compute relative change
            initial_value = np.mean(values[:3]) if len(values) >= 3 else values[0]
            final_value = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
            relative_change = (final_value - initial_value) / (initial_value + 1e-10) * 100

            trends[metric_name] = {
                'correlation_with_time': correlation,
                'trend_p_value': p_value,
                'relative_change_percent': relative_change,
                'initial_value': initial_value,
                'final_value': final_value,
                'mean_value': np.mean(values),
                'std_value': np.std(values)
            }

    return trends




# Import our fatigue analysis functions
# from eeg_fatigue_analysis import compute_fatigue_metrics, analyze_fatigue_trends

def compare_multiple_recordings(raw_objects, labels, analysis_params=None):
    """
    Compare fatigue metrics across multiple EEG recordings

    Parameters:
    -----------
    raw_objects : list of mne.Raw
        List of MNE Raw objects to analyze
    labels : list of str
        Labels for each recording (e.g., ['Baseline', 'After 4h', 'After 8h'])
    analysis_params : dict
        Analysis parameters (duration, overlap, etc.)

    Returns:
    --------
    dict : Comprehensive comparison results
    """

    if analysis_params is None:
        analysis_params = {
            'duration': 60,  # 60-second epochs
            'overlap': 0.5,  # 50% overlap
            'min_epochs': 5  # Minimum epochs required
        }

    results = {
        'recordings': {},
        'comparison': {},
        'summary_stats': {}
    }

    # Analyze each recording
    print("Analyzing individual recordings...")
    for i, (raw, label) in enumerate(zip(raw_objects, labels)):
        print(f"  Processing {label}...")

        # Apply consistent preprocessing
        raw_processed = raw.copy()
        raw_processed.filter(l_freq=0.5, h_freq=45, verbose=False)
        raw_processed.notch_filter(freqs=60, verbose=False)  # or 60 Hz for US

        # Compute fatigue metrics
        metrics = compute_fatigue_metrics(
            raw_processed, 
            duration=analysis_params['duration'],
            overlap=analysis_params['overlap']
        )

        # Analyze trends within this recording
        trends = analyze_fatigue_trends(metrics['metrics'], metrics['epoch_times'])

        results['recordings'][label] = {
            'metrics': metrics,
            'trends': trends,
            'n_epochs': len(metrics['epoch_times']),
            'duration_minutes': max(metrics['epoch_times']) / 60
        }

    # Cross-recording comparisons
    print("Performing cross-recording comparisons...")
    results['comparison'] = compare_recordings(results['recordings'], labels)

    # Generate summary statistics
    results['summary_stats'] = generate_summary_stats(results['recordings'], labels)

    return results

def compare_recordings(recordings_data, labels):
    """Compare key metrics between recordings"""

    comparison_results = {}

    # Key fatigue metrics to focus on
    key_metrics = [
        'theta_alpha_ratio_global',
        'engagement_index', 
        'individual_alpha_frequency',
        'theta_power_frontal',
        'alpha_power_occipital',
        'spectral_entropy',
        'frontal_midline_theta'
    ]

    # Extract mean values for each metric and recording
    metric_means = {}
    for metric in key_metrics:
        metric_means[metric] = {}
        for label in labels:
            if label in recordings_data and metric in recordings_data[label]['metrics']['metrics']:
                values = recordings_data[label]['metrics']['metrics'][metric]
                metric_means[metric][label] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'n': len(values)
                }

    comparison_results['metric_means'] = metric_means

    # Statistical comparisons (if we have baseline and other conditions)
    if len(labels) >= 2 and 'baseline' in [l.lower() for l in labels]:
        baseline_idx = [l.lower() for l in labels].index('baseline')
        baseline_label = labels[baseline_idx]

        statistical_comparisons = {}
        for metric in key_metrics:
            statistical_comparisons[metric] = {}

            if metric in metric_means and baseline_label in metric_means[metric]:
                baseline_values = recordings_data[baseline_label]['metrics']['metrics'].get(metric, [])

                for label in labels:
                    if label != baseline_label and metric in metric_means and label in metric_means[metric]:
                        comparison_values = recordings_data[label]['metrics']['metrics'].get(metric, [])

                        if len(baseline_values) > 0 and len(comparison_values) > 0:
                            # Paired t-test (if same number of epochs) or independent t-test
                            try:
                                if len(baseline_values) == len(comparison_values):
                                    t_stat, p_val = ttest_rel(comparison_values, baseline_values)
                                    test_type = 'paired'
                                else:
                                    from scipy.stats import ttest_ind
                                    t_stat, p_val = ttest_ind(comparison_values, baseline_values)
                                    test_type = 'independent'

                                effect_size = (np.mean(comparison_values) - np.mean(baseline_values)) / np.std(baseline_values)

                                statistical_comparisons[metric][f'{label}_vs_{baseline_label}'] = {
                                    't_statistic': t_stat,
                                    'p_value': p_val,
                                    'effect_size': effect_size,
                                    'test_type': test_type,
                                    'significant': p_val < 0.05,
                                    'direction': 'increase' if effect_size > 0 else 'decrease'
                                }
                            except Exception as e:
                                print(f"Statistical comparison failed for {metric}: {e}")

        comparison_results['statistical_tests'] = statistical_comparisons

    return comparison_results

def generate_summary_stats(recordings_data, labels):
    """Generate summary statistics across recordings"""

    summary = {
        'recording_info': {},
        'fatigue_indicators': {},
        'recommendations': []
    }

    # Basic recording information
    for label in labels:
        if label in recordings_data:
            data = recordings_data[label]
            summary['recording_info'][label] = {
                'duration_minutes': data['duration_minutes'],
                'n_epochs': data['n_epochs'],
                'avg_epoch_interval': data['duration_minutes'] * 60 / data['n_epochs'] if data['n_epochs'] > 0 else 0
            }

    # Identify strongest fatigue indicators
    key_metrics = ['theta_alpha_ratio_global', 'engagement_index', 'individual_alpha_frequency']

    for metric in key_metrics:
        metric_summary = {}
        for label in labels:
            if (label in recordings_data and 
                metric in recordings_data[label]['trends']):
                trend_data = recordings_data[label]['trends'][metric]
                metric_summary[label] = {
                    'relative_change': trend_data['relative_change_percent'],
                    'correlation_with_time': trend_data['correlation_with_time'],
                    'p_value': trend_data['trend_p_value'],
                    'mean_value': trend_data['mean_value']
                }

        summary['fatigue_indicators'][metric] = metric_summary

    # Generate recommendations
    summary['recommendations'] = generate_recommendations(summary)

    return summary

def generate_recommendations(summary_stats):
    """Generate practical recommendations based on analysis results"""

    recommendations = []

    # Check for strong fatigue indicators
    fatigue_indicators = summary_stats['fatigue_indicators']

    # Theta/Alpha ratio recommendations
    if 'theta_alpha_ratio_global' in fatigue_indicators:
        theta_alpha_data = fatigue_indicators['theta_alpha_ratio_global']
        max_increase = max([data.get('relative_change', 0) for data in theta_alpha_data.values()])

        if max_increase > 50:
            recommendations.append({
                'metric': 'Theta/Alpha Ratio',
                'finding': f'High increase ({max_increase:.1f}%)',
                'interpretation': 'Strong indication of progressive fatigue',
                'recommendation': 'Monitor theta/alpha ratio as primary fatigue indicator'
            })
        elif max_increase > 20:
            recommendations.append({
                'metric': 'Theta/Alpha Ratio', 
                'finding': f'Moderate increase ({max_increase:.1f}%)',
                'interpretation': 'Mild to moderate fatigue detected',
                'recommendation': 'Consider additional fatigue metrics for confirmation'
            })

    # Engagement index recommendations
    if 'engagement_index' in fatigue_indicators:
        engagement_data = fatigue_indicators['engagement_index']
        max_decrease = min([data.get('relative_change', 0) for data in engagement_data.values()])

        if max_decrease < -30:
            recommendations.append({
                'metric': 'Engagement Index',
                'finding': f'Substantial decrease ({abs(max_decrease):.1f}%)',
                'interpretation': 'Significant reduction in cognitive engagement',
                'recommendation': 'Use engagement index for real-time alertness monitoring'
            })

    # Alpha frequency recommendations
    if 'individual_alpha_frequency' in fatigue_indicators:
        iaf_data = fatigue_indicators['individual_alpha_frequency']
        frequency_changes = [data.get('relative_change', 0) for data in iaf_data.values()]
        avg_change = np.mean(frequency_changes)

        if avg_change < -5:
            recommendations.append({
                'metric': 'Individual Alpha Frequency',
                'finding': f'Frequency slowing ({abs(avg_change):.1f}%)',
                'interpretation': 'Alpha rhythm slowing indicates neural fatigue',
                'recommendation': 'Track individual alpha frequency for personalized fatigue detection'
            })

    # General recommendations
    if len(recommendations) == 0:
        recommendations.append({
            'metric': 'Overall Assessment',
            'finding': 'Minimal fatigue indicators detected',
            'interpretation': 'Low fatigue levels across recordings',
            'recommendation': 'Continue monitoring with current metrics'
        })

    return recommendations

def visualize_fatigue_comparison(results, save_plots=True):
    """Create comprehensive visualization of fatigue analysis results"""
    import seaborn as sns
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    recordings_data = results['recordings']
    labels = list(recordings_data.keys())

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Key metrics to visualize
    key_metrics = [
        ('theta_alpha_ratio_global', 'Theta/Alpha Ratio', 'Higher = More Fatigue'),
        ('engagement_index', 'Engagement Index', 'Lower = Less Engaged'), 
        ('individual_alpha_frequency', 'Alpha Peak Frequency (Hz)', 'Lower = More Fatigue'),
        ('spectral_entropy', 'Spectral Entropy', 'Lower = Less Complex')
    ]

    # 1. Time course plots for each recording
    for i, (metric_key, metric_name, interpretation) in enumerate(key_metrics):
        ax = plt.subplot(2, 2, i+1)

        for label in labels:
            if (label in recordings_data and 
                metric_key in recordings_data[label]['metrics']['metrics']):

                times = recordings_data[label]['metrics']['epoch_times']
                values = recordings_data[label]['metrics']['metrics'][metric_key]

                plt.plot(np.array(times)/60, values, 'o-', label=label, alpha=0.7, linewidth=2)

        plt.xlabel('Time (minutes)')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name}\n({interpretation})')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plots:
        plt.savefig('fatigue_comparison_timecourses.png', dpi=300, bbox_inches='tight')
        print("Time course plots saved to 'fatigue_comparison_timecourses.png'")

    # 2. Summary comparison plot
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract summary data for plotting
    summary_data = []
    for label in labels:
        if label in recordings_data:
            row_data = {'Recording': label}
            for metric_key, _, _ in key_metrics:
                if metric_key in recordings_data[label]['trends']:
                    trend_data = recordings_data[label]['trends'][metric_key]
                    row_data[f'{metric_key}_change'] = trend_data['relative_change_percent']
                    row_data[f'{metric_key}_mean'] = trend_data['mean_value']
            summary_data.append(row_data)

    summary_df = pd.DataFrame(summary_data)

    # Plot relative changes
    if len(summary_df) > 0:
        change_cols = [col for col in summary_df.columns if '_change' in col]
        if change_cols:
            change_data = summary_df[['Recording'] + change_cols].set_index('Recording')
            change_data.columns = [col.replace('_change', '').replace('_global', '') for col in change_data.columns]

            ax = axes[0, 0]
            change_data.T.plot(kind='bar', ax=ax, rot=45)
            ax.set_title('Relative Change Over Time (%)')
            ax.set_ylabel('Percent Change')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_plots:
        plt.savefig('fatigue_comparison_summary.png', dpi=300, bbox_inches='tight')
        print("Summary comparison plots saved to 'fatigue_comparison_summary.png'")

    plt.show()

def print_analysis_report(results):
    """Print a comprehensive analysis report"""

    print("\n" + "="*80)
    print("                    EEG FATIGUE ANALYSIS REPORT")
    print("="*80)

    recordings_data = results['recordings']
    comparison_data = results['comparison']
    summary_stats = results['summary_stats']

    # Recording overview
    print("\n📊 RECORDING OVERVIEW")
    print("-" * 40)
    for label, info in summary_stats['recording_info'].items():
        print(f"{label}:")
        print(f"  Duration: {info['duration_minutes']:.1f} minutes")
        print(f"  Epochs: {info['n_epochs']}")
        print(f"  Avg interval: {info['avg_epoch_interval']:.1f} seconds")

    # Key findings
    print("\n🔍 KEY FATIGUE INDICATORS")
    print("-" * 40)

    for metric, recordings in summary_stats['fatigue_indicators'].items():
        metric_display = metric.replace('_', ' ').replace('global', '').title()
        print(f"\n{metric_display}:")

        for recording, data in recordings.items():
            change = data['relative_change']
            p_val = data.get('p_value', 1.0)
            significance = "**" if p_val < 0.05 else ""
            direction = "↑" if change > 0 else "↓"

            print(f"  {recording}: {direction} {abs(change):.1f}% change {significance}")

    # Statistical comparisons
    if 'statistical_tests' in comparison_data:
        print("\n📈 STATISTICAL COMPARISONS")
        print("-" * 40)

        for metric, comparisons in comparison_data['statistical_tests'].items():
            if comparisons:  # Only show metrics with comparisons
                metric_display = metric.replace('_', ' ').replace('global', '').title()
                print(f"\n{metric_display}:")

                for comparison_name, stats in comparisons.items():
                    p_val = stats['p_value']
                    effect_size = stats['effect_size']
                    direction = stats['direction']

                    significance = ""
                    if p_val < 0.001:
                        significance = "***"
                    elif p_val < 0.01:
                        significance = "**" 
                    elif p_val < 0.05:
                        significance = "*"

                    print(f"  {comparison_name}: {direction} (p={p_val:.3f}{significance}, d={effect_size:.2f})")

    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)

    for i, rec in enumerate(summary_stats['recommendations'], 1):
        print(f"\n{i}. {rec['metric']}")
        print(f"   Finding: {rec['finding']}")
        print(f"   Interpretation: {rec['interpretation']}")
        print(f"   Recommendation: {rec['recommendation']}")

    print("\n" + "="*80)

# Example usage function
def example_analysis():
    """
    Example of how to use the fatigue analysis framework

    Replace the file loading section with your actual MNE Raw objects
    """

    print("EEG Fatigue Analysis Framework - Example Usage")
    print("=" * 60)

    # Example file paths - replace with your actual files
    example_files = [
        'baseline_recording.fif',
        'after_4h_recording.fif', 
        'after_8h_recording.fif'
    ]

    labels = ['Baseline', 'After 4h', 'After 8h']

    # Load your MNE Raw objects here
    # This is where you would load your actual data:

    print("\nTo use this framework with your data:")
    print("1. Load your MNE Raw objects:")
    print("   raw_objects = [")
    print("       mne.io.read_raw_fif('recording1.fif', preload=True),")
    print("       mne.io.read_raw_fif('recording2.fif', preload=True),")
    print("       mne.io.read_raw_fif('recording3.fif', preload=True)")
    print("   ]")
    print("\n2. Define labels for your recordings:")
    print("   labels = ['Baseline', 'Fatigue Condition 1', 'Fatigue Condition 2']")
    print("\n3. Run the analysis:")
    print("   results = compare_multiple_recordings(raw_objects, labels)")
    print("   print_analysis_report(results)")
    print("   visualize_fatigue_comparison(results)")

    # If you have actual data files, uncomment and modify this:
    """
    try:
        raw_objects = []
        for file_path in example_files:
            if Path(file_path).exists():
                raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
                raw_objects.append(raw)

        if len(raw_objects) >= 2:
            results = compare_multiple_recordings(raw_objects, labels[:len(raw_objects)])
            print_analysis_report(results)
            visualize_fatigue_comparison(results)

            return results
        else:
            print("Not enough data files found for comparison.")

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please check your file paths and make sure MNE can read the files.")
    """

if __name__ == "__main__":
    # Example usage:
    """
    # Load your MNE Raw object
    raw = mne.io.read_raw_fif('your_eeg_file.fif', preload=True)

    # Apply basic preprocessing
    raw.filter(l_freq=0.5, h_freq=45)  # Bandpass filter
    raw.notch_filter(freqs=50)  # Remove line noise

    # Compute fatigue metrics
    results = compute_fatigue_metrics(raw, duration=60, overlap=0.5)

    # Analyze trends
    trends = analyze_fatigue_trends(results['metrics'], results['epoch_times'])

    # Convert to DataFrame for easy analysis
    df_metrics = pd.DataFrame(results['metrics'])
    df_metrics['time'] = results['epoch_times']

    # Print key fatigue indicators
    key_metrics = [
        'theta_alpha_ratio_global',
        'engagement_index',
        'individual_alpha_frequency',
        'frontal_midline_theta',
        'spectral_entropy'
    ]

    print("Key Fatigue Metrics Trends:")
    for metric in key_metrics:
        if metric in trends:
            trend = trends[metric]
            direction = "↑" if trend['correlation_with_time'] > 0 else "↓"
            significance = "**" if trend['trend_p_value'] < 0.05 else ""
            print(f"{metric}: {direction} {trend['relative_change_percent']:.1f}% change {significance}")
    """


    """
    Practical Example: EEG Fatigue Analysis with Multiple MNE Raw Objects
    ====================================================================

    This script demonstrates how to analyze fatigue-related changes across 
    multiple EEG recordings using the comprehensive fatigue metrics.

    Author: EEG Fatigue Analysis Framework
    Usage: python fatigue_comparison_analysis.py
    """


    example_analysis()
