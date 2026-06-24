"""Audio analysis and visualization for accent training.

Provides spectrogram, waveform, and pitch contour visualizations
for comparing ideal pronunciation against user attempts.
"""

import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np

# Dark theme for all plots
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eaeaea',
    'text.color': '#eaeaea',
    'xtick.color': '#aaaaaa',
    'ytick.color': '#aaaaaa',
    'grid.color': '#2a2a4a',
    'font.family': 'sans-serif',
})


def create_spectrogram(audio_path, title="Spectrogram"):
    """Create a mel spectrogram visualization.

    Args:
        audio_path: Path to audio file.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure
    """
    if audio_path is None:
        return _empty_figure("No audio provided")

    try:
        y, sr = librosa.load(audio_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        img = librosa.display.specshow(
            S_dB, sr=sr, x_axis='time', y_axis='mel',
            ax=ax, cmap='magma'
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
        ax.set_title(title, fontsize=14, fontweight='bold', color='#e94560')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        fig.tight_layout()
        return fig
    except Exception as e:
        return _empty_figure(f"Error: {str(e)}")


def create_waveform(audio_path, title="Waveform"):
    """Create a waveform visualization.

    Args:
        audio_path: Path to audio file.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure
    """
    if audio_path is None:
        return _empty_figure("No audio provided")

    try:
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        time_axis = np.linspace(0, duration, len(y))

        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(time_axis, y, color='#0f3460', linewidth=0.5, alpha=0.8)
        ax.fill_between(time_axis, y, alpha=0.3, color='#e94560')
        ax.set_title(title, fontsize=14, fontweight='bold', color='#e94560')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_xlim(0, duration)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    except Exception as e:
        return _empty_figure(f"Error: {str(e)}")


def create_pitch_contour(audio_path, title="Pitch Contour"):
    """Create a pitch (F0) contour visualization.

    Args:
        audio_path: Path to audio file.
        title: Plot title.

    Returns:
        matplotlib.figure.Figure
    """
    if audio_path is None:
        return _empty_figure("No audio provided")

    try:
        y, sr = librosa.load(audio_path, sr=22050)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        times = librosa.times_like(f0, sr=sr)

        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(times, f0, color='#e94560', linewidth=2, label='F0')
        ax.set_title(title, fontsize=14, fontweight='bold', color='#e94560')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception as e:
        return _empty_figure(f"Error: {str(e)}")


def create_comparison(ideal_path, attempt_path):
    """Create a full side-by-side comparison visualization.

    Generates a 3x2 grid:
    - Row 1: Mel spectrograms (ideal vs attempt)
    - Row 2: Waveforms (ideal vs attempt)
    - Row 3: Pitch contours (ideal vs attempt)

    Args:
        ideal_path: Path to ideal (VC'd) audio.
        attempt_path: Path to user's attempt audio.

    Returns:
        matplotlib.figure.Figure
    """
    if ideal_path is None and attempt_path is None:
        return _empty_figure("No audio to compare")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Pronunciation Comparison', fontsize=18,
                 fontweight='bold', color='#e94560', y=0.98)

    # Column labels
    axes[0, 0].set_title('🎯 Ideal Pronunciation', fontsize=13,
                          fontweight='bold', color='#53d769')
    axes[0, 1].set_title('🎤 Your Attempt', fontsize=13,
                          fontweight='bold', color='#5ac8fa')

    for col, (audio_path, label) in enumerate([
        (ideal_path, 'Ideal'),
        (attempt_path, 'Attempt')
    ]):
        if audio_path is None:
            for row in range(3):
                axes[row, col].text(
                    0.5, 0.5, f'No {label.lower()} audio',
                    ha='center', va='center', fontsize=12,
                    color='#666666', transform=axes[row, col].transAxes
                )
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue

        try:
            y, sr = librosa.load(audio_path, sr=22050)
            duration = len(y) / sr

            # Row 0: Mel Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(
                S_dB, sr=sr, x_axis='time', y_axis='mel',
                ax=axes[0, col], cmap='magma'
            )
            axes[0, col].set_ylabel('Freq (Hz)' if col == 0 else '')

            # Row 1: Waveform
            time_axis = np.linspace(0, duration, len(y))
            color = '#53d769' if col == 0 else '#5ac8fa'
            axes[1, col].plot(time_axis, y, color=color, linewidth=0.5, alpha=0.8)
            axes[1, col].fill_between(time_axis, y, alpha=0.2, color=color)
            axes[1, col].set_ylabel('Amplitude' if col == 0 else '')
            axes[1, col].set_xlim(0, duration)
            axes[1, col].grid(True, alpha=0.2)

            # Row 2: Pitch Contour
            f0, _, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'), sr=sr
            )
            times = librosa.times_like(f0, sr=sr)
            axes[2, col].plot(times, f0, color=color, linewidth=2)
            axes[2, col].set_ylabel('F0 (Hz)' if col == 0 else '')
            axes[2, col].set_xlabel('Time (s)')
            axes[2, col].grid(True, alpha=0.2)

        except Exception as e:
            for row in range(3):
                axes[row, col].text(
                    0.5, 0.5, f'Error: {str(e)[:50]}',
                    ha='center', va='center', fontsize=10,
                    color='#e94560', transform=axes[row, col].transAxes
                )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _empty_figure(message):
    """Create an empty figure with a message."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.text(
        0.5, 0.5, message,
        ha='center', va='center', fontsize=14,
        color='#666666', transform=ax.transAxes
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig
