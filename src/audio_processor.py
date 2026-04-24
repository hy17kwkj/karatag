"""
音声読み込みと前処理。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import librosa


class AudioProcessor:
    """音声ファイルを読み込み、アライメントに適した形式（16kHz モノラル）に変換します。"""

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def load(self, path: str | Path) -> np.ndarray:
        """
        音声ファイルを読み込みます。librosaを使用して、自動的に指定されたサンプリングレートに
        リサンプルし、モノラルに変換します。
        """
        # librosa.load は内部で ffmpeg や audioread を使用し、
        # 浮動小数点数（-1.0〜1.0）の numpy 配列を返します。
        audio, _ = librosa.load(str(path), sr=self.sample_rate, mono=True)
        return audio

    def get_duration(self, audio: np.ndarray) -> float:
        """音声データの長さを秒単位で返します。"""
        return len(audio) / self.sample_rate
