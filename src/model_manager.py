"""
HuggingFace からの学習済みモデルのダウンロードと管理。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


class ModelManager:
    """
    Whisper および Wav2Vec2 モデルのダウンロードとキャッシュディレクトリの管理を行います。
    """

    def __init__(self, cache_dir: str | Path = "models") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # HuggingFace のキャッシュディレクトリ環境変数を設定
        os.environ["HF_HOME"] = str(self.cache_dir)

    def download_all(self, progress_cb: Callable[[str], None] | None = None) -> None:
        """必要なすべてのモデル（Whisper, Wav2Vec2）をダウンロードします。"""
        
        # 1. Wav2Vec2 (Japanese)
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
        if progress_cb: progress_cb(f"Wav2Vec2 モデルをダウンロード中: {model_name}")
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        Wav2Vec2Processor.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=str(self.cache_dir))

        # 2. Faster-Whisper (Medium)
        whisper_model = "medium"
        if progress_cb: progress_cb(f"Whisper モデルをダウンロード中: {whisper_model}")
        from faster_whisper import WhisperModel
        # WhisperModel の初期化時に自動的にダウンロードされる
        WhisperModel(whisper_model, device="cpu", compute_type="int8", download_root=str(self.cache_dir))

    def get_model_path(self, model_id: str) -> Path:
        """特定のモデルのキャッシュパスを取得します（デバッグ用）。"""
        return self.cache_dir
