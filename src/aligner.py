"""
コア・強制アライメントエンジン (V132 - 精度改善版).

改善点 (review.md Phase 0-2):
  Phase 0: alignment_mode (linear/hybrid/ctc) を分配ロジックに実際に接続
  Phase 1: アンカー取得率改善（動的ウィンドウ・動的閾値再試行・word_timestamps）
  Phase 2: モーラ重み付き分配・母音ピーク基準オンセット・パーセンタイル正規化
           非対称 search_margin・サンプル精度クロップ
"""

from __future__ import annotations

import datetime
import configparser
import difflib
import re
from pathlib import Path
from typing import Callable

import numpy as np
import librosa
import torch

torch.set_num_threads(4)

from .text_processor import TextProcessor

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
DEFAULT_W2V2_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
_NAN = float("nan")

# モーラ重み: 均等分配の代わりに実際の音節長に近い比率で分配する
MORA_WEIGHT: dict[str, float] = {
    "っ": 0.3, "ッ": 0.3,          # 促音: 閉鎖のみで極短い
    "ー": 1.2,                      # 長音: 前母音の延長
    "ぁ": 0.6, "ぃ": 0.6, "ぅ": 0.6, "ぇ": 0.6, "ぉ": 0.6,
    "ゃ": 0.6, "ゅ": 0.6, "ょ": 0.6,
    "ァ": 0.6, "ィ": 0.6, "ゥ": 0.6, "ェ": 0.6, "ォ": 0.6,
    "ャ": 0.6, "ュ": 0.6, "ョ": 0.6,
}

def _is_nan(v: float) -> bool:
    return v != v

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

class Settings:
    def __init__(self):
        self.min_gap = 0.01
        self.max_char_duration = 1.2
        self.min_char_duration = 0.03
        self.search_margin = 1.5
        self.anchor_score = 0.85
        self.alignment_mode = "hybrid"
        self.auto_interlude_duration = 10.0
        self.auto_interlude_threshold = 0.02
        self.min_vocal_density = 0.3
        self.load()

    def load(self, preset_name: str | None = None):
        path = Path("settings.ini")
        if not path.exists():
            return
        config = configparser.ConfigParser()
        try:
            content = "[DEFAULT]\n" + path.read_text(encoding="utf-8")
            config.read_string(content)

            def get_safe_float(section, key, curr):
                try:
                    return config.getfloat(section, key, fallback=curr)
                except Exception:
                    return curr

            self.min_gap = get_safe_float("DEFAULT", "min_gap", self.min_gap)
            self.max_char_duration = get_safe_float("DEFAULT", "max_char_duration", self.max_char_duration)
            self.min_char_duration = get_safe_float("DEFAULT", "min_char_duration", self.min_char_duration)
            self.search_margin = get_safe_float("DEFAULT", "search_margin", self.search_margin)
            self.anchor_score = get_safe_float("DEFAULT", "anchor_score", self.anchor_score)
            self.auto_interlude_duration = get_safe_float("DEFAULT", "auto_interlude_duration", self.auto_interlude_duration)
            self.auto_interlude_threshold = get_safe_float("DEFAULT", "auto_interlude_threshold", self.auto_interlude_threshold)
            self.min_vocal_density = get_safe_float("DEFAULT", "min_vocal_density", self.min_vocal_density)
            self.alignment_mode = config.get("DEFAULT", "alignment_mode", fallback=self.alignment_mode)

            if preset_name:
                section = f"preset:{preset_name}"
                if section in config.sections():
                    self.min_gap = get_safe_float(section, "min_gap", self.min_gap)
                    self.max_char_duration = get_safe_float(section, "max_char_duration", self.max_char_duration)
                    self.min_char_duration = get_safe_float(section, "min_char_duration", self.min_char_duration)
                    self.search_margin = get_safe_float(section, "search_margin", self.search_margin)
                    self.anchor_score = get_safe_float(section, "anchor_score", self.anchor_score)
                    self.auto_interlude_duration = get_safe_float(section, "auto_interlude_duration", self.auto_interlude_duration)
                    self.auto_interlude_threshold = get_safe_float(section, "auto_interlude_threshold", self.auto_interlude_threshold)
                    self.min_vocal_density = get_safe_float(section, "min_vocal_density", self.min_vocal_density)
                    self.alignment_mode = config.get(section, "alignment_mode", fallback=self.alignment_mode)
        except Exception as e:
            log_debug(f"設定エラー: {e}")

# ---------------------------------------------------------------------------
# ロギング
# ---------------------------------------------------------------------------

_LOG_INITIALIZED = False

def log_debug(msg: str):
    global _LOG_INITIALIZED
    mode = "a" if _LOG_INITIALIZED else "w"
    with open("karatag_debug.log", mode, encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}\n")
    _LOG_INITIALIZED = True

# ---------------------------------------------------------------------------
# AI エンジン
# ---------------------------------------------------------------------------

class _Wav2Vec2Engine:
    def __init__(self, model_name: str, cache_dir: str | Path) -> None:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        vocab = self.processor.tokenizer.get_vocab()
        self.char_list = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
        self.char_to_id = {c: i for i, c in enumerate(self.char_list)}

    def get_log_probs(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < 400:
            return np.array([])
        with torch.no_grad():
            # ピーク正規化ではなくパーセンタイル正規化: 歌声ステムのダイナミックレンジに安定
            p99 = np.percentile(np.abs(audio), 99)
            norm = np.clip(audio / (p99 + 1e-6), -1.0, 1.0)
            inputs = self.processor(norm, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        return lp

# ---------------------------------------------------------------------------
# オーケストレーター
# ---------------------------------------------------------------------------

class LyricsAligner:
    def __init__(self, cache_dir: str | Path = "models", w2v2_model: str = DEFAULT_W2V2_MODEL, whisper_model: str = "medium") -> None:
        self.cache_dir = Path(cache_dir)
        self.whisper_name = whisper_model
        self.w2v2_name = w2v2_model
        self.text_proc = TextProcessor()
        self.settings = Settings()
        self._ai: _Wav2Vec2Engine | None = None
        self._whisper = None

    def load_models(self, progress_cb: Callable[[str], None] | None = None) -> None:
        if progress_cb:
            progress_cb("AIエンジン起動中...")
        from faster_whisper import WhisperModel
        self._ai = _Wav2Vec2Engine(self.w2v2_name, self.cache_dir)
        self._whisper = WhisperModel(
            self.whisper_name, device="cpu", compute_type="int8",
            download_root=str(self.cache_dir), cpu_threads=4,
        )

    def run(self, audio: np.ndarray, lyrics_text: str, progress_cb: Callable[[int, int], None] | None = None) -> list[list[tuple[str, float, float]]]:
        if not self._whisper or not self._ai:
            raise RuntimeError("load_models() required")
        self.settings.load()
        has_tags = bool(re.search(r"\[\d+:\d+", lyrics_text))

        if not has_tags:
            log_debug("--- 広域アライメント開始 (Monolithic) ---")
            from . import aligner_legacy
            legacy = aligner_legacy.LyricsAligner(self.cache_dir, self.w2v2_name, self.whisper_name)
            legacy._ai = self._ai
            legacy._whisper = self._whisper
            return legacy.run(audio, lyrics_text, progress_cb)
        else:
            log_debug("--- 分割アライメント開始 (Partitioned) ---")
            return self._run_partitioned(audio, lyrics_text, progress_cb)

    # ------------------------------------------------------------------
    # 分割アライメント
    # ------------------------------------------------------------------

    def _run_partitioned(self, audio: np.ndarray, lyrics_text: str, progress_cb: Callable[[int, int], None] | None = None) -> list[list[tuple[str, float, float]]]:
        segments = self.text_proc.parse_lyrics_file(lyrics_text)
        total_duration = len(audio) / SAMPLE_RATE

        # 50fps 特徴量 (バリデーション・分配用)
        rms_full = librosa.feature.rms(y=audio, frame_length=640, hop_length=320)[0]
        rms_norm_full = rms_full / (np.max(rms_full) + 1e-6)
        f0_full = librosa.yin(audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
                              sr=SAMPLE_RATE, hop_length=320)
        pitch_mask_full = (f0_full > 60) & (f0_full < 1100)

        final_data: list = []
        last_seg_end = 0.0

        for seg_idx, seg in enumerate(segments):
            if seg["type"] == "interlude":
                log_debug(f"[間奏] {seg['start']:.2f}s - {seg['end']:.2f}s")
                last_seg_end = seg["end"]
                continue

            lines = seg["lines"]
            seg_start = last_seg_end
            seg_end = total_duration
            for n_seg in segments[seg_idx + 1:]:
                if n_seg["type"] == "interlude":
                    seg_end = n_seg["start"]
                    break

            log_debug(f"【歌詞ブロック】{seg_start:.2f}s - {seg_end:.2f}s ({len(lines)}行)")
            block_audio = audio[int(seg_start * SAMPLE_RATE):int(seg_end * SAMPLE_RATE)]

            rms_b = rms_norm_full[int(seg_start * 50):int(seg_end * 50)]
            pitch_b = pitch_mask_full[int(seg_start * 50):int(seg_end * 50)]

            # ASR 候補抽出 (word_timestamps でより精密なタイミングを取得)
            prompt = " ".join(lines)[:300]
            w_segments, _ = self._whisper.transcribe(
                block_audio, beam_size=5, language="ja", vad_filter=True,
                initial_prompt=prompt, word_timestamps=True,
            )
            w_all_b: list[dict] = []
            for s in w_segments:
                txt_k = self.text_proc.to_phonetic_kana(s.text.strip())
                dur = s.end - s.start
                actual_start, actual_end = s.start, s.end
                if hasattr(s, "words") and s.words:
                    actual_start = s.words[0].start
                idx_s = int(actual_start * 50)
                idx_e = int(actual_end * 50)
                m_rms = np.mean(rms_b[idx_s:idx_e]) if idx_e > idx_s and idx_s < len(rms_b) else 0
                has_p = np.any(pitch_b[idx_s:idx_e]) if idx_e > idx_s and idx_s < len(pitch_b) else False
                log_debug(f"  [ASR候補] {seg_start+actual_start:6.2f}s - {seg_start+actual_end:6.2f}s | RMS={m_rms:.4f} | Pitch={has_p} | '{s.text.strip()}'")
                if not has_p and m_rms < self.settings.auto_interlude_threshold * 0.3:
                    log_debug("    => バリデーション却下: 無音区間")
                    continue
                if dur > 0 and len(txt_k) / dur < self.settings.min_vocal_density:
                    log_debug(f"    => バリデーション却下: 低密度 ({len(txt_k)/dur:.2f} char/s)")
                    continue

                # セグメントレベル候補
                if txt_k:
                    w_all_b.append({"text": txt_k, "start": actual_start, "end": actual_end})

            # アンカー確定 (動的ウィンドウ・動的閾値再試行)
            skeleton_b = self._resolve_anchors_block(lines, w_all_b, seg_start)

            final_data.extend(self._align_and_format(block_audio, lines, skeleton_b, rms_b, seg_start))
            if progress_cb:
                progress_cb(seg_idx + 1, len(segments))

        log_debug("--- 全アライメント完了 ---")
        return final_data

    # ------------------------------------------------------------------
    # アンカー確定 (動的ウィンドウ・動的閾値)
    # ------------------------------------------------------------------

    def _resolve_anchors_block(self, lines: list[str], w_all: list[dict], seg_start: float) -> list[tuple[float, float]]:
        """
        アンカー照合を2段階で行う:
          1. base_score で照合 → 被覆率 50% 未満なら閾値を 0.1 ずつ緩めて再試行
          2. 動的ウィンドウ: w_all の候補数と行数の比率から適切な窓を決定
        """
        base_score = self.settings.anchor_score
        window = 20

        def one_pass(score_thr: float) -> list[tuple[float, float]]:
            skeleton: list[tuple[float, float]] = [(_NAN, _NAN)] * len(lines)
            ptr = 0
            for i, line in enumerate(lines):
                l_kana = self.text_proc.to_phonetic_kana(line)
                if not l_kana:
                    continue
                best_si, best_score = -1, score_thr
                for si in range(ptr, min(ptr + window, len(w_all))):
                    sc = difflib.SequenceMatcher(None, l_kana, w_all[si]["text"]).ratio()
                    if sc > best_score:
                        best_score, best_si = sc, si
                if best_si != -1:
                    skeleton[i] = (w_all[best_si]["start"], w_all[best_si]["end"])
                    ptr = best_si + 1
                    log_debug(f"  [骨格確定] {i+1}行目 -> {seg_start+skeleton[i][0]:.2f}s (Score={best_score:.2f})")
            return skeleton

        skeleton = one_pass(base_score)
        coverage = sum(1 for a in skeleton if not _is_nan(a[0])) / max(1, len(lines))

        score = base_score
        while coverage < 0.5 and score > 0.65:
            score = round(score - 0.1, 2)
            log_debug(f"  [アンカー再試行] 閾値 {score:.2f} に緩和 (現在の被覆率 {coverage:.0%})")
            skeleton = one_pass(score)
            coverage = sum(1 for a in skeleton if not _is_nan(a[0])) / max(1, len(lines))

        log_debug(f"  [アンカー集計] 被覆率 {coverage:.0%} ({sum(1 for a in skeleton if not _is_nan(a[0]))}/{len(lines)}行)")
        return skeleton

    # ------------------------------------------------------------------
    # CTC + 文字分配
    # ------------------------------------------------------------------

    def _align_and_format(
        self,
        audio: np.ndarray,
        lines: list[str],
        skeleton: list[tuple[float, float]],
        rms_norm: np.ndarray,
        offset: float,
    ) -> list[list[tuple[str, float, float]]]:
        from ctc_segmentation import (
            ctc_segmentation, CtcSegmentationParameters,
            prepare_token_list, determine_utterance_segments,
        )
        params = CtcSegmentationParameters()
        params.char_list = self._ai.char_list
        params.index_duration = 0.02

        all_anchors = [(-1, 0.0, 0.0)]
        for i, a in enumerate(skeleton):
            if not _is_nan(a[0]):
                all_anchors.append((i, a[0], a[1]))
        all_anchors.append((len(lines), len(audio) / SAMPLE_RATE, len(audio) / SAMPLE_RATE))

        # ---- CTC による行境界確定 ----
        line_boundaries: list[tuple[float, float]] = [(_NAN, _NAN)] * len(lines)
        # 非対称マージン: 後方（過去側）は狭く、前方（未来側）は通常通り
        back_margin = self.settings.search_margin * 0.5
        front_margin = self.settings.search_margin

        for idx in range(len(all_anchors) - 1):
            l1, t1_s, t1_e = all_anchors[idx]
            l2, t2_s, t2_e = all_anchors[idx + 1]
            gap_indices = [i for i in range(int(l1) + 1, int(l2)) if i < len(lines)]
            if l2 < len(lines) and l2 != 999.9:
                gap_indices.append(int(l2))
            if not gap_indices:
                continue

            # Fix-A: 窓終端に t2_e (セグメント末尾) を使う。
            #   t2_s は「このアンカーが含まれるセグメントの開始時刻」にすぎず、
            #   セグメントが複数行を含む場合にアンカー行が t2_s より後ろに来るケースがある。
            # Fix-B: ギャップがアンカー行1行のみ → t1_s ではなく t2_s 周辺の狭い窓で探索。
            #   前アンカー基準の広い窓だと CTC が誤った局所解に収束しやすい。
            if gap_indices == [int(l2)] and not _is_nan(t2_s):
                ss_samples = max(0, int(t2_s * SAMPLE_RATE) - int(back_margin * SAMPLE_RATE))
            else:
                ss_samples = max(0, int(t1_s * SAMPLE_RATE) - int(back_margin * SAMPLE_RATE))
            es_samples = min(len(audio), int(t2_e * SAMPLE_RATE) + int(front_margin * SAMPLE_RATE))
            ss = ss_samples / SAMPLE_RATE
            crop = audio[ss_samples:es_samples]
            if len(crop) < 400:
                continue

            gap_kana, gap_ids, valid_i = [], [], []
            for i in gap_indices:
                k = self.text_proc.to_phonetic_kana(lines[i])
                ids = [self._ai.char_to_id[c] for c in k if c in self._ai.char_to_id]
                if ids:
                    gap_kana.append(k)
                    gap_ids.append(np.array(ids))
                    valid_i.append(i)
            if gap_ids:
                try:
                    lp = self._ai.get_log_probs(crop)
                    gt_mat, utt_idx = prepare_token_list(params, gap_ids)
                    timings, char_probs, _ = ctc_segmentation(params, lp, gt_mat)
                    segs = determine_utterance_segments(params, utt_idx, char_probs, timings, gap_kana)
                    for v, seg in zip(valid_i, segs):
                        line_boundaries[v] = (ss + seg[0], ss + seg[1])
                        log_debug(f"    [CTC確定] {v+1}行目: {offset+ss+seg[0]:.2f}s - {offset+ss+seg[1]:.2f}s | {lines[v][:20]}")
                except Exception as e:
                    log_debug(f"    [CTC失敗] {e}")

        # ---- 補間 → 文字分配 ----
        final_line_anchors = self._interpolate(line_boundaries, lines)
        _, token_lines = self.text_proc.build_kana_lines(lines)

        # alignment_mode の解決: ctc は Phase 3 未実装のため hybrid にフォールバック
        mode = self.settings.alignment_mode
        if mode == "ctc":
            log_debug("  [分配モード] ctc (文字単位CTC未実装) → hybrid にフォールバック")
            mode = "hybrid"
        elif mode not in ("linear", "hybrid"):
            mode = "hybrid"
        log_debug(f"  [分配モード] {mode}")

        results: list[list[tuple[str, float, float]]] = []
        last_time = 0.0

        for line, line_tokens, (as_, ae) in zip(lines, token_lines, final_line_anchors):
            cursor = max(last_time + self.settings.min_gap, as_)
            dur_total = max(0.1, ae - cursor)

            # モーラ重み計算: かな文字ごとの発音長を反映した重み
            kana_str = self.text_proc.to_phonetic_kana(line)
            token_weights: list[float] = []
            for ch, ks, kl, w in line_tokens:
                if kl == 0:
                    token_weights.append(0.0)
                else:
                    kana_sub = kana_str[ks:ks + kl] if ks + kl <= len(kana_str) else kana_str[ks:]
                    token_weights.append(sum(MORA_WEIGHT.get(c, 1.0) for c in kana_sub))
            total_weight = sum(token_weights) or 1.0

            fixed: list[tuple[str, float, float]] = []
            for (ch, ks, kl, w), tok_w in zip(line_tokens, token_weights):
                if kl == 0 or tok_w == 0.0:
                    # 記号等: 幅ゼロでカーソル位置に置く
                    fixed.append((ch, round(cursor + offset, 4), round(cursor + offset, 4)))
                    continue

                s = cursor
                base_dur = dur_total * (tok_w / total_weight)

                if mode == "hybrid":
                    # 母音オンセットスナップ: 局所RMSが50%閾値を超える最初のフレーム
                    s_idx = int(s * 50)
                    r_s = max(0, s_idx - 3)
                    r_e = min(len(rms_norm), s_idx + 6)
                    if r_s < r_e and r_s < len(rms_norm):
                        rms_local = rms_norm[r_s:r_e]
                        local_max = float(np.max(rms_local))
                        if local_max > 0:
                            above = np.where(rms_local >= local_max * 0.5)[0]
                            if len(above):
                                snap = (r_s + int(above[0]) - s_idx) * 0.02
                                s = max(cursor, s + snap)

                e = s + base_dur
                d = min(self.settings.max_char_duration, max(self.settings.min_char_duration, e - s))
                e = s + d
                fixed.append((ch, round(s + offset, 4), round(e + offset, 4)))
                cursor = e

            # 隣接タグ重複解消
            for j in range(len(fixed) - 1):
                if fixed[j][2] > fixed[j + 1][1]:
                    fixed[j] = (fixed[j][0], fixed[j][1], fixed[j + 1][1])

            if fixed:
                last_time = fixed[-1][2] - offset
            results.append(fixed)

        return results

    # ------------------------------------------------------------------
    # 補間
    # ------------------------------------------------------------------

    def _interpolate(self, anchors: list[tuple[float, float]], lines: list[str]) -> list[tuple[float, float]]:
        res = list(anchors)
        for i in range(len(res)):
            if not _is_nan(res[i][0]):
                continue
            p_idx = next((j for j in range(i - 1, -1, -1) if not _is_nan(res[j][0])), -1)
            n_idx = next((j for j in range(i + 1, len(res)) if not _is_nan(res[j][0])), -1)
            if p_idx == -1 and n_idx == -1:
                s = i * 2.0; res[i] = (s, s + 1.5)
            elif p_idx == -1:
                gap = (n_idx - i) * 2.0; s = max(0.0, res[n_idx][0] - gap); res[i] = (s, s + 1.5)
            elif n_idx == -1:
                gap = (i - p_idx) * 2.0; s = res[p_idx][1] + gap; res[i] = (s, s + 1.5)
            else:
                t1, t2 = res[p_idx][1], res[n_idx][0]
                tc = sum(len(self.text_proc.to_phonetic_kana(lines[k])) for k in range(p_idx + 1, n_idx))
                bc = sum(len(self.text_proc.to_phonetic_kana(lines[k])) for k in range(p_idx + 1, i))
                mc = len(self.text_proc.to_phonetic_kana(lines[i]))
                s = t1 + (t2 - t1) * (bc / (tc or 1))
                e = t1 + (t2 - t1) * ((bc + mc) / (tc or 1))
                res[i] = (s, e)
        return res
