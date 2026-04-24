"""
コア・強制アライメントエンジン (V131.9 - 詳細ログ復活版).

特徴:
1. 詳細デバッグログ: ASR候補、バリデーション、CTC確定プロセスをすべて実況。
2. ハイブリッド分割: タグの有無で最適ロジックを使い分け。
3. 自己診断レポートとの統合。
"""

from __future__ import annotations

import sys
import datetime
import logging
import configparser
import difflib
import re
from pathlib import Path
from typing import Callable, Any

import numpy as np
import librosa
import torch

# 高速化のため並列処理を許可
torch.set_num_threads(4)

from .text_processor import TextProcessor

# ---------------------------------------------------------------------------
# 定数と設定
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
DEFAULT_W2V2_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
_NAN = float("nan")

def _is_nan(v: float) -> bool:
    return v != v

class Settings:
    def __init__(self):
        self.min_gap = 0.01
        self.max_char_duration = 1.2
        self.min_char_duration = 0.03
        self.search_margin = 1.5
        self.anchor_score = 0.9
        self.alignment_mode = "linear"
        self.auto_interlude_duration = 10.0
        self.auto_interlude_threshold = 0.02
        self.min_vocal_density = 0.3
        self.load()

    def load(self, preset_name: str | None = None):
        path = Path("settings.ini")
        if not path.exists(): return
        config = configparser.ConfigParser()
        try:
            content = "[DEFAULT]\n" + path.read_text(encoding="utf-8")
            config.read_string(content)
            def get_safe_float(section, key, curr):
                try: return config.getfloat(section, key, fallback=curr)
                except: return curr

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
        self.model.eval(); self.device = torch.device("cpu"); self.model.to(self.device)
        vocab = self.processor.tokenizer.get_vocab()
        self.char_list = [k for k, _ in sorted(vocab.items(), key=lambda x: x[1])]
        self.char_to_id = {c: i for i, c in enumerate(self.char_list)}

    def get_log_probs(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) < 400: return np.array([])
        with torch.no_grad():
            norm = audio / (np.max(np.abs(audio)) + 1e-6)
            inputs = self.processor(norm, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            lp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        return lp

# ---------------------------------------------------------------------------
# オーケストレーター
# ---------------------------------------------------------------------------

class LyricsAligner:
    def __init__(self, cache_dir: str | Path = "models", w2v2_model: str = DEFAULT_W2V2_MODEL, whisper_model: str = "medium") -> None:
        self.cache_dir, self.whisper_name = Path(cache_dir), whisper_model
        self.w2v2_name = w2v2_model
        self.text_proc, self.settings = TextProcessor(), Settings()
        self._ai, self._whisper = None, None

    def load_models(self, progress_cb: Callable[[str], None] | None = None) -> None:
        if progress_cb: progress_cb("AIエンジン起動中...")
        from faster_whisper import WhisperModel
        self._ai = _Wav2Vec2Engine(self.w2v2_name, self.cache_dir)
        self._whisper = WhisperModel(self.whisper_name, device="cpu", compute_type="int8", download_root=str(self.cache_dir), cpu_threads=4)

    def run(self, audio: np.ndarray, lyrics_text: str, progress_cb: Callable[[int, int], None] | None = None) -> list[list[tuple[str, float, float]]]:
        if not self._whisper or not self._ai: raise RuntimeError("load_models() required")
        self.settings.load()
        has_tags = bool(re.search(r"\[\d+:\d+", lyrics_text))
        
        if not has_tags:
            log_debug("--- 広域アライメント開始 (Monolithic) ---")
            from . import aligner_legacy
            legacy_aligner = aligner_legacy.LyricsAligner(self.cache_dir, self.w2v2_name, self.whisper_name)
            legacy_aligner._ai = self._ai
            legacy_aligner._whisper = self._whisper
            return legacy_aligner.run(audio, lyrics_text, progress_cb)
        else:
            log_debug("--- 分割アライメント開始 (Partitioned) ---")
            return self._run_partitioned(audio, lyrics_text, progress_cb)

    def _run_partitioned(self, audio: np.ndarray, lyrics_text: str, progress_cb: Callable[[int, int], None] | None = None) -> list[list[tuple[str, float, float]]]:
        segments = self.text_proc.parse_lyrics_file(lyrics_text)
        total_duration = len(audio) / SAMPLE_RATE
        
        # 50fps 特徴量 (バリデーション用)
        rms_full = librosa.feature.rms(y=audio, frame_length=640, hop_length=320)[0]
        rms_norm_full = rms_full / (np.max(rms_full) + 1e-6)
        f0_full = librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=SAMPLE_RATE, hop_length=320)
        pitch_mask_full = (f0_full > 60) & (f0_full < 1100)

        final_data = []; last_seg_end = 0.0
        for seg_idx, seg in enumerate(segments):
            if seg["type"] == "interlude":
                log_debug(f"[間奏] {seg['start']:.2f}s - {seg['end']:.2f}s")
                last_seg_end = seg["end"]; continue
            
            lines = seg["lines"]; seg_start = last_seg_end; seg_end = total_duration
            for n_seg in segments[seg_idx+1:]:
                if n_seg["type"] == "interlude": seg_end = n_seg["start"]; break
            
            log_debug(f"【歌詞ブロック】{seg_start:.2f}s - {seg_end:.2f}s ({len(lines)}行)")
            block_audio = audio[int(seg_start * SAMPLE_RATE) : int(seg_end * SAMPLE_RATE)]
            
            # ブロック内特徴量
            rms_b = rms_norm_full[int(seg_start*50):int(seg_end*50)]
            pitch_b = pitch_mask_full[int(seg_start*50):int(seg_end*50)]
            
            # ASR 候補抽出
            prompt = " ".join(lines)[:300]
            w_segments, _ = self._whisper.transcribe(block_audio, beam_size=5, language="ja", vad_filter=True, initial_prompt=prompt)
            w_all_b = []
            for s in w_segments:
                idx_s, idx_e = int(s.start * 50), int(s.end * 50)
                m_rms = np.mean(rms_b[idx_s:idx_e]) if idx_e > idx_s and idx_s < len(rms_b) else 0
                has_p = np.any(pitch_b[idx_s:idx_e]) if idx_e > idx_s and idx_s < len(pitch_b) else False
                txt_k = self.text_proc.to_phonetic_kana(s.text.strip())
                dur = s.end - s.start
                
                # 詳細ログ
                log_debug(f"  [ASR候補] {seg_start+s.start:6.2f}s - {seg_start+s.end:6.2f}s | RMS={m_rms:.4f} | Pitch={has_p} | '{s.text.strip()}'")
                
                if not has_p and m_rms < self.settings.auto_interlude_threshold * 0.3:
                    log_debug(f"    => ババリデーション却下: 無音区間")
                    continue
                if dur > 0 and (len(txt_k) / dur) < self.settings.min_vocal_density:
                    log_debug(f"    => ババリデーション却下: 低密度 ({len(txt_k)/dur:.2f} char/s)")
                    continue
                w_all_b.append({"text": txt_k, "start": s.start, "end": s.end})
            
            # アンカー確定
            skeleton_b = [(_NAN, _NAN)] * len(lines); w_ptr_b = 0
            for i, line in enumerate(lines):
                l_kana = self.text_proc.to_phonetic_kana(line)
                if not l_kana: continue
                best_si, best_score = -1, self.settings.anchor_score
                for si in range(w_ptr_b, min(w_ptr_b + 5, len(w_all_b))):
                    score = difflib.SequenceMatcher(None, l_kana, w_all_b[si]["text"]).ratio()
                    if score > best_score: best_score = score; best_si = si
                if best_si != -1:
                    skeleton_b[i] = (w_all_b[best_si]["start"], w_all_b[best_si]["end"])
                    w_ptr_b = best_si + 1
                    log_debug(f"  [骨格確定] {i+1}行目 -> {seg_start+skeleton_b[i][0]:.2f}s (Score={best_score:.2f})")
            
            # 整形
            final_data.extend(self._align_and_format(block_audio, lines, skeleton_b, rms_b, seg_start))
            if progress_cb: progress_cb(seg_idx + 1, len(segments))
        log_debug("--- 全アライメント完了 ---")
        return final_data

    def _align_and_format(self, audio: np.ndarray, lines: list[str], skeleton: list[tuple[float, float]], rms_norm: np.ndarray, offset: float) -> list[list[tuple[str, float, float]]]:
        from ctc_segmentation import ctc_segmentation, CtcSegmentationParameters, prepare_token_list, determine_utterance_segments
        params = CtcSegmentationParameters(); params.char_list = self._ai.char_list; params.index_duration = 0.02
        all_anchors = [(-1, 0.0, 0.0)]
        for i, a in enumerate(skeleton):
            if not _is_nan(a[0]): all_anchors.append((i, a[0], a[1]))
        all_anchors.append((len(lines), len(audio)/SAMPLE_RATE, len(audio)/SAMPLE_RATE))
        
        line_boundaries = [(_NAN, _NAN)] * len(lines)
        for idx in range(len(all_anchors) - 1):
            l1, t1_s, t1_e = all_anchors[idx]; l2, t2_s, t2_e = all_anchors[idx+1]
            gap_indices = [i for i in range(int(l1)+1, int(l2)) if i < len(lines)]
            if l2 < len(lines) and l2 != 999.9: gap_indices.append(int(l2))
            if not gap_indices: continue
            ss, es = max(0.0, t1_e - self.settings.search_margin), min(len(audio)/SAMPLE_RATE, t2_s + self.settings.search_margin)
            crop = audio[int(ss*SAMPLE_RATE):int(es*SAMPLE_RATE)]
            if len(crop) < 400: continue
            
            gap_kana, gap_ids, valid_i = [], [], []
            for i in gap_indices:
                k = self.text_proc.to_phonetic_kana(lines[i])
                ids = [self._ai.char_to_id[c] for c in k if c in self._ai.char_to_id]
                if ids: gap_kana.append(k); gap_ids.append(np.array(ids)); valid_i.append(i)
            if gap_ids:
                try:
                    lp = self._ai.get_log_probs(crop); gt_mat, utt_idx = prepare_token_list(params, gap_ids)
                    timings, char_probs, _ = ctc_segmentation(params, lp, gt_mat)
                    segments = determine_utterance_segments(params, utt_idx, char_probs, timings, gap_kana)
                    for v, seg in zip(valid_i, segments): 
                        line_boundaries[v] = (ss + seg[0], ss + seg[1])
                        log_debug(f"    [CTC確定] {v+1}行目: {offset+ss+seg[0]:.2f}s - {offset+ss+seg[1]:.2f}s | {lines[v][:20]}")
                except Exception as e: log_debug(f"    [CTC失敗] {e}")

        final_line_anchors = self._interpolate(line_boundaries, lines)
        results = []; last_time = 0.0
        _, token_lines = self.text_proc.build_kana_lines(lines)
        rms_diff = np.diff(rms_norm, prepend=0)
        for i, (line_tokens, (as_, ae)) in enumerate(zip(token_lines, final_line_anchors)):
            cursor = max(last_time + self.settings.min_gap, as_)
            dur_total = max(0.1, ae - cursor)
            n_chars = len([t for t in line_tokens if t[1] != -1])
            fixed = []
            for t_i, (ch, ks, kl, w) in enumerate(line_tokens):
                s = cursor
                if ks != -1:
                    s_idx = int(s * 50)
                    if s_idx < len(rms_diff):
                        r_s, r_e = max(0, s_idx-5), min(len(rms_diff), s_idx+5)
                        s = max(cursor, s + (np.argmax(rms_diff[r_s:r_e]) - (s_idx - r_s)) * 0.02)
                e = s + (dur_total / max(1, n_chars))
                d = min(self.settings.max_char_duration, max(self.settings.min_char_duration, e - s))
                e = s + d; fixed.append((ch, round(s + offset, 4), round(e + offset, 4))); cursor = e
            for j in range(len(fixed)-1):
                if fixed[j][2] > fixed[j+1][1]: fixed[j] = (fixed[j][0], fixed[j][1], fixed[j+1][1])
            if fixed: last_time = fixed[-1][2] - offset
            results.append(fixed)
        return results

    def _interpolate(self, anchors, lines):
        res = list(anchors)
        for i in range(len(res)):
            if _is_nan(res[i][0]):
                p_idx = next((j for j in range(i-1, -1, -1) if not _is_nan(res[j][0])), -1)
                n_idx = next((j for j in range(i+1, len(res)) if not _is_nan(res[j][0])), -1)
                if p_idx == -1 and n_idx == -1: s = i * 2.0; res[i] = (s, s + 1.5)
                elif p_idx == -1: gap = (n_idx - i) * 2.0; s = max(0.0, res[n_idx][0] - gap); res[i] = (s, s + 1.5)
                elif n_idx == -1: gap = (i - p_idx) * 2.0; s = res[p_idx][1] + gap; res[i] = (s, s + 1.5)
                else:
                    t1, t2 = res[p_idx][1], res[n_idx][0]
                    tc = sum(len(self.text_proc.to_phonetic_kana(lines[k])) for k in range(p_idx+1, n_idx))
                    bc = sum(len(self.text_proc.to_phonetic_kana(lines[k])) for k in range(p_idx+1, i))
                    mc = len(self.text_proc.to_phonetic_kana(lines[i]))
                    s = t1 + (t2 - t1) * (bc / (tc or 1)); e = t1 + (t2 - t1) * ((bc + mc) / (tc or 1)); res[i] = (s, e)
        return res
