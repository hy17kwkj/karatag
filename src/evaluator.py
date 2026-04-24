"""
アライメント品質評価・自己診断モジュール (V131.20 - パラメータ提案対応版).

目標: 正解がない場合、統計的指標から異常を検知し、パラメータ調整を含む改善策をログの末尾に提示する。
"""

from __future__ import annotations
import numpy as np
import re
import difflib
from .text_processor_legacy import TextProcessor as LegacyProcessor

class AlignmentEvaluator:
    def __init__(self):
        self.tp = LegacyProcessor()

    def evaluate(self, ours: list[list[tuple[str, float, float]]], reference_kra: str = None) -> str:
        report = ["====================================================",
                  "  Karatag アライメント品質レポート",
                  "===================================================="]
        
        if reference_kra:
            report.append(self._compare_with_kra(ours, reference_kra))
        else:
            report.append("  [!] 正解データ未指定のため、統計的自己診断を実行します。")
            report.append(self._self_diagnose(ours))
        
        report.append("====================================================")
        return "\n".join(report)

    def _self_diagnose(self, ours: list[list[tuple[str, float, float]]]) -> str:
        diag = ["  [自己診断モード] 統計的指標に基づく解析結果:"]
        
        densities = []
        char_durs = []
        for line in ours:
            if not line: continue
            valid_chars = [c for c in line if not re.match(r"[\s　\(\)（）\[\]\|]", c[0])]
            dur = line[-1][2] - line[0][1]
            if dur > 0: densities.append(len(valid_chars) / dur)
            for _, s, e in line: char_durs.append(e - s)

        if not densities: return "  (解析可能なデータがありません)"
        
        avg_den = np.mean(densities)
        diag.append(f"  - 平均歌唱密度: {avg_den:.2f} 文字/秒")
        
        issues = []
        for i, (line, d) in enumerate(zip(ours, densities)):
            idx = i + 1
            if d > avg_den * 2.5:
                issues.append(f"  ● {idx:2d}行目: 異常高密度 ({d:4.1f} char/s) -> 「ワープ」の疑い\n    【対策】直前に 間奏タグ [mm:ss-mm:ss] を挿入するか、'search_margin' を広げてください。")
            elif d < avg_den * 0.2:
                issues.append(f"  ● {idx:2d}行目: 異常低密度 ({d:4.1f} char/s) -> 「ハルシネーション」の疑い\n    【対策】'--ignore' で歌い出しを調整するか、'anchor_score' を 0.95 以上に上げてください。")
        
        # 物理制約への衝突チェック
        clipped_max = len([d for d in char_durs if d > 1.15])
        if clipped_max > len(char_durs) * 0.1:
            issues.append(f"  ● 全体: 歌唱が不自然に長く伸びています (Max持続時間への到達率: {clipped_max/len(char_durs)*100:.1f}%)\n    【対策】settings.ini で 'max_char_duration' を 2.0s 等に広げるか、間奏を正しく設定してください。")

        if not issues:
            diag.append("  [OK] 統計的な明らかな異常は見つかりませんでした。良好な可能性が高いです。")
        else:
            diag.append(f"  [!] {len(issues)} 件の潜在的な問題を検知しました:")
            diag.extend(issues[:15])
            if len(issues) > 15: diag.append("      (以下省略)")

        diag.append("\n  [パラメータチューニング・ガイド]")
        diag.append("  - 全体的にズレる場合: '--ignore' で歌い出しの直前(例: 0.5s前)を指定してリセットしてください。")
        diag.append("  - 滑舌が悪い/正しく聞き取れない場合: 'anchor_score' を 0.70 程度に下げ、'alignment_mode = ctc' を試してください。")
        diag.append("  - 伴奏を歌と誤認する場合: 'auto_interlude_threshold' を 0.05 程度に上げてください。")
        
        return "\n".join(diag)

    def _compare_with_kra(self, ours, reference_kra):
        # 以前の精密マッチングロジック (V131.17) をそのまま維持
        from .formatter import KaraFormatter
        ref_data = KaraFormatter().parse_kra(reference_kra)
        if not ref_data: return "  [Error] 正解パース失敗"
        def to_vocal_atoms(data):
            atoms = []
            for line_idx, line in enumerate(data):
                for char, s, e in line:
                    kana = self.tp.to_phonetic_kana(char)
                    if kana: atoms.append({"char": char, "kana": kana[0], "time": s, "line": line_idx})
            return atoms
        o_atoms = to_vocal_atoms(ours)
        r_atoms = to_vocal_atoms(ref_data)
        o_str = "".join([x["kana"] for x in o_atoms])
        r_str = "".join([x["kana"] for x in r_atoms])
        matcher = difflib.SequenceMatcher(None, o_str, r_str)
        errors = []; line_stats = {}; matched_count = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for k in range(i2 - i1):
                    o_item, r_item = o_atoms[i1 + k], r_atoms[j1 + k]
                    err = abs(o_item["time"] - r_item["time"])
                    errors.append(err); matched_count += 1
                    l_idx = o_item["line"]
                    if l_idx not in line_stats: line_stats[l_idx] = []
                    line_stats[l_idx].append(err)
        if not errors: return "  [Error] マッチする歌唱音節がありません。"
        res = [f"  比較音節数   : {matched_count}", f"  全体 MAE     : {np.mean(errors):.3f}s", f"  全体 最大誤差: {np.max(errors):.3f}s"]
        return "\n".join(res)
