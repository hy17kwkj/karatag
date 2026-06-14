"""
アライメント結果の出力フォーマット（.result.txt）。
"""

from __future__ import annotations
from pathlib import Path
import re

class KaraFormatter:
    def format_timestamp(self, seconds: float) -> str:
        """秒数を [mm:ss:cc] 形式に変換。"""
        if seconds < 0: seconds = 0
        m = int(seconds // 60)
        s = int(seconds % 60)
        c = int((seconds * 100) % 100)
        return f"[{m:02d}:{s:02d}:{c:02d}]"

    def save_result(self, lyrics_data: list[list[tuple[str, float, float]]], output_path: str | Path):
        """アライメント結果をカラオケタグ形式で保存。"""
        lines = []
        for line in lyrics_data:
            if not line:
                lines.append("")
                continue
            formatted_line = ""
            last_end = 0.0
            for char, start, end in line:
                ts = self.format_timestamp(start)
                formatted_line += f"{ts}{char}"
                last_end = end
            # 行末に終了時間を追加
            formatted_line += self.format_timestamp(last_end)
            lines.append(formatted_line)
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")

    def parse_kra(self, kra_path: str) -> list[list[tuple[str, float, float]]]:
        """KRA形式をパース。歌詞を1文字ずつに分解して評価可能にする。"""
        try:
            content = Path(kra_path).read_text(encoding="utf-8")
            lines = []
            for raw_line in content.splitlines():
                if not raw_line.strip(): continue
                # [mm:ss:cc]歌詞 のパターン
                parts = re.findall(r"\[(\d+):(\d+):(\d+)\]([^\[]+)", raw_line)
                line_data = []
                for mm, ss, cc, text in parts:
                    t_start = int(mm)*60 + int(ss) + int(cc)/100.0
                    # 【重要】歌詞テキストを1文字ずつに分解
                    chars = list(text.strip())
                    if not chars: continue
                    # 1文字あたりの持続時間を 0.1s (ダミー) として展開
                    for i, char in enumerate(chars):
                        # 評価に使用するのは開始時間(t_start)のみなので、
                        # 同一タグ内の文字はすべて同じ開始時間を持つものとして扱う
                        line_data.append((char, t_start, t_start + 0.1))
                if line_data: lines.append(line_data)
            return lines
        except:
            return []
