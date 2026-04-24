"""
歌詞テキスト処理モジュール (V131.2 - ブロック分割対応版).

特徴:
1. [表示|よみ] 記法のパースと音素数カウント。
2. 間奏タグ [mm:ss - mm:ss] による楽曲の物理分割。
3. 高度なルビ・正規化処理。
"""

import re
import pykakasi
from pathlib import Path

class TextProcessor:
    def __init__(self):
        self.kakasi = pykakasi.kakasi()
        # [mm:ss.xx - mm:ss.xx] または [mm:ss - mm:ss]
        self.interlude_pattern = re.compile(r"\[(\d+):(\d+\.?\d*)\s*-\s*(\d+):(\d+\.?\d*)\]")
        # [表示|よみ]
        self.ruby_pattern = re.compile(r"\[([^\|\]]+)\|([^\|\]]+)\]")

    def to_phonetic_kana(self, text: str) -> str:
        """テキストを比較・カウント用のかな（ひらがな）に変換する。"""
        if not text or text.isspace(): return ""
        
        # 1. [表示|よみ] を 「よみ」に置換
        working_text = text
        for m in self.ruby_pattern.finditer(text):
            working_text = working_text.replace(m.group(0), m.group(2))
        
        # 2. Kakasi でひらがな化
        result = self.kakasi.convert(working_text)
        kana = "".join([item["hira"] for item in result])
        
        # 3. 記号等の除去
        kana = re.sub(r"[^ぁ-んーa-zA-Z0-9]", "", kana)
        return kana

    def parse_lyrics_file(self, text: str) -> list[dict]:
        """
        歌詞ファイルを解析し、ブロック構造のリストを返す。
        Returns: [
            {"type": "lyrics", "lines": ["行1", "行2", ...]},
            {"type": "interlude", "start": 10.0, "end": 20.0},
            ...
        ]
        """
        raw_lines = text.splitlines()
        segments = []
        current_lyrics = []

        for line in raw_lines:
            line = line.strip()
            if not line: continue

            # 間奏タグの判定
            match = self.interlude_pattern.match(line)
            if match:
                # 溜まっていた歌詞があればブロックとして追加
                if current_lyrics:
                    segments.append({"type": "lyrics", "lines": current_lyrics})
                    current_lyrics = []
                
                # 間奏セグメントを追加
                s_min, s_sec, e_min, e_sec = map(float, match.groups())
                segments.append({
                    "type": "interlude",
                    "start": s_min * 60 + s_sec,
                    "end": e_min * 60 + e_sec
                })
            else:
                current_lyrics.append(line)

        # 最後に残った歌詞を追加
        if current_lyrics:
            segments.append({"type": "lyrics", "lines": current_lyrics})

        return segments

    def build_kana_lines(self, lines: list[str]) -> tuple[list[str], list[list[tuple]]]:
        """
        各行をトークン化し、(表示文字, かな開始, かな長, 重み) のリストを返す。
        """
        processed_lines = []
        token_lines = []

        for line in lines:
            tokens = []
            cursor = 0
            # [表示|よみ] を探す
            matches = list(self.ruby_pattern.finditer(line))
            
            last_pos = 0
            for m in matches:
                # 前の通常テキスト部分
                pre_text = line[last_pos:m.start()]
                if pre_text:
                    for char in pre_text:
                        k = self.to_phonetic_kana(char)
                        tokens.append((char, cursor, len(k), 1.0))
                        cursor += len(k)
                
                # [表示|よみ] 部分
                display, reading = m.group(1), m.group(2)
                k = self.to_phonetic_kana(reading)
                tokens.append((display, cursor, len(k), 1.2)) # ルビは少し重めに
                cursor += len(k)
                last_pos = m.end()
            
            # 残りのテキスト
            post_text = line[last_pos:]
            if post_text:
                for char in post_text:
                    k = self.to_phonetic_kana(char)
                    tokens.append((char, cursor, len(k), 1.0))
                    cursor += len(k)
            
            processed_lines.append(line)
            token_lines.append(tokens)
            
        return processed_lines, token_lines
