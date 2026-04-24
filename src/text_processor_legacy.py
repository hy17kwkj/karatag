"""
レガシー歌詞テキストプロセッサ (V131.15 - 構造的アライメント遵守版).

目標: 1文字1タイムスタンプの原則を守り、KRA比較での MAE 0.2s を実現。
"""

import re
import pykakasi

class TextProcessor:
    def __init__(self):
        self.kakasi = pykakasi.kakasi()
        self.ruby_pattern = re.compile(r"\[([^\|\]]+)\|([^\|\]]+)\]")

    def to_phonetic_kana(self, text: str) -> str:
        """読み側の「かな」のみを返す。"""
        if not text: return ""
        # 構造的パースを優先
        matches = list(self.ruby_pattern.finditer(text))
        if matches:
            # タグが含まれる場合は再帰的に処理（または単純に右側を採用）
            working_text = text
            for m in reversed(matches):
                working_text = working_text[:m.start()] + m.group(2) + working_text[m.end():]
            text = working_text

        k_res = self.kakasi.convert(text)
        kana = "".join([item["hira"] for item in k_res])
        # スペースや記号は音素カウントに含めない
        return re.sub(r"[^\u3040-\u309F\u30FCa-zA-Z0-9]", "", kana)

    def parse_lyrics_file(self, text: str):
        """歌詞行のみを抽出。"""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        lines = [l for l in lines if not re.match(r"\[\d+:\d+", l)]
        return lines, []

    def build_kana_lines(self, lines: list[str]) -> tuple[list[str], list[list[tuple]]]:
        """[表示|よみ] を1文字ずつに展開し、KRAとの整合性を100%にする。"""
        token_lines = []
        for line in lines:
            tokens = []
            curr_kana_idx = 0
            
            # 1. ブロックに分割
            last_pos = 0
            for m in self.ruby_pattern.finditer(line):
                # 地の文 (1文字ずつ処理)
                pre_text = line[last_pos:m.start()]
                for char in pre_text:
                    k = self.to_phonetic_kana(char)
                    tokens.append((char, curr_kana_idx if k else -1, len(k), 1.0))
                    curr_kana_idx += len(k)
                
                # ルビ部分 (表示文字を1文字ずつ展開)
                display, reading = m.group(1), m.group(2)
                kana_total = self.to_phonetic_kana(reading)
                k_len = len(kana_total)
                d_len = len(display)
                
                if d_len > 0:
                    # 読みの文字数を表示文字数で等分する (整数除算)
                    base_per_char = k_len // d_len
                    extra = k_len % d_len
                    for i, d_char in enumerate(display):
                        # 余った音素は前の文字から順に割り振る
                        this_k_len = base_per_char + (1 if i < extra else 0)
                        tokens.append((d_char, curr_kana_idx if this_k_len > 0 else -1, this_k_len, 1.2))
                        curr_kana_idx += this_k_len
                
                last_pos = m.end()
            
            # 残りの地の文
            post_text = line[last_pos:]
            for char in post_text:
                k = self.to_phonetic_kana(char)
                tokens.append((char, curr_kana_idx if k else -1, len(k), 1.0))
                curr_kana_idx += len(k)
                
            token_lines.append(tokens)
            
        return lines, token_lines
