from src.text_processor import TextProcessor

def test_parsing():
    tp = TextProcessor()
    lines = [
        "無敵の笑顔で沸かす[わかす]メディア",
        "Perfect[ぱーふぇくと]な君は",
        "宇宙(そら)を駆ける",
        "I love you[あいらぶゆー]"
    ]
    
    print("--- Phonetic Kana ---")
    for line in lines:
        print(f"Original: {line}")
        print(f"Phonetic: {tp.to_phonetic_kana(line)}")
        print()

    print("--- Token Mapping ---")
    kana_lines, token_lines = tp.build_kana_lines(lines)
    for line, tokens in zip(lines, token_lines):
        print(f"Line: {line}")
        for ch, start, length in tokens:
            if start != -1:
                print(f"  '{ch}' -> idx:{start}, len:{length}")
        print()

if __name__ == "__main__":
    test_parsing()
