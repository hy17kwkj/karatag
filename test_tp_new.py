from src.text_processor import TextProcessor

def test_new_format():
    tp = TextProcessor()
    lines = [
        "無敵の[笑顔|えがお]で[沸かす|わかす]メディア",
        "[I love you|あいらぶゆー]な君は",
        "宇宙(そら)を駆ける", # 従来の形式もチェック
        "一番星の[生まれ変わり|うまれかわり]"
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
    test_new_format()
