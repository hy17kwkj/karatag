from src.text_processor import TextProcessor

def test_space_check():
    tp = TextProcessor()
    
    # OK
    tp.to_phonetic_kana("[笑顔|えがお]")
    print("Success: [笑顔|えがお] is OK")
    
    # NG (右側に空白)
    try:
        tp.to_phonetic_kana("[笑顔 |えがお]")
        print("Fail: [笑顔 |えがお] should raise ValueError")
    except ValueError as e:
        print(f"Success: Caught expected error (Right): {e}")

    # NG (左側に空白)
    try:
        tp.to_phonetic_kana("[笑顔| えがお]")
        print("Fail: [笑顔| えがお] should raise ValueError")
    except ValueError as e:
        print(f"Success: Caught expected error (Left): {e}")

    # NG (両側に空白)
    try:
        tp.to_phonetic_kana("[ 笑顔 | えがお ]")
        print("Fail: [ 笑顔 | えがお ] should raise ValueError")
    except ValueError as e:
        print(f"Success: Caught expected error (Both): {e}")

if __name__ == "__main__":
    test_space_check()
