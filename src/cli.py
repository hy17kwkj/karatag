"""
Karatag CLI インターフェース (V131.20 - インテリジェント・診断レポート版).

責務:
1. タグバリデーション。
2. LyricsAligner 呼び出し。
3. formatter による出力。
4. evaluator による品質診断（ログ末尾に出力）。
"""

import argparse
import sys
import re
from pathlib import Path

import librosa
import numpy as np

from .aligner import LyricsAligner, log_debug
from .formatter import KaraFormatter
from .evaluator import AlignmentEvaluator

def validate_lyrics_tags(text: str) -> list[str]:
    """歌詞内のブラケット記法が正しい形式かチェックする。"""
    errors = []
    tags = re.findall(r"\[[^\]]+\]", text)
    ruby_pattern = re.compile(r"\[[^\|\]]+\|[^\|\]]+\]")
    interlude_pattern = re.compile(r"\[\d+:\d+.*-.*\d+:\d+.*\]")
    for tag in tags:
        if ruby_pattern.match(tag): continue
        if interlude_pattern.match(tag): continue
        errors.append(f"不正なタグ形式です: '{tag}' (正しい形式: [表示|よみ] または [mm:ss-mm:ss])")
    return errors

def main():
    parser = argparse.ArgumentParser(description="Karatag: Karaoke Tag Generator")
    parser.add_argument("audio", help="入力音声ファイル (WAV/MP3)")
    parser.add_argument("lyrics", help="歌詞テキストファイル (.txt)")
    parser.add_argument("-o", "--output", help="出力ファイル名 (.result.txt)")
    parser.add_argument("--preset", choices=["standard", "fast", "ballad"], default="standard", help="楽曲プリセット")
    parser.add_argument("--ignore", type=float, default=0.0, help="冒頭のスキップ秒数")
    parser.add_argument("--eval", help="比較対象の正解KRAファイル (評価用)")
    args = parser.parse_args()

    # 1. 歌詞読み込みとバリデーション
    try:
        lyrics_text = Path(args.lyrics).read_text(encoding="utf-8")
        tag_errors = validate_lyrics_tags(lyrics_text)
        if tag_errors:
            print("\nError: 歌詞ファイルに修正が必要です:")
            for err in tag_errors: print(f"  - {err}")
            return 1
    except Exception as e:
        print(f"Error: 歌詞ファイルの読み込みに失敗しました: {e}")
        return 1

    # 2. 音声読み込み
    print("音声ファイルを読み込み中...")
    try:
        audio, sr = librosa.load(args.audio, sr=16000, mono=True)
        if args.ignore > 0:
            print(f"  冒頭 {args.ignore} 秒をスキップします")
            audio = audio[int(args.ignore * 16000):]
        log_debug(f"CLI起動: audio={args.audio}, lyrics={args.lyrics}")
    except Exception as e:
        print(f"Error: 音声の読み込みに失敗しました: {e}")
        return 1

    # 3. アライメント実行
    aligner = LyricsAligner()
    print("モデルを読み込み中...")
    aligner.load_models()
    aligner.settings.load(args.preset)

    print("アライメント実行中...")
    def progress(current, total):
        bar_len = 30; filled = int(bar_len * current / total)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {current}/{total} 行")
        sys.stdout.flush()

    try:
        result_raw = aligner.run(audio, lyrics_text, progress_cb=progress)
        print("\nアライメント完了。")
    except Exception as e:
        print(f"\nError: アライメント中にエラーが発生しました: {e}")
        import traceback
        log_debug(traceback.format_exc())
        return 1

    # 4. 出力保存
    output_path = args.output or f"{Path(args.audio).stem}.result.txt"
    formatter = KaraFormatter()
    offset_result = []
    for line in result_raw:
        offset_line = []
        for char, s, e in line:
            offset_line.append((char, s + args.ignore, e + args.ignore))
        offset_result.append(offset_line)
    
    formatter.save_result(offset_result, output_path)
    print(f"出力ファイルを保存しました: {output_path}")

    # 5. 品質診断とアドバイスの出力 (ログ末尾へ)
    evaluator = AlignmentEvaluator()
    quality_report = evaluator.evaluate(offset_result, reference_kra=args.eval)
    
    # 【最重要】既存の処理を変更せず、事後的にログへ追記
    log_debug("\n" + quality_report)
    
    if args.eval:
        print("\n正解データと比較中: " + args.eval)
        print(quality_report)
    else:
        print("\n自己診断レポートとチューニング案をログに記録しました (karatag_debug.log)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
