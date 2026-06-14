import sys
import os
import re
import pykakasi

def convert_to_ruby(input_path):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    # Prepare output path: append '_kana' to the filename
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_kana.txt"

    # Initialize pykakasi
    kks = pykakasi.kakasi()

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        final_lines = []
        # Regex to detect Kanji and Katakana
        target_pattern = re.compile(r'[\u4e00-\u9faf\u3400-\u4dbf\u30a0-\u30ff]')

        for line in lines:
            # Process each line, removing trailing newlines to avoid pykakasi duplication bug
            stripped_line = line.rstrip('\r\n')
            result = kks.convert(stripped_line)
            output_segments = []
            for item in result:
                orig = item['orig']
                hira = item['hira']
                
                if target_pattern.search(orig):
                    if orig != hira:
                        output_segments.append(f"[{orig}|{hira}]")
                    else:
                        output_segments.append(orig)
                else:
                    output_segments.append(orig)
            
            # Re-add newline
            final_lines.append("".join(output_segments) + "\n")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)

        print(f"Successfully converted. Output saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ruby_converter.py <input_file>")
    else:
        convert_to_ruby(sys.argv[1])
