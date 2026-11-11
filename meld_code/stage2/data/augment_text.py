"""Back-translate a training manifest using local OPUS-MT models."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--en-zh', required=True, help='Local English-to-Chinese model directory')
    parser.add_argument('--zh-en', required=True, help='Local Chinese-to-English model directory')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.input.resolve() == args.output.resolve():
        parser.error('Provide a separate augmented manifest')

    frame = pd.read_csv(args.input, keep_default_na=False)
    texts = frame['text'].tolist()
    if not texts or any(not str(text).strip() for text in texts):
        raise ValueError('The training manifest needs nonempty transcripts')
    en_zh = MarianTokenizer.from_pretrained(args.en_zh, local_files_only=True)
    zh_en = MarianTokenizer.from_pretrained(args.zh_en, local_files_only=True)
    to_chinese = MarianMTModel.from_pretrained(args.en_zh, local_files_only=True).to(args.device).eval()
    to_english = MarianMTModel.from_pretrained(args.zh_en, local_files_only=True).to(args.device).eval()

    translated = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            batch = [str(text) for text in texts[start:start + args.batch_size]]
            source = en_zh(batch, return_tensors='pt', padding=True, truncation=True).to(args.device)
            chinese = en_zh.batch_decode(to_chinese.generate(**source), skip_special_tokens=True)
            source = zh_en(chinese, return_tensors='pt', padding=True, truncation=True).to(args.device)
            english = zh_en.batch_decode(to_english.generate(**source), skip_special_tokens=True)
            translated.extend(result if result.strip() else original for original, result in zip(batch, english))

    frame['aug_text'] = translated
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f'Wrote {len(frame)} augmented transcripts to {args.output}')


if __name__ == '__main__':
    main()
