"""IEMOCAP audio-text data loading and collation."""
import csv
import hashlib
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from mmsd.batch import collate
from mmsd.ctc import build_ctc_record, resolve_conv_config

LABELS = ['ang', 'hap', 'neu', 'sad']


def read_rows(csv_path):
    with open(csv_path, encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    ids = [r['file'] for r in rows]
    assert len(ids) == len(set(ids)) == 5531
    assert all(r['emotion'] in LABELS and r['text'].strip() for r in rows)
    assert all(r['speaker'][:5] == r['file'][:5] for r in rows)
    return rows


def split_rows(rows, fold):
    assert fold in range(1, 6)
    train = [r for r in rows if int(r['file'][3:5]) != fold]
    heldout = [r for r in rows if int(r['file'][3:5]) == fold]
    assert {int(r['file'][3:5]) for r in train} == set(range(1, 6)) - {fold}
    assert {r['file'] for r in train}.isdisjoint(r['file'] for r in heldout)
    assert len(train) + len(heldout) == len(rows)
    return train, heldout


def rows_digest(rows):
    return hashlib.sha256('\n'.join(r['file'] for r in rows).encode()).hexdigest()


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


class IEMOCAPDataset(Dataset):
    def __init__(self, rows, config):
        self.rows = rows
        self.audio_dir = Path(config['audio_dir'])
        self.sample_rate = config['sample_rate']
        self.max_samples = int(config['max_audio_seconds'] * self.sample_rate)
        tokenizer = RobertaTokenizer.from_pretrained(config['roberta_path'], local_files_only=True)

        self.encoded = tokenizer([r['text'] for r in rows], truncation=True,
                                 max_length=config['max_text_tokens'], padding=False)
        self.pad_id = tokenizer.pad_token_id

        self.ctc_conv_kernel, self.ctc_conv_stride, self.ctc_encoder_config = resolve_conv_config(config)
        self.ctc_records = []
        for row in rows:
            header = sf.info(str(self.audio_dir / (row['file'] + '.wav')))
            self.ctc_records.append(build_ctc_record(
                row, header.frames, header.samplerate,
                self.ctc_conv_kernel, self.ctc_conv_stride,
                sample_rate=self.sample_rate, max_samples=self.max_samples))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.audio_dir / (row['file'] + '.wav')

        wave, rate = sf.read(str(path), dtype='float32', always_2d=True)
        wave = torch.from_numpy(wave.mean(axis=1).copy())
        if rate != self.sample_rate:
            wave = torchaudio.functional.resample(wave, rate, self.sample_rate)
        wave = wave[:self.max_samples]
        assert wave.numel() >= 400 and torch.isfinite(wave).all(), str(path)
        if wave.numel() != self.ctc_records[index]['audio_samples_used']:
            raise ValueError(f'WAV length differs from the CTC header manifest: {path}')
        return {'audio': wave, 'input_ids': self.encoded['input_ids'][index],
                'label': LABELS.index(row['emotion']), 'file_id': row['file'],
                'speaker': row['speaker'], 'pad_id': self.pad_id,
                'ctc_record': self.ctc_records[index]}
