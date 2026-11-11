"""Original and augmented MMSD views, with fold-bound teacher targets."""
from pathlib import Path
import csv
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
from .ctc import build_ctc_record, resolve_conv_config
from .batch import collate as base_collate


def read_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != len({r['file'] for r in rows}):
        raise ValueError(f'Empty manifest or repeated file IDs: {path}')
    for row in rows:
        file_id = row['file']
        if not file_id or '/' in file_id or '\\' in file_id or file_id in ('.', '..'):
            raise ValueError(f'Invalid audio basename: {file_id}')
    return rows


class Views(Dataset):
    def __init__(self, rows, cfg, labels, aug_rows=None, audio_dir=None,
                 augmented_audio=None, targets=None, augmented_text_column='text'):
        self.cfg, self.labels = cfg, labels
        self.audio_dir = Path(audio_dir or cfg['audio_dir'])
        self.augmented_audio = Path(augmented_audio) if augmented_audio else None
        self.rows = [(r, False) for r in rows]
        self.aug = {r['file']:r for r in aug_rows} if aug_rows else {}
        self.target = targets
        self.text_column = augmented_text_column
        if targets is not None:
            assert set(self.aug) == {r['file'] for r in rows} == set(targets)
            self.rows += [(r, True) for r in rows]
        tokenizer = RobertaTokenizer.from_pretrained(cfg['roberta_path'], local_files_only=True)
        self.pad_id = tokenizer.pad_token_id
        texts = []
        self.records, self.paths = [], []
        kernel, stride, _ = resolve_conv_config(cfg)
        for row, soft in self.rows:
            aug_text = self.aug[row['file']][self.text_column] if self.aug else row['text']
            text = aug_text if soft else row['text']
            assert text.strip() and aug_text.strip()
            texts.append(text)
            folder = self.augmented_audio if soft else self.audio_dir
            path = folder / (row['file']+'.wav')
            if not path.is_file():
                path = folder / (row['file']+'_aug.wav')
            header = sf.info(str(path))
            record_row = dict(row, text=text, label_id=labels.index(row['emotion']),
                              split=row['file'].split('_')[1] if len(labels)==7 else 'session')
            self.records.append(build_ctc_record(record_row, header.frames, header.samplerate,
                kernel, stride, sample_rate=cfg['sample_rate'],
                max_samples=int(cfg['sample_rate']*cfg['max_audio_seconds'])))
            self.paths.append(path)
        self.encoded = tokenizer(texts, truncation=True, max_length=cfg['max_text_tokens'], padding=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row, soft = self.rows[i]
        wave, rate = sf.read(str(self.paths[i]), dtype='float32', always_2d=True)
        wave = torch.from_numpy(wave.mean(axis=1).copy())
        if rate != self.cfg['sample_rate']:
            wave = torchaudio.functional.resample(wave, rate, self.cfg['sample_rate'])
        wave = wave[:int(self.cfg['max_audio_seconds']*self.cfg['sample_rate'])]
        assert len(wave) >= 400 and torch.isfinite(wave).all()
        assert len(wave) == self.records[i]['audio_samples_used']
        return {'audio':wave, 'input_ids':self.encoded['input_ids'][i],
            'label':self.labels.index(row['emotion']), 'file_id':row['file'],
            'speaker':row['speaker'], 'pad_id':self.pad_id, 'ctc_record':self.records[i],
            'is_soft':soft, 'soft_target':self.target[row['file']] if soft else [0.]*len(self.labels)}


def collate(items):
    batch = base_collate(items)
    batch['is_soft'] = torch.tensor([i['is_soft'] for i in items], dtype=torch.bool)
    batch['soft_targets'] = torch.tensor(np.asarray([i['soft_target'] for i in items]), dtype=torch.float32)
    return batch
