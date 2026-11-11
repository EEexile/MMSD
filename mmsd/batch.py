"""Pad utterance batches and pack eligible CTC targets."""
import torch


def collate(batch):
    # Batch-wise padding; all variants use identical order, limits and preprocessing.
    lengths = torch.tensor([b['audio'].numel() for b in batch], dtype=torch.long)
    audio = torch.zeros(len(batch), int(lengths.max()))
    text_len = max(len(b['input_ids']) for b in batch)
    tokens = torch.full((len(batch), text_len), batch[0]['pad_id'], dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for i, item in enumerate(batch):
        audio[i, :lengths[i]] = item['audio']
        n = len(item['input_ids'])
        tokens[i, :n] = torch.tensor(item['input_ids'])
        mask[i, :n] = 1
    records = [item['ctc_record'] for item in batch]
    target_lengths = torch.tensor([r['target_length_clean'] for r in records], dtype=torch.long)
    # Padding uses blank=0, but only the first target_lengths entries are targets.
    targets = torch.zeros((len(batch), int(target_lengths.max())), dtype=torch.long)
    for i, record in enumerate(records):
        targets[i, :target_lengths[i]] = torch.tensor(record['target_ids'], dtype=torch.long)
    return {'audio_inputs': audio, 'audio_lengths': lengths,
            'text_input_ids': tokens, 'text_attention_mask': mask,
            'labels': torch.tensor([b['label'] for b in batch], dtype=torch.long),
            'file_ids': [b['file_id'] for b in batch],
            'speakers': [b['speaker'] for b in batch],
            'ctc_targets': targets, 'ctc_target_lengths': target_lengths,
            'ctc_eligible': torch.tensor([r['eligible'] for r in records], dtype=torch.bool),
            'ctc_exclusion_reasons': [r['exclusion_reasons'] for r in records],
            'ctc_target_texts': [r['target_text'] for r in records],
            'ctc_target_hashes': [r['target_hash'] for r in records],
            'ctc_expected_feature_lengths': torch.tensor([r['audio_feature_frames'] for r in records], dtype=torch.long),
            'ctc_minimum_frames': torch.tensor([r['minimum_ctc_frames_clean'] for r in records], dtype=torch.long)}
