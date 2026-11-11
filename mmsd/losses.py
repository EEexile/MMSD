"""CTC and paired audio-text contrastive losses."""
from collections import Counter

import torch
import torch.nn.functional as F


def auxiliary_terms(outputs, batch, config):

    zero = outputs['logits'].new_zeros((), dtype=torch.float32)
    ctc, contrastive = zero, zero
    n = int(batch['labels'].numel())
    counts = {'ctc_eligible_count': 0, 'ctc_target_chars': 0,
              'ctc_excluded_count': n, 'ctc_exclusion_counts': {},
              'contrastive_candidates': n}
    if config['auxiliary_weights']['ctc']:
        logits = outputs['ctc_logits']
        lengths = outputs['ctc_input_lengths']
        targets = batch['ctc_targets']
        target_lengths = batch['ctc_target_lengths']
        eligible = batch['ctc_eligible'].bool()
        if logits is None or lengths is None:
            raise ValueError('Active CTC requires pre-fusion audio outputs')
        if logits.shape[0] != n or logits.shape[-1] != 33:
            raise ValueError('CTC logits must have shape (B,T,33)')
        if eligible.shape != (n,) or target_lengths.shape != (n,):
            raise ValueError('CTC eligibility/length batch mismatch')
        if 'ctc_expected_feature_lengths' in batch and not torch.equal(lengths, batch['ctc_expected_feature_lengths']):
            raise ValueError('Waveform length differs from audited CTC metadata: ' + repr(batch['file_ids']))
        reasons = Counter(reason for item in batch['ctc_exclusion_reasons'] for reason in item)
        counts['ctc_exclusion_counts'] = dict(reasons)
        indices = eligible.nonzero(as_tuple=False).flatten()
        counts['ctc_eligible_count'] = int(indices.numel())
        counts['ctc_excluded_count'] = n - int(indices.numel())
        if indices.numel():
            chosen = targets.index_select(0, indices)
            u = target_lengths.index_select(0, indices)
            t = lengths.index_select(0, indices)
            mask = torch.arange(chosen.shape[1], device=chosen.device)[None, :] < u[:, None]
            if not (u > 0).all() or not ((chosen[mask] > 0) & (chosen[mask] < 33)).all():
                raise ValueError('Eligible CTC target is empty, contains blank, or is out of vocabulary')
            repeat_mask = torch.arange(max(0, chosen.shape[1] - 1), device=chosen.device)[None, :] < (u - 1)[:, None]
            repeats = ((chosen[:, 1:] == chosen[:, :-1]) & repeat_mask).sum(1)
            if 'ctc_minimum_frames' in batch and not torch.equal(u + repeats, batch['ctc_minimum_frames'].index_select(0, indices)):
                raise ValueError('CTC target differs from the audited repeated-character contract')
            if not ((t >= u + repeats) & (t <= logits.shape[1])).all():
                ids = [batch['file_ids'][int(i)] for i in indices.cpu().tolist()]
                raise ValueError('Eligible CTC target failed actual model frame check: ' + repr(ids))

            with torch.autocast(device_type=logits.device.type, enabled=False):
                log_probs = F.log_softmax(logits.index_select(0, indices).float(), dim=-1).transpose(0, 1).contiguous()
                raw = F.ctc_loss(log_probs, chosen, t.cpu(), u.cpu(), blank=0,
                                 reduction='none', zero_infinity=False)
                if not torch.isfinite(raw).all():
                    bad = indices[~torch.isfinite(raw)].detach().cpu().tolist()
                    raise FloatingPointError('Non-finite eligible CTC: ' + repr([batch['file_ids'][i] for i in bad]))
                ctc = (raw / u.to(raw.dtype)).mean()
            counts['ctc_target_chars'] = int(u.sum().item())

    if config['auxiliary_weights']['contrastive']:
        contrastive = outputs['contrastive_loss']
        if contrastive is None or contrastive.ndim != 0 or not torch.isfinite(contrastive):
            raise FloatingPointError('Missing/non-finite scalar instance contrastive loss')
    return ctc, contrastive, counts


def training_terms(outputs, batch, config):
    if config['variant'] != 'anchor_j':
        raise ValueError('Expected the MMSD backbone')
    main = F.cross_entropy(outputs['logits'].float(), batch['labels'])
    ctc, contrastive, counts = auxiliary_terms(outputs, batch, config)
    total = main + config['auxiliary_weights']['ctc'] * ctc + config['auxiliary_weights']['contrastive'] * contrastive
    return {'total': total, 'main_ce': main, 'ctc': ctc, 'contrastive': contrastive,
            'ctc_weighted': config['auxiliary_weights']['ctc'] * ctc,
            'contrastive_weighted': config['auxiliary_weights']['contrastive'] * contrastive,
            **counts}
