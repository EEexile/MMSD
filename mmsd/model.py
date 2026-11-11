"""Audio-text fusion with CTC and instance contrastive supervision."""
import hashlib
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint
from transformers import RobertaModel, Wav2Vec2Model

from .attention import AttentionPooling, BertCrossEncoder, BertSelfEncoder


def masked_mean(tokens, mask):
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("Expected tokens (B,T,D) and mask (B,T)")
    weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def extended_mask(mask, dtype):
    return (1.0 - mask[:, None, None, :].to(dtype=dtype)) * -10000.0


def mmsd_config(config):
    return SimpleNamespace(
        hidden_size=int(config.get("hidden_size", 768)),
        num_attention_heads=int(config.get("num_heads", 12)),
        intermediate_size=int(config.get("feedforward_size", 3072)),
        attention_probs_dropout_prob=float(config.get("dropout", 0.1)),
        hidden_dropout_prob=float(config.get("dropout", 0.1)),
    )


class CTCHead(nn.Module):


    def __init__(self, hidden_size=768):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(hidden_size, 256, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(512)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(512, 33)

    def forward(self, audio_hidden, feature_lengths):
        if audio_hidden.ndim != 3 or feature_lengths.shape != (audio_hidden.size(0),):
            raise ValueError("CTC requires hidden (B,T,D) and lengths (B,)")
        if feature_lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("CTC frame lengths must be integers")
        if not bool(((feature_lengths > 0) &
                     (feature_lengths <= audio_hidden.size(1))).all()):
            raise ValueError("CTC frame lengths must be in [1, padded sequence length]")

        with torch.autocast(device_type=audio_hidden.device.type, enabled=False):
            features = self.dropout(audio_hidden.float())
            packed = pack_padded_sequence(features, feature_lengths.detach().cpu(),
                                          batch_first=True, enforce_sorted=False)
            packed_hidden, _ = self.lstm(packed)
            hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True,
                                            total_length=audio_hidden.size(1))
            return self.classifier(self.activation(self.layer_norm(hidden)))


class InstanceContrastiveHead(nn.Module):


    def __init__(self, hidden_size=768):
        super().__init__()
        self.audio_projection = nn.Linear(hidden_size, 256, bias=True)
        self.text_projection = nn.Linear(hidden_size, 256, bias=True)
        self.temperature = 0.07

    def forward(self, audio_hidden, text_hidden, audio_mask, text_mask):
        if audio_hidden.size(0) != text_hidden.size(0) or audio_hidden.size(0) == 0:
            raise ValueError("Instance CL requires the same nonempty A/T batch")
        if not bool(audio_mask.bool().any(-1).all() and text_mask.bool().any(-1).all()):
            raise ValueError("Instance CL cannot pool an empty audio/text sequence")
        with torch.autocast(device_type=audio_hidden.device.type, enabled=False):
            audio = F.normalize(self.audio_projection(masked_mean(audio_hidden.float(), audio_mask)),
                                p=2, dim=-1)
            text = F.normalize(self.text_projection(masked_mean(text_hidden.float(), text_mask)),
                               p=2, dim=-1)
            similarity = audio @ text.transpose(0, 1) / self.temperature
            positive = torch.arange(audio.size(0), device=audio.device)

            return (F.cross_entropy(similarity, positive) +
                    F.cross_entropy(similarity.transpose(0, 1), positive)) / 2.0


def auxiliary_state_sha256(modules):

    digest = hashlib.sha256()
    found = False
    for prefix, module in sorted(modules.items()):
        if module is None:
            continue
        for name, value in sorted(module.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update((prefix + "." + name + "|" + str(value.dtype) + "|" +
                           str(tuple(value.shape)) + "\n").encode("utf-8"))
            digest.update(value.numpy().tobytes())
            found = True
    return digest.hexdigest() if found else None


class AnchorFusion(nn.Module):


    def __init__(self, config):
        super().__init__()
        cfg = mmsd_config(config)
        d = cfg.hidden_size
        self.gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
        self.audio2text_v2 = nn.Linear(d, d)
        self.text2audio_attention = BertCrossEncoder(cfg, layer_num=1)
        self.audio2text_attention = BertCrossEncoder(cfg, layer_num=1)
        self.text2text_attention = BertCrossEncoder(cfg, layer_num=1)
        self.gate = nn.Linear(d * 2, d)
        self.ser_attention_pooling = AttentionPooling(d * 2, 256, cfg.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Dropout(float(config.get("head_dropout", 0.2))),
            nn.Linear(d * 2, int(config.get("num_classes", 4))),
        )

    def _cross(self, module, query, context, mask):
        if self.training and self.gradient_checkpointing and (query.requires_grad or context.requires_grad):
            return checkpoint(lambda q, c: module(q, c, mask)[-1],
                              query, context, use_reentrant=False)
        return module(query, context, mask)[-1]

    def forward(self, audio_hidden, text_enhanced, audio_mask, text_mask,
                audio_projected=None):
        if audio_projected is None:
            raise ValueError("AnchorFusion requires the common audio2text projection")

        dtype = next(self.parameters()).dtype
        am = extended_mask(audio_mask, dtype)
        tm = extended_mask(text_mask, dtype)
        text2audio_features = self._cross(self.text2audio_attention, text_enhanced, audio_projected, am)
        audio2text_features = self._cross(self.audio2text_attention, self.audio2text_v2(audio_hidden), text_enhanced, tm)
        text_final = self._cross(self.text2text_attention, text_enhanced, audio2text_features, am)
        merge = torch.cat((text_final, text2audio_features), dim=-1)
        gated_audio = torch.sigmoid(self.gate(merge)) * text2audio_features
        fused_features = torch.cat((text_final, gated_audio), dim=-1)
        pooled, _ = self.ser_attention_pooling(fused_features, attention_mask=text_mask)
        return {"logits": self.classifier(pooled), "pooled": pooled}


class Model(nn.Module):


    def __init__(self, config, audio_encoder=None, text_encoder=None):
        super().__init__()
        self.config = dict(config)
        self.variant = config["variant"]
        if self.variant != "anchor_j":
            raise ValueError("Unknown variant: " + self.variant)
        cfg = mmsd_config(config)
        self.gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
        self.wav2vec2 = audio_encoder if audio_encoder is not None else Wav2Vec2Model.from_pretrained(
            config["wav2vec2_path"], local_files_only=True,
            output_hidden_states=False, return_dict=True, apply_spec_augment=False,
        )
        self.roberta = text_encoder if text_encoder is not None else RobertaModel.from_pretrained(
            config["roberta_path"], local_files_only=True, add_pooling_layer=False,
        )
        if config.get("freeze_audio_extractor", True):
            for p in self.wav2vec2.feature_extractor.parameters():
                p.requires_grad_(False)

        if hasattr(self.wav2vec2, "masked_spec_embed"):
            self.wav2vec2.masked_spec_embed.requires_grad_(False)
        if config.get("freeze_text_embeddings", True):
            for p in self.roberta.embeddings.parameters():
                p.requires_grad_(False)
        for layer in self.roberta.encoder.layer[:int(config.get("freeze_text_layers", 6))]:
            for p in layer.parameters():
                p.requires_grad_(False)
        if self.gradient_checkpointing:
            for encoder in (self.wav2vec2, self.roberta):

                if hasattr(encoder, "gradient_checkpointing_enable"):
                    encoder.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False})


        self.self_attention_text = BertSelfEncoder(cfg)
        self.audio2text = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.fusion = AnchorFusion(config)

        weights = config.get("auxiliary_weights", {})
        self.ctc_head = None
        self.contrastive_head = None
        if float(weights.get("ctc", 0.0)) != 0 or float(weights.get("contrastive", 0.0)) != 0:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(config.get("auxiliary_init_seed", 4242)))
                if float(weights.get("ctc", 0.0)) != 0:
                    self.ctc_head = CTCHead(cfg.hidden_size)
                if float(weights.get("contrastive", 0.0)) != 0:
                    self.contrastive_head = InstanceContrastiveHead(cfg.hidden_size)
        self.auxiliary_initial_sha256 = auxiliary_state_sha256({
            "ctc_head": self.ctc_head, "contrastive_head": self.contrastive_head,
        })

    def forward(self, batch, compute_auxiliary=None):
        audio = batch["audio_inputs"]
        lengths = batch["audio_lengths"].to(device=audio.device)
        audio_attention_mask = (torch.arange(audio.size(1), device=audio.device)[None, :] < lengths[:, None]).long()
        audio_hidden = self.wav2vec2(audio, attention_mask=audio_attention_mask).last_hidden_state
        feature_lengths = self.wav2vec2._get_feat_extract_output_lengths(lengths).long()
        if feature_lengths.shape != (audio.size(0),) or not bool(
                ((feature_lengths > 0) & (feature_lengths <= audio_hidden.size(1))).all()):
            raise ValueError("Invalid Wav2Vec2 effective feature lengths")
        audio_mask = (torch.arange(audio_hidden.size(1), device=audio.device)[None, :] < feature_lengths[:, None])
        text_mask = batch["text_attention_mask"]
        text_hidden = self.dropout(self.roberta(
            batch["text_input_ids"], attention_mask=text_mask,
        ).last_hidden_state)
        tm = extended_mask(text_mask, next(self.parameters()).dtype)
        if self.training and self.gradient_checkpointing and text_hidden.requires_grad:
            text_enhanced = checkpoint(lambda t: self.self_attention_text(t, tm)[-1],
                                       text_hidden, use_reentrant=False)
        else:
            text_enhanced = self.self_attention_text(text_hidden, tm)[-1]
        audio_projected = self.audio2text(audio_hidden)
        fused = self.fusion(audio_hidden, text_enhanced, audio_mask, text_mask, audio_projected)
        result = {"logits": fused["logits"], "joint_logits": fused["logits"],
                  "expert_logits": None, "pooled": fused["pooled"]}

        compute_auxiliary = self.training if compute_auxiliary is None else bool(compute_auxiliary)
        result.update(ctc_logits=None, ctc_input_lengths=None, contrastive_loss=None)
        if compute_auxiliary:
            if self.ctc_head is not None:
                result["ctc_logits"] = self.ctc_head(audio_hidden, feature_lengths)
                result["ctc_input_lengths"] = feature_lengths
            if self.contrastive_head is not None:
                result["contrastive_loss"] = self.contrastive_head(
                    audio_hidden, text_hidden, audio_mask, text_mask)
        return result

    def parameter_summary(self):
        groups = {}
        for name, parameter in self.named_parameters():
            group = groups.setdefault(name.split(".")[0], {"total": 0, "trainable": 0})
            group["total"] += parameter.numel()
            group["trainable"] += parameter.numel() if parameter.requires_grad else 0
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "groups": groups}
