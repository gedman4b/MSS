# sep_gpt_v0.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# =========================
# Config (minimal recipe)
# =========================
class Cfg:
    sr = 44100
    n_fft = 1024
    hop = 256             # 75% overlap
    win = 1024
    pad = 0
    center = True
    power = 1.0           # magnitude (not power) for STFT
    # Tokenization range on log10 magnitude (clipped)
    logmag_min_db = -12.0  # ~ -12 dB relative floor after per-window normalize
    logmag_max_db = 2.0
    n_bits = 8            # 8-bit uniform quant
    # Model
    d_model = 512
    n_heads = 8
    n_layers = 8
    ff_mult = 4
    dropout = 0.1
    # Windows
    seconds = 2.0
    # Stems
    stems = ["VOCALS", "DRUMS", "BASS", "OTHER"]  # control tokens
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Cfg()

# =========================
# Helper: STFT / iSTFT
# =========================
def stft_mag_phase(wav: torch.Tensor):
    """
    wav: (T,) mono tensor, float32 [-1,1]
    returns: mag (frames, freq), phase (frames, freq)
    """
    window = torch.hann_window(cfg.win, device=wav.device)
    stft = torch.stft(
        wav,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        win_length=cfg.win,
        window=window,
        center=cfg.center,
        return_complex=True,
        pad_mode="reflect",
    )  # (freq, frames)
    spec = stft.transpose(0,1)          # (frames, freq)
    mag = spec.abs()
    phase = torch.angle(spec)
    return mag, phase

def istft_from_mag_phase(mag: torch.Tensor, phase: torch.Tensor, length: int):
    """
    mag/phase: (frames, freq)
    length: samples to trim/pad to
    """
    spec = (mag * torch.exp(1j*phase)).transpose(0,1)  # (freq, frames)
    window = torch.hann_window(cfg.win, device=mag.device)
    wav = torch.istft(
        spec,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop,
        win_length=cfg.win,
        window=window,
        center=cfg.center,
        length=length,
    )
    return wav

# =========================
# Quantization (log-mag)
# =========================
def to_logmag(mag: torch.Tensor, eps=1e-8):
    # Per-window normalize: scale by max to stabilize, then log10
    m = torch.clamp(mag / (mag.max() + eps), min=eps)
    logm = torch.log10(m)
    # map to "dB-ish" range by multiplying 20 (optional), but here we treat log10 directly.
    # We'll just clamp in fixed range (already tuned for normalized mags).
    logm = torch.clamp(logm, math.log10(10**(cfg.logmag_min_db/20)),
                             math.log10(10**(cfg.logmag_max_db/20)))
    return logm

def quantize_logmag(logm: torch.Tensor):
    """
    logm: (frames, freq)
    returns: tokens (frames*freq,)
    """
    lo = math.log10(10**(cfg.logmag_min_db/20))
    hi = math.log10(10**(cfg.logmag_max_db/20))
    qlevels = (1 << cfg.n_bits)
    x = (logm - lo) / (hi - lo)  # [0,1]
    x = torch.clamp(x, 0, 1)
    q = torch.round(x * (qlevels - 1)).to(torch.long)  # [0..255]
    return q.view(-1)

def dequantize_to_mag(tokens: torch.Tensor, shape):
    """
    tokens: (frames*freq,)
    shape: (frames, freq)
    returns: mag (frames, freq)
    """
    lo = math.log10(10**(cfg.logmag_min_db/20))
    hi = math.log10(10**(cfg.logmag_max_db/20))
    qlevels = (1 << cfg.n_bits)
    x = tokens.float() / (qlevels - 1)
    logm = x * (hi - lo) + lo
    m = 10 ** logm  # invert log10 amplitude
    return m.view(*shape)

# =========================
# Token vocab & packing
# =========================
class Vocab:
    # quantized bins 0..255
    # special/control tokens appended after
    PAD = 256
    BOS = 257
    EOS = 258
    SEP = 259
    MIX = 260
    STEM_BASE = 300  # STEM tokens at STEM_BASE + idx

    def __init__(self):
        self.num_bins = 1 << cfg.n_bits
        self.stem_to_id = {name: self.STEM_BASE + i for i, name in enumerate(cfg.stems)}
        self.id_to_stem = {v:k for k,v in self.stem_to_id.items()}
        self.vocab_size = self.STEM_BASE + len(cfg.stems)

vocab = Vocab()

def pack_sequence(mix_tokens: torch.Tensor, stem_name: str, target_tokens: torch.Tensor):
    """
    Create LM sequence:
    [BOS, MIX, mix..., SEP, STEM(token), target..., EOS]
    Returns tokens (L,), loss_mask (L,) where mask=1 for target positions only.
    """
    stem_tok = torch.tensor([vocab.stem_to_id[stem_name]], device=mix_tokens.device, dtype=torch.long)
    seq = torch.cat([
        torch.tensor([vocab.BOS, vocab.MIX], device=mix_tokens.device),
        mix_tokens,
        torch.tensor([vocab.SEP], device=mix_tokens.device),
        stem_tok,
        target_tokens,
        torch.tensor([vocab.EOS], device=mix_tokens.device)
    ])
    # Loss mask: predict from position after STEM token inclusive
    loss_mask = torch.zeros_like(seq, dtype=torch.bool)
    # positions where target tokens + EOS reside
    start = (2 + mix_tokens.numel() + 1 + 1)  # after BOS,MIX + mix + SEP + STEM
    loss_mask[start:] = True
    return seq, loss_mask

def split_mix_target_from_seq(seq: torch.Tensor):
    # helper for debugging; not needed in training loop
    pass

# =========================
# GPT-style decoder-only model
# =========================
class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult*d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x, attn_mask):
        h = self.ln1(x)
        # attn_mask: (L,L) with True = block attention (causal)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + y
        h = self.ln2(x)
        x = x + self.ff(h)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, ff_mult, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 32768, d_model))  # big cap; or switch to RoPE
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, loss_mask=None):
        """
        tokens: (B, L) long
        loss_mask: (B, L) bool (optional)
        returns: logits (B, L, V), loss (scalar) if mask provided
        """
        B, L = tokens.shape
        max_pos = self.pos.size(1)
        if L > max_pos:
            tokens = tokens[:, :max_pos]
            if loss_mask is not None:
                loss_mask = loss_mask[:, :max_pos]
            L = max_pos
        x = self.embed(tokens) + self.pos[:, :L, :]
        # causal mask: no lookahead
        causal = torch.triu(torch.ones(L, L, device=tokens.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, causal)
        x = self.ln(x)
        logits = self.out(x)
        loss = None
        if loss_mask is not None:
            # shift targets by one (next token prediction)
            logits_shift = logits[:, :-1, :].contiguous()
            targets = tokens[:, 1:].contiguous()
            mask = loss_mask[:, 1:].contiguous()
            # Only compute loss if mask selects at least one position
            if mask.view(-1).sum() > 0:
                loss = F.cross_entropy(
                    logits_shift.view(-1, logits.size(-1))[mask.view(-1)],
                    targets.view(-1)[mask.view(-1)]
                )
            else:
                loss = torch.tensor(0.0, device=logits.device)
        return logits, loss

# =========================
# Data demo utilities
# =========================
def pad_or_trim(wav, length):
    T = wav.numel()
    if T >= length:
        return wav[:length]
    out = torch.zeros(length, device=wav.device)
    out[:T] = wav
    return out

def mono(wav):
    if wav.dim() == 2:
        return wav.mean(0)
    return wav

def load_wav(path, target_sr):
    wav, sr = torchaudio.load(path)
    wav = mono(wav)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.clamp(-1, 1)
    return wav

# =========================
# End-to-end: one batch example
# =========================
def make_batch(mix_wav, stem_wav, stem_name):
    """
    Build a (1, L) token batch and loss mask for one 2s window.
    """
    device = cfg.device
    # STFTs
    mix_mag, mix_phase = stft_mag_phase(mix_wav)
    stem_mag, _ = stft_mag_phase(stem_wav)

    # Tokenize
    mix_log = to_logmag(mix_mag)
    stem_log = to_logmag(stem_mag)
    mix_tok = quantize_logmag(mix_log)
    stem_tok = quantize_logmag(stem_log)

    # Pack
    seq, mask = pack_sequence(mix_tok, stem_name, stem_tok)
    return seq.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), mix_phase, mix_mag.shape

@torch.no_grad()
def greedy_decode(model, mix_tokens, stem_name, max_new=None):
    """
    mix_tokens: (Lmix,) long
    Returns full sequence including prefix+generated tokens.
    """
    device = cfg.device
    stem_tok = torch.tensor([vocab.stem_to_id[stem_name]], device=device, dtype=torch.long)
    seq = torch.cat([
        torch.tensor([vocab.BOS, vocab.MIX], device=device),
        mix_tokens.to(device),
        torch.tensor([vocab.SEP], device=device),
        stem_tok
    ])
    seq = seq.unsqueeze(0)  # (1, L)
    # generate until EOS or max_new
    max_new = max_new or (mix_tokens.numel() + 1024)  # cap
    for _ in range(max_new):
        logits, _ = model(seq)
        next_logits = logits[:, -1, :]
        next_tok = torch.argmax(next_logits, dim=-1)  # shape: (1,)
        seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)  # fix shape here
        if int(next_tok.item()) == vocab.EOS:
            break
    return seq.squeeze(0)

def reconstruct_from_sequence(seq, mix_phase, spec_shape, target_len):
    """
    Extract predicted target tokens from seq and reconstruct WAV via mixture phase.
    """
    # Locate SEP index robustly
    sep_pos = (seq == vocab.SEP).nonzero(as_tuple=True)[0]
    if sep_pos.numel() == 0:
        raise ValueError("SEP token not found in sequence")
    sep_idx = sep_pos[-1].item()
    start = sep_idx + 2  # skip SEP and STEM token

    # Collect until EOS, but clamp to expected length
    eos_pos = (seq[start:] == vocab.EOS).nonzero(as_tuple=True)[0]
    max_target = spec_shape[0] * spec_shape[1]
    if eos_pos.numel() > 0:
        end = start + int(eos_pos[0].item())
    else:
        end = min(start + max_target, len(seq))
    tgt_tokens = seq[start:end]
    # Dequantize to magnitude
    pred_mag = dequantize_to_mag(tgt_tokens, spec_shape).to(mix_phase.device)
    # Use mixture phase for reconstruction
    wav = istft_from_mag_phase(pred_mag, mix_phase, target_len)
    return wav.clamp(-1, 1)

def demo_step(model, optimizer, mix_wav, stem_wav, stem_name="VOCALS"):
    """
    Runs one supervised step and a greedy decode.
    """
    # Build batch
    seq, mask, mix_phase, spec_shape = make_batch(mix_wav, stem_wav, stem_name)
    model.train()
    logits, loss = model(seq, loss_mask=mask)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Greedy decode (eval)
    model.eval()
    mix_tok = seq[0, 2:2+spec_shape[0]*spec_shape[1]]  # slice of mixture tokens
    with torch.no_grad():
        full = greedy_decode(model, mix_tok, stem_name)
    # Reconstruct predicted stem
    target_len = mix_wav.numel()
    pred_wav = reconstruct_from_sequence(full, mix_phase, spec_shape, target_len)
    return float(loss.item()), pred_wav

# =========================
# Main (toy run)
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = cfg.device
    print("Device:", device)

    # For a smoke test without a dataset, synthesize a trivial mixture:
    # mixture = sine(220Hz) + noise; "VOCALS" = sine; "OTHER" = noise
    T = int(cfg.seconds * cfg.sr)
    t = torch.linspace(0, cfg.seconds, T, dtype=torch.float32)
    sine = 0.5*torch.sin(2*math.pi*220*t)
    noise = 0.1*torch.randn_like(sine)
    vocals = sine
    other = noise
    mix = (vocals + other).to(device)

    # Pad/trim exactly 2s
    mix = pad_or_trim(mix, T)
    vocals = pad_or_trim(vocals.to(device), T)

    # Model
    model = GPT(
        vocab_size=vocab.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ff_mult=cfg.ff_mult,
        dropout=cfg.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    # One training step + decode
    loss, pred = demo_step(model, opt, mix, vocals, stem_name="VOCALS")
    print("Loss:", loss)

    # Save predicted wav for inspection
    torchaudio.save("pred_vocals_demo.wav", pred.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote pred_vocals_demo.wav")
