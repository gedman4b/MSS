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
    seconds = 0.1  # Reduced seconds for smaller sequence
    max_seq_len = 4096 # Further reduced max sequence length config
    # Stems
    stems = ["VOCALS", "SQUARE", "DRUMS", "BASS", "OTHER"]  # added 'SQUARE' stem
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
        # attn_mask should have shape (B, L, L) or (L, L) when batch_first=True
        # We pass (B, L, L) from GPT forward, potentially (B * n_heads, L, L)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + y
        h = self.ln2(x)
        x = x + self.ff(h)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, ff_mult, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # Use cfg.max_seq_len for positional embedding
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_seq_len, d_model))
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.n_heads = n_heads # Store n_heads for mask creation

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
        # Repeat causal mask for each head
        causal = causal.unsqueeze(0).repeat(B * self.n_heads, 1, 1)

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
def make_batch(mix_wav, stem_wav, stem_name: str):
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

    # Trim to max_seq_len
    if seq.numel() > cfg.max_seq_len:
        seq = seq[:cfg.max_seq_len]
        mask = mask[:cfg.max_seq_len]

    return seq.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), mix_phase, mix_mag.shape

@torch.no_grad()
def greedy_decode(model, mix_tokens, stem_name: str, max_new=None):
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
    # Cap generation length at max_seq_len
    max_new = min(max_new, cfg.max_seq_len - seq.size(1))
    for _ in range(max_new):
        logits, _ = model(seq)
        next_logits = logits[:, -1, :]
        next_tok = torch.argmax(next_logits, dim=-1)  # shape: (1,)
        seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
        if int(next_tok.item()) == vocab.EOS:
            break
    return seq.squeeze(0)


def reconstruct_from_sequence(seq, mix_phase, spec_shape, target_len):
    """
    Extract predicted target tokens from seq and reconstruct WAV via mixture phase.
    """
    # Locate SEP index
    sep_pos = (seq == vocab.SEP).nonzero(as_tuple=True)[0]
    if sep_pos.numel() == 0:
        print("Warning: SEP token not found in sequence. Reconstruction may be incomplete.")
        # If SEP is not found, assume the entire sequence after the prefix is target tokens
        # (2 for BOS, MIX + mix_tokens length)
        prefix_len = 2 + (mix_phase.shape[0] * mix_phase.shape[1]) # Approximate mix token length from spec shape
        start = prefix_len + 1 + 1 # after BOS, MIX, mix_tokens, SEP, STEM
        start = min(start, seq.numel()) # Ensure start is within bounds
        predicted_tokens_segment = seq[start:]
    else:
        sep_idx = sep_pos[-1].item()
        start = sep_idx + 2  # skip SEP and STEM token

        # Collect predicted tokens until EOS or end of sequence
        eos_pos = (seq[start:] == vocab.EOS).nonzero(as_tuple=True)[0]
        if eos_pos.numel() > 0:
            end = start + int(eos_pos[0].item())
        else:
            end = len(seq)
        predicted_tokens_segment = seq[start:end]

    # Determine the expected length of the target token sequence
    expected_len = spec_shape[0] * spec_shape[1]

    # Explicitly pad or trim the predicted tokens to the expected length
    if predicted_tokens_segment.numel() < expected_len:
        pad_len = expected_len - predicted_tokens_segment.numel()
        # Pad with a valid quantized bin (e.g., 0 for lowest magnitude)
        tgt_tokens_padded = torch.cat([predicted_tokens_segment, torch.full((pad_len,), 0, device=predicted_tokens_segment.device, dtype=torch.long)])
    else:
        tgt_tokens_padded = predicted_tokens_segment[:expected_len] # Trim if longer

    # Ensure the tensor is detached before dequantization if it's not already
    tgt_tokens_padded = tgt_tokens_padded.detach()

    # Dequantize to magnitude
    pred_mag = dequantize_to_mag(tgt_tokens_padded, spec_shape).to(mix_phase.device)

    # Use mixture phase for reconstruction
    wav = istft_from_mag_phase(pred_mag, mix_phase, target_len)
    return wav.clamp(-1, 1)


def demo_step(model, optimizer, mix_wav, stem_wav, stem_name="VOCALS"):
    """
    Runs one supervised step and a greedy decode.
    """
    # Build batch
    seq, mask, mix_phase, spec_shape = make_batch(mix_wav, stem_wav, stem_name)
    # Visibility into how many target tokens are being trained this step
    mask_tokens = int(mask.sum().item())
    print(f"[debug] loss_mask target tokens this step: {mask_tokens}")
    # Ensure we actually have a training signal (with our shorter STFT, we should)
    # assert mask_tokens > 0, "Loss mask is empty—target tokens were entirely truncated. Increase max_seq_len or cut STFT/tokenization further."

    model.train()
    logits, loss = model(seq, loss_mask=mask)
    optimizer.zero_grad()
    # Only call backward if loss requires grad
    if loss is not None and loss.requires_grad:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Greedy decode (eval)
    model.eval()
    # Slice of mixture tokens must also be capped at max_seq_len
    # The mix_tokens slice should match the part of the sequence that represents the mixture tokens
    # This is from after BOS and MIX tokens up to the SEP token
    sep_pos = (seq[0] == vocab.SEP).nonzero(as_tuple=True)[0]
    if sep_pos.numel() > 0:
        mix_tok_end = sep_pos[0].item()
    else:
        # If SEP not found (e.g., trimmed), approximate based on expected mix token length
        mix_tok_end = 2 + (spec_shape[0] * spec_shape[1]) # BOS, MIX + mix_tokens
        mix_tok_end = min(mix_tok_end, seq.size(1)) # Cap at actual seq length

    mix_tok = seq[0, 2:mix_tok_end] # slice after BOS and MIX tokens

    with torch.no_grad():
        full = greedy_decode(model, mix_tok, stem_name)
    # Reconstruct predicted stem
    target_len = mix_wav.numel()
    pred_wav = reconstruct_from_sequence(full, mix_phase, spec_shape, target_len)
    return float(loss.item()) if loss is not None else float('nan'), pred_wav

# =========================
# Main (toy run)
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = cfg.device
    print("Device:", device)

    # For a smoke test without a dataset, synthesize a trivial mixture:
    # mixture = sine(440Hz) + square(523Hz) + noise
    # "VOCALS" = sine(440Hz); "SQUARE" = square(523Hz)
    T = int(cfg.seconds * cfg.sr)
    t = torch.linspace(0, cfg.seconds, T, dtype=torch.float32)
    sine = 0.5*torch.sin(2*math.pi*440*t)
    square = 0.3*torch.sign(torch.sin(2*math.pi*523*t))
    noise = 0.1*torch.randn_like(sine)
    vocals = sine
    # Keep 'other' as just noise; square is now its own stem
    other = noise
    mix = (vocals + square + other).to(device)

    # Pad/trim exactly cfg.seconds
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

    path = "/content/drive/MyDrive/MSS_Audio/"

    torchaudio.save(path + "mixture_test.wav", mix.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote mixture_test.wav")
    
    # Save predicted wav for inspection
    torchaudio.save(path + "pred_vocals_demo.wav", pred.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote pred_vocals_demo.wav")

    # === Square stem: train+decode and save ===
    loss_sq, pred_sq = demo_step(model, opt, mix, square.to(device), stem_name="SQUARE")
    print("Square Loss:", loss_sq)
    torchaudio.save(path + "pred_square_demo.wav", pred_sq.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote pred_square_demo.wav")

    # === Ground-truth recon sanity checks ===
    # 1) Save the clean target (should be a pure 440 Hz tone)
    torchaudio.save(path + "target_vocals_ground_truth.wav", vocals.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote target_vocals_ground_truth.wav")

    # 2) Quantize→dequantize the stem and reconstruct with STEM PHASE (cleanest)
    with torch.no_grad():
        stem_mag_dbg, stem_phase_dbg = stft_mag_phase(vocals)
        stem_log_dbg = to_logmag(stem_mag_dbg)
        stem_tok_dbg = quantize_logmag(stem_log_dbg)
        deq_mag_dbg = dequantize_to_mag(stem_tok_dbg, stem_mag_dbg.shape).to(stem_phase_dbg.device)
        recon_gt_stemphase = istft_from_mag_phase(deq_mag_dbg, stem_phase_dbg, T).clamp(-1, 1)
    torchaudio.save(path + "recon_from_gt_tokens.wav", recon_gt_stemphase.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote recon_from_gt_tokens.wav")

    # 3) Same dequantized magnitude but use MIXTURE PHASE (slightly dirtier)
    with torch.no_grad():
        mix_mag_dbg, mix_phase_dbg = stft_mag_phase(mix)
        recon_gt_mixphase = istft_from_mag_phase(deq_mag_dbg, mix_phase_dbg, T).clamp(-1, 1)
    #torchaudio.save(path + "recon_from_gt_tokens_mixphase.wav", recon_gt_mixphase.unsqueeze(0).cpu(), cfg.sr)
    torchaudio.save(path + "recon_from_gt_tokens_mixphase.wav", recon_gt_mixphase.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote recon_from_gt_tokens_mixphase.wav")

    # === Ground-truth & recon for SQUARE stem ===
    # 1) Clean square
    torchaudio.save(path + "target_square_ground_truth.wav", square.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote target_square_ground_truth.wav")
    # 2) Quantize→dequantize square then reconstruct with SQUARE PHASE
    with torch.no_grad():
        sq_mag_dbg, sq_phase_dbg = stft_mag_phase(square.to(device))
        sq_log_dbg = to_logmag(sq_mag_dbg)
        sq_tok_dbg = quantize_logmag(sq_log_dbg)
        deq_sq_mag_dbg = dequantize_to_mag(sq_tok_dbg, sq_mag_dbg.shape).to(sq_phase_dbg.device)
        sq_recon_stemphase = istft_from_mag_phase(deq_sq_mag_dbg, sq_phase_dbg, T).clamp(-1, 1)
    torchaudio.save(path + "square_recon_from_gt_tokens.wav", sq_recon_stemphase.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote square_recon_from_gt_tokens.wav")
    # 3) Same dequantized magnitude but use MIXTURE PHASE
    with torch.no_grad():
        sq_recon_mixphase = istft_from_mag_phase(deq_sq_mag_dbg, mix_phase_dbg, T).clamp(-1, 1)
    torchaudio.save(path + "square_recon_from_gt_tokens_mixphase.wav", sq_recon_mixphase.unsqueeze(0).cpu(), cfg.sr)
    print("Wrote square_recon_from_gt_tokens_mixphase.wav")
