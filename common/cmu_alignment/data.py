"""Formal CMU ARCTIC full-utterance data path.

One module deliberately owns frozen-V2 feature reproduction, full-utterance
reference construction, the manifest reader, and time-only batch padding.
There is no crop recipe or training-time K construction in this module.
"""
from __future__ import annotations

import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from cmu_alignment.core import read_jsonl, sha256_file, write_jsonl

PAIR_ORDER = tuple((i, j) for i in range(4) for j in range(4) if i != j)
SILENCE = {"pau", "sil", "h#"}
EXPECTED_COUNTS = {"train": 2229, "val": 474, "test": 469}


def load_crf_reference_length(root: str | Path) -> int:
    """Read the frozen train-only unique-utterance calibration metadata."""
    value=json.loads((Path(root)/"metadata/crf_reference_length.json").read_text(encoding="utf8"))
    if value.get("source") != "unique_train_utterance_native_lengths" or int(value["crf_reference_length"]) < 2: raise ValueError("invalid CRF reference-length provenance")
    return int(value["crf_reference_length"])


@dataclass(frozen=True)
class PhoneInterval:
    phone: str
    start_sec: float
    end_sec: float


def parse_lab(path: str | Path) -> list[PhoneInterval]:
    """Parse HTK-style cumulative phone end times without changing labels."""
    previous, output = 0.0, []
    for line in Path(path).read_text(encoding="utf8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"invalid LAB line: {path}")
        end = float(fields[0])
        if end < previous:
            raise ValueError(f"non-monotone LAB: {path}")
        output.append(PhoneInterval(fields[-1].lower(), previous, end))
        previous = end
    return output


def content_phones(phones: list[PhoneInterval]) -> list[PhoneInterval]:
    """Drop only external silence; internal silence remains reference content."""
    left, right = 0, len(phones)
    while left < right and phones[left].phone in SILENCE:
        left += 1
    while right > left and phones[right - 1].phone in SILENCE:
        right -= 1
    return phones[left:right]


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1 or stream.getsampwidth() not in {2, 4}:
            raise ValueError("expected mono PCM16/PCM32 WAV")
        kind = np.int16 if stream.getsampwidth() == 2 else np.int32
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), kind)
        return pcm.astype(np.float32) / np.iinfo(kind).max, stream.getframerate()


FROZEN_V2_MFCC = {
    "sample_frequency": 16000.0, "frame_length": 25.0, "frame_shift": 10.0,
    "dither": 0.0, "preemphasis_coefficient": 0.97, "remove_dc_offset": True,
    "round_to_power_of_two": True, "window_type": "povey", "num_mel_bins": 23,
    "low_freq": 20.0, "high_freq": 7800.0, "num_ceps": 13, "use_energy": False,
    "raw_energy": False, "energy_floor": 0.0, "cepstral_lifter": 22.0,
    "snip_edges": True,
}


def extract_mfcc13_raw(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Frozen V2 Kaldi-compatible MFCC13 frontend; no hidden resampling."""
    if sample_rate != 16000:
        raise ValueError(f"frozen V2 expects 16 kHz WAV, received {sample_rate}")
    try:
        import torchaudio
    except Exception as error:  # pragma: no cover - environment boundary
        raise RuntimeError("torchaudio.compliance.kaldi is required for V2 reproduction") from error
    value = torchaudio.compliance.kaldi.mfcc(torch.as_tensor(np.asarray(waveform, dtype=np.float32))[None], **FROZEN_V2_MFCC)
    output = value.cpu().numpy().astype(np.float32, copy=False)
    if output.ndim != 2 or output.shape[1] != 13 or not np.isfinite(output).all():
        raise ValueError("Kaldi MFCC frontend produced invalid MFCC13")
    return output


def assert_mfcc39(value: np.ndarray) -> None:
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != 39 or not np.isfinite(value).all():
        raise ValueError("MFCC39 must be finite float32 [T,39]")


def static_cmn(value: np.ndarray, speaker_mean: np.ndarray) -> np.ndarray:
    if value.ndim != 2 or value.shape[1] != 13 or speaker_mean.shape != (13,):
        raise ValueError("static speaker CMN requires [T,13] and [13]")
    return (value - speaker_mean).astype(np.float32)


def fit_train_speaker_cmn(records: Iterable[tuple[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Fit static CMN means only from records explicitly supplied as train."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for speaker, value in records:
        if value.ndim != 2 or value.shape[1] != 13:
            raise ValueError("CMN input must be MFCC13")
        sums[speaker] = sums.get(speaker, np.zeros(13, np.float64)) + value.sum(0)
        counts[speaker] = counts.get(speaker, 0) + len(value)
    return {speaker: (sums[speaker] / counts[speaker]).astype(np.float32) for speaker in sums}


def kaldi_deltas(value: np.ndarray, window: int = 2) -> np.ndarray:
    if value.ndim != 2 or window != 2:
        raise ValueError("frozen V2 delta convention requires [T,C], window=2")
    padded = np.pad(value, ((window, window), (0, 0)), mode="edge")
    numerator = sum(n * (padded[window+n:window+n+len(value)] - padded[window-n:window-n+len(value)]) for n in range(1, window+1))
    return (numerator / (2 * sum(n*n for n in range(1, window+1)))).astype(np.float32)


def make_mfcc39(static_normalized: np.ndarray) -> np.ndarray:
    result = np.concatenate((static_normalized, kaldi_deltas(static_normalized), kaldi_deltas(kaldi_deltas(static_normalized))), axis=1).astype(np.float32)
    assert_mfcc39(result)
    return result


def reference_coordinate(frame_centers_sec: np.ndarray, phones: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Content-phone coordinate at centers; external frames remain input only."""
    if not phones:
        raise ValueError("eligible utterance must have content phones")
    starts = np.asarray([p["start_sec"] for p in phones], np.float64)
    ends = np.asarray([p["end_sec"] for p in phones], np.float64)
    canon = np.asarray([p.get("canonical_index", i) for i, p in enumerate(phones)], np.float64)
    if np.any(ends <= starts) or np.any(starts[1:] < ends[:-1]):
        raise ValueError("invalid content-phone timing")
    centers = np.asarray(frame_centers_sec, np.float64)
    indices = np.searchsorted(ends, centers, side="right")
    valid = (centers >= starts[0]) & (centers < ends[-1]) & (indices < len(phones))
    valid &= centers >= starts[np.minimum(indices, len(phones)-1)]
    coordinate = np.full(len(centers), -1.0, np.float32)
    ii = indices[valid]
    coordinate[valid] = (canon[ii] + (centers[valid] - starts[ii]) / (ends[ii] - starts[ii])).astype(np.float32)
    values = coordinate[valid]
    if len(values) < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("reference frame coordinates must be strictly increasing on valid frames")
    return coordinate, valid


def _ragged(values: list[np.ndarray], dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for value in values: offsets.append(offsets[-1] + len(value))
    return np.concatenate([np.asarray(value, dtype=dtype) for value in values]), np.asarray(offsets, np.int64)


def _full_pair_gt(reference_u: list[np.ndarray], reference_valid: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    qs, supports = [], []
    intervals = np.full((4, 4, 2), -1, np.int64)
    for source_id, target_id in PAIR_ORDER:
        source_u, source_ok = reference_u[source_id], reference_valid[source_id]
        target_u, target_ok = reference_u[target_id], reference_valid[target_id]
        target_indices, target_coordinate = np.flatnonzero(target_ok), target_u[target_ok]
        support = source_ok & (source_u >= target_coordinate[0]) & (source_u <= target_coordinate[-1])
        chosen = np.flatnonzero(support)
        if len(chosen) < 2 or not support[chosen[0]:chosen[-1]+1].all():
            raise ValueError("each full pair needs contiguous support of at least two frames")
        q = np.full(len(source_u), -1.0, np.float32)
        q[support] = np.interp(source_u[support], target_coordinate, target_indices.astype(np.float64)).astype(np.float32)
        intervals[source_id, target_id] = (chosen[0], chosen[-1]+1)
        qs.append(q); supports.append(support)
    return qs, supports, intervals


def common_k4_content_valid(reference_u: torch.Tensor, reference_valid: torch.Tensor) -> torch.Tensor:
    """Return the labelled semantic-content intersection of all four speakers.

    ``reference_u`` is the materialised content-phone coordinate at every
    *native* MFCC frame.  A K=4 group may have different leading/trailing
    external-silence duration per speaker, so a frame is group-common only
    when its labelled content coordinate lies in the intersection of the four
    speakers' labelled coordinate ranges.  This does not crop, renumber, or
    interpolate any sequence: the returned mask remains indexed by original
    MFCC frame positions.
    """
    if reference_u.ndim != 3 or reference_u.shape[1] != 4 or reference_valid.shape != reference_u.shape:
        raise ValueError("reference_u/reference_valid must be [B,4,L]")
    valid = reference_valid.bool()
    if (~valid).all(dim=-1).any():
        raise ValueError("every CMU sequence requires labelled content frames")
    lower_per_curve = reference_u.masked_fill(~valid, torch.inf).amin(dim=-1)
    upper_per_curve = reference_u.masked_fill(~valid, -torch.inf).amax(dim=-1)
    lower = lower_per_curve.amax(dim=1, keepdim=True)
    upper = upper_per_curve.amin(dim=1, keepdim=True)
    common = valid & (reference_u >= lower[..., None]) & (reference_u <= upper[..., None])
    if (common.sum(dim=-1) < 2).any():
        raise ValueError("every K4 member requires at least two frames in the shared labelled content domain")
    return common


def common_k4_pair_valid(
    q_of_p: torch.Tensor,
    pair_valid: torch.Tensor,
    group_common_valid: torch.Tensor,
) -> torch.Tensor:
    """Restrict each directed GT path to the K=4 common labelled domain.

    A source index must be in the common domain and its existing GT target
    coordinate must land inside the target's common native-index interval.
    Coordinates themselves are left untouched; this only refines the boolean
    supervision/evaluation mask from the stored CMU phone alignment.
    """
    if q_of_p.ndim != 4 or pair_valid.shape != q_of_p.shape or group_common_valid.shape != q_of_p.shape[:2] + q_of_p.shape[-1:]:
        raise ValueError("q_of_p/pair_valid/group_common_valid shapes are inconsistent")
    if (group_common_valid.sum(dim=-1) < 2).any():
        raise ValueError("group common domain must contain at least two native frames")
    width = q_of_p.shape[-1]
    indices = torch.arange(width, device=q_of_p.device)
    starts = group_common_valid.to(torch.int64).argmax(dim=-1)
    counts = group_common_valid.sum(dim=-1).to(torch.long)
    ends = starts + counts - 1
    source_common = group_common_valid[:, :, None, :]
    target_start = starts[:, None, :, None].to(q_of_p.dtype)
    target_end = ends[:, None, :, None].to(q_of_p.dtype)
    target_inside = (q_of_p >= target_start) & (q_of_p <= target_end)
    refined = pair_valid.bool() & source_common & target_inside
    # The source-domain subset of every CMU pair remains a contiguous native
    # interval.  This catches provenance or phone-boundary inconsistencies
    # before a CRF is trained against them.
    for source, target in PAIR_ORDER:
        mask = refined[:, source, target]
        count = mask.sum(dim=-1)
        begin = mask.to(torch.int64).argmax(dim=-1)
        expected = (indices[None] >= begin[:, None]) & (indices[None] < (begin + count)[:, None])
        if (count < 2).any() or not torch.equal(mask, expected):
            raise ValueError(f"K4-common CMU support for {source}->{target} must be contiguous and have at least two frames")
    return refined


def _quantiles(values: list[float | int]) -> dict[str, float]:
    value = np.asarray(values, np.float64)
    return {"min": float(value.min()), "p01": float(np.quantile(value,.01)), "p05": float(np.quantile(value,.05)), "median": float(np.median(value)), "p95": float(np.quantile(value,.95)), "p99": float(np.quantile(value,.99)), "max": float(value.max()), "mean": float(value.mean()), "std": float(value.std())}


def build_full_alignment_dataset(k4_root: str | Path, native_v2_root: str | Path, out_root: str | Path, overwrite: bool = False) -> dict[str, Any]:
    """Materialize full-utterance GT only; frozen V2 MFCC cache is referenced."""
    k4, native, output = Path(k4_root).resolve(), Path(native_v2_root).resolve(), Path(out_root).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite: raise FileExistsError(output)
    for name in ("manifests", "gt", "metadata"): (output/name).mkdir(parents=True, exist_ok=True)
    phones = {row["utterance_id"]: row for row in read_jsonl(native/"interim/phone_alignment_index.jsonl")}
    utterances = {row["utterance_id"]: row for row in read_jsonl(native/"interim/utterance_index.jsonl")}
    all_rows: list[dict[str, Any]] = []; split_summary: dict[str, Any] = {}
    forbidden = {"crop_recipe_id", "crop_frame_start", "crop_frame_end_exclusive", "crop_lengths", "coverage_start_u", "coverage_end_u", "source_feature_root"}
    for split, expected in EXPECTED_COUNTS.items():
        rows = []
        for upstream in read_jsonl(k4/f"manifests/{split}_k4_groups.jsonl"):
            if len(upstream["sequence_ids"]) != 4 or len(set(upstream["sequence_ids"])) != 4 or len(set(upstream["speaker_ids"])) != 4: raise ValueError(f"bad K4 {upstream['k4_group_id']}")
            if forbidden & upstream.keys(): raise ValueError("full manifest cannot inherit crop fields")
            refs, ref_valid = [], []
            for uid, relpath, length in zip(upstream["sequence_ids"], upstream["feature_paths"], upstream["native_lengths"], strict=True):
                with np.load(native/relpath) as cache: feature, centers = cache["mfcc39"], cache["frame_center_sec"]
                assert_mfcc39(feature)
                if len(feature) != length or len(centers) != length: raise ValueError(f"V2 provenance mismatch: {uid}")
                u, ok = reference_coordinate(centers, phones[uid]["content_phones"]); refs.append(u); ref_valid.append(ok)
            q, support, interval = _full_pair_gt(refs, ref_valid)
            uv, uo = _ragged(refs, np.float32); rv, ro = _ragged(ref_valid, np.bool_); qv, qo = _ragged(q, np.float32); pv, po = _ragged(support, np.bool_)
            name = f"{upstream['k4_group_id']}.npz"
            np.savez_compressed(output/"gt"/name, reference_u_values=uv, reference_u_offsets=uo, reference_valid_values=rv, reference_valid_offsets=ro, q_values=qv, q_offsets=qo, pair_valid_values=pv, pair_valid_offsets=po, valid_interval=interval, pair_order=np.asarray(PAIR_ORDER, np.int64))
            rows.append({**upstream, "duration_sec": [float(utterances[uid]["duration_sec"]) for uid in upstream["sequence_ids"]], "gt_path": f"gt/{name}", "full_utterance": True, "input_representation": "native_mfcc39_full_utterance", "reference_representation": "content_phone_coordinate_at_frame_centers"})
        if len(rows) != expected: raise RuntimeError(f"{split} expected {expected}, got {len(rows)}")
        write_jsonl(output/f"manifests/{split}_full_alignment.jsonl", rows); all_rows.extend(rows)
        split_summary[split] = {"k4_groups": len(rows), "parents": len({r['parent_prompt_id'] for r in rows}), "unique_utterances": len({u for r in rows for u in r['sequence_ids']}), "unique_speakers": len({s for r in rows for s in r['speaker_ids']})}
    write_jsonl(output/"manifests/all_full_alignment.jsonl", all_rows)
    if len(all_rows) != sum(EXPECTED_COUNTS.values()): raise RuntimeError("total K4 count mismatch")
    unique_lengths, unique_durations = {}, {}
    for row in all_rows:
        for uid, length, duration in zip(row["sequence_ids"], row["native_lengths"], row.get("duration_sec", [None]*4), strict=True):
            unique_lengths[uid] = int(length)
            if duration is not None: unique_durations[uid] = float(duration)
    pair_cells = {split: sum(sum(int(a)*int(b) for i,a in enumerate(r["native_lengths"]) for j,b in enumerate(r["native_lengths"]) if i != j) for r in all_rows if r["split"] == split) for split in EXPECTED_COUNTS}
    hashes = {"k4_all_manifest_sha256": sha256_file(k4/"manifests/all_k4_groups.jsonl"), "native_phone_index_sha256": sha256_file(native/"interim/phone_alignment_index.jsonl"), "native_feature_config_sha256": sha256_file(native/"metadata/acoustic_feature_reference_v2.yaml")}
    train_unique={uid:int(length) for row in all_rows if row["split"]=="train" for uid,length in zip(row["sequence_ids"],row["native_lengths"],strict=True)}
    reference_stats=_quantiles(list(train_unique.values()))
    summary = {"dataset": "CMU_ARCTIC_MFCC39_K4_FULL_ALIGNMENT_V1", "counts": {**split_summary, "total_k4_groups": len(all_rows), "total_parent_prompts": len({r['parent_prompt_id'] for r in all_rows})}, "unique_full_utterance_T": _quantiles(list(unique_lengths.values())), "unique_full_duration_sec": _quantiles(list(unique_durations.values())), "crf_reference_length": {"value":int(round(reference_stats["median"])),"source":"unique_train_utterance_native_lengths","n_unique_train_utterances":len(train_unique),"stats":reference_stats}, "directed_pair_cells": pair_cells, "source_hashes": hashes}
    reference_metadata={"crf_reference_length":int(round(reference_stats["median"])),"source":"unique_train_utterance_native_lengths","n_unique_train_utterances":len(train_unique),"median":reference_stats["median"],"mean":reference_stats["mean"],"p05":reference_stats["p05"],"p95":reference_stats["p95"],"split":"train"}
    (output/"metadata/source_hashes.json").write_text(json.dumps(hashes, indent=2)+"\n", encoding="utf8")
    (output/"metadata/crf_reference_length.json").write_text(json.dumps(reference_metadata, indent=2)+"\n", encoding="utf8")
    (output/"metadata/full_alignment_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")
    (output/"metadata/schema.json").write_text(json.dumps({"manifest": "K4 rows with full native feature references only", "gt_npz": ["reference_u_values", "reference_u_offsets", "reference_valid_values", "reference_valid_offsets", "q_values", "q_offsets", "pair_valid_values", "pair_valid_offsets", "valid_interval", "pair_order"]}, indent=2)+"\n", encoding="utf8")
    return summary


class K4FullUtteranceAlignmentDataset(Dataset):
    """Read one materialized full K4 record; never sample, crop, or create K."""
    def __init__(self, root: str | Path, feature_root: str | Path, split: str):
        self.root, self.feature_root, self.split = Path(root).resolve(), Path(feature_root).resolve(), split
        self.rows = read_jsonl(self.root/f"manifests/{split}_full_alignment.jsonl")
        self._feature_cache: dict[str, torch.Tensor] | None = None
        self._gt_cache: dict[str, tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]] | None = None
        if not self.rows: raise ValueError("empty full alignment manifest")
        for row in self.rows:
            if row.get("feature_dim") != 39 or not row.get("full_utterance") or any(len(row[key]) != 4 for key in ("sequence_ids", "speaker_ids", "feature_paths", "native_lengths")): raise ValueError("invalid full K4 row")
            if len(set(row["sequence_ids"])) != 4 or len(set(row["speaker_ids"])) != 4: raise ValueError("K4 needs four unique speakers/sequences")
            if "source_feature_root" in row or any(key.startswith("crop_") for key in row): raise ValueError("formal manifests cannot contain crop/source-root fields")

    def _read_feature(self, path: str, native: int) -> torch.Tensor:
        with np.load(self.feature_root/path) as cache: value = cache["mfcc39"].astype(np.float32, copy=True)
        assert_mfcc39(value)
        if len(value) != native: raise ValueError("full feature length mismatch")
        return torch.from_numpy(value)

    def _read_gt(self, relative_path: str) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        with np.load(self.root/relative_path) as gt:
            q = [torch.from_numpy(gt["q_values"][gt["q_offsets"][p]:gt["q_offsets"][p+1]].astype(np.float32,copy=True)) for p in range(12)]
            pair_valid = [torch.from_numpy(gt["pair_valid_values"][gt["pair_valid_offsets"][p]:gt["pair_valid_offsets"][p+1]].astype(bool,copy=True)) for p in range(12)]
            refs = [torch.from_numpy(gt["reference_u_values"][gt["reference_u_offsets"][i]:gt["reference_u_offsets"][i+1]].astype(np.float32,copy=True)) for i in range(4)]
            ref_valid = [torch.from_numpy(gt["reference_valid_values"][gt["reference_valid_offsets"][i]:gt["reference_valid_offsets"][i+1]].astype(bool,copy=True)) for i in range(4)]
            interval = torch.from_numpy(gt["valid_interval"].astype(np.int64,copy=True))
        return q, pair_valid, refs, ref_valid, interval

    def preload_to_memory(self) -> dict[str, int | str]:
        if self._feature_cache is not None and self._gt_cache is not None: return self.cache_summary()
        feature_specs = {path: int(length) for row in self.rows for path, length in zip(row["feature_paths"], row["native_lengths"], strict=True)}
        self._feature_cache = {path: self._read_feature(path, length) for path, length in feature_specs.items()}
        self._gt_cache = {path: self._read_gt(path) for path in sorted({str(row["gt_path"]) for row in self.rows})}
        return self.cache_summary()

    def cache_summary(self) -> dict[str, int | str]:
        if self._feature_cache is None or self._gt_cache is None: return {"mode":"disk","feature_files":0,"gt_files":0,"tensor_bytes":0}
        tensors=list(self._feature_cache.values())
        for values in self._gt_cache.values():
            for collection in values[:4]: tensors.extend(collection)
            tensors.append(values[4])
        return {"mode":"ram","feature_files":len(self._feature_cache),"gt_files":len(self._gt_cache),"tensor_bytes":sum(value.numel()*value.element_size() for value in tensors)}
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, index: int) -> dict[str, Any]:
        schedule_padding=index<0; row = self.rows[-index-1] if schedule_padding else self.rows[index]
        if self._feature_cache is None or self._gt_cache is None:
            features = [self._read_feature(path, native) for path, native in zip(row["feature_paths"], row["native_lengths"], strict=True)]
            q, pair_valid, refs, ref_valid, interval = self._read_gt(str(row["gt_path"]))
        else:
            features = [self._feature_cache[path] for path in row["feature_paths"]]
            q, pair_valid, refs, ref_valid, interval = self._gt_cache[str(row["gt_path"])]
        return {"features": features, "lengths": torch.tensor(row["native_lengths"],dtype=torch.long), "q_pairs": q, "pair_valid_pairs": pair_valid, "reference_u": refs, "reference_valid": ref_valid, "valid_interval": interval, "pair_mask": ~torch.eye(4,dtype=torch.bool), "sample_weight": torch.tensor(row["sample_weight"],dtype=torch.float32), "schedule_weight":torch.tensor(0.0 if schedule_padding else 1.0), **{key:row[key] for key in ("k4_group_id","parent_prompt_id","speaker_ids")}, "utterance_ids": row["sequence_ids"]}


def collate_k4_full_utterance(batch: list[dict[str, Any]], pad_multiple: int = 16) -> dict[str, Any]:
    """The only rectangularization: batch-local time padding, never K or T edits."""
    if not batch or any(len(item["features"]) != 4 for item in batch): raise ValueError("requires materialized K4")
    B = len(batch); L = math.ceil(max(int(item["lengths"].max()) for item in batch)/pad_multiple)*pad_multiple
    x, valid = torch.zeros(B,4,L,39), torch.zeros(B,4,L,dtype=torch.bool)
    lengths, q, pair_valid = torch.empty(B,4,dtype=torch.long), torch.full((B,4,4,L),-1.0), torch.zeros(B,4,4,L,dtype=torch.bool)
    refs, ref_valid, interval = torch.full((B,4,L),-1.0), torch.zeros(B,4,L,dtype=torch.bool), torch.full((B,4,4,2),-1,dtype=torch.long)
    for b,item in enumerate(batch):
        lengths[b], interval[b] = item["lengths"], item["valid_interval"]
        for i,feature in enumerate(item["features"]):
            x[b,i,:len(feature)], valid[b,i,:len(feature)] = feature, True; refs[b,i,:len(feature)] = item["reference_u"][i]; ref_valid[b,i,:len(feature)] = item["reference_valid"][i]
        for p,(i,j) in enumerate(PAIR_ORDER): q[b,i,j,:len(item["q_pairs"][p])] = item["q_pairs"][p]; pair_valid[b,i,j,:len(item["pair_valid_pairs"][p])] = item["pair_valid_pairs"][p]
    if not torch.equal(valid.sum(-1), lengths) or (pair_valid & ~valid[:,:,None,:]).any(): raise ValueError("full-utterance padding/GT contract broken")
    # `expand` alone creates a stride-0 batch dimension.  That representation is
    # valid for read-only tensor operations, but the DataLoader pin-memory worker
    # cannot pin it when B > 1.  Materialize the small [B, 4, 4] mask so batching
    # never changes the training semantics or the memory-pinning contract.
    pair_mask = (~torch.eye(4, dtype=torch.bool)).unsqueeze(0).expand(B, -1, -1).clone()
    group_common_valid = common_k4_content_valid(refs, ref_valid)
    # Training retains the original formal pair supervision unchanged.  The
    # K4-common refinement is evaluation-only, so it cannot alter the D34-
    # matched CRF or Group-PSD objective.
    evaluation_pair_valid = common_k4_pair_valid(q, pair_valid, group_common_valid)
    return {"x":x,"sequence_valid_mask":valid,"lengths":lengths,"q_of_p":q,"pair_valid":pair_valid,"evaluation_pair_valid":evaluation_pair_valid,"reference_u":refs,"reference_valid":ref_valid,"group_common_valid":group_common_valid,"valid_interval":interval,"pair_mask":pair_mask,"sample_weight":torch.stack([item["sample_weight"] for item in batch]),"schedule_weight":torch.stack([item["schedule_weight"] for item in batch]),**{key:[item[key] for item in batch] for key in ("k4_group_id","parent_prompt_id","speaker_ids","utterance_ids")}}


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Deterministic full-manifest batching; it has no group-membership logic."""
    def __init__(self, lengths: list[int], batch_size: int, seed: int = 0): self.lengths,self.batch_size,self.seed,self.epoch,self.cursor = lengths,batch_size,seed,0,0
    def set_epoch(self, epoch: int) -> None:
        epoch=int(epoch)
        if epoch != self.epoch: self.epoch,self.cursor=epoch,0
    def state_dict(self) -> dict[str,int]: return {"epoch":self.epoch,"next_batch_cursor":self.cursor}
    def load_state_dict(self,state: dict[str,int]) -> None: self.epoch,self.cursor=int(state["epoch"]),int(state.get("next_batch_cursor",0))
    def _batches(self) -> list[list[int]]:
        import random
        order=sorted(range(len(self.lengths)),key=self.lengths.__getitem__); batches=[order[i:i+self.batch_size] for i in range(0,len(order),self.batch_size)]; random.Random(self.seed+self.epoch).shuffle(batches); return batches
    def __iter__(self):
        batches=self._batches()
        while self.cursor < len(batches):
            item=batches[self.cursor]; self.cursor+=1; yield item
    def __len__(self) -> int: return math.ceil(len(self.lengths)/self.batch_size)


class DistributedLengthBucketBatchSampler(LengthBucketBatchSampler):
    """Shard one global deterministic length-bucket schedule across ranks."""
    def __init__(self,lengths:list[int],batch_size:int,world_size:int,rank:int,seed:int=0,pad_to_equal_steps:bool=True):
        super().__init__(lengths,batch_size,seed)
        if not 0<=rank<world_size:raise ValueError('invalid DDP rank')
        self.world_size,self.rank,self.pad_to_equal_steps=int(world_size),int(rank),bool(pad_to_equal_steps)
    def _batches(self) -> list[list[int]]:
        batches=super()._batches()
        # Synthetic negative indices are materialized by the Dataset but carry
        # schedule_weight=0, so DDP ranks take equal steps without duplicated
        # training gradients or an altered parent-weighted objective.
        synthetic_source=batches[-1]
        while self.pad_to_equal_steps and len(batches)%self.world_size:
            batches.append([-(index+1) for index in synthetic_source])
        return batches[self.rank::self.world_size]
    def __len__(self) -> int:
        total=len(super()._batches())
        if self.pad_to_equal_steps:return math.ceil(total/self.world_size)
        return len(range(self.rank,total,self.world_size))


def validate_full_alignment_root(root: str | Path) -> dict[str,int]:
    answer={}
    for split,count in EXPECTED_COUNTS.items():
        rows=read_jsonl(Path(root)/f"manifests/{split}_full_alignment.jsonl")
        if len(rows)!=count: raise AssertionError(f"{split}: expected {count}, received {len(rows)}")
        if any(not row.get("full_utterance") or "source_feature_root" in row or any(key.startswith("crop_") for key in row) for row in rows): raise AssertionError("crop provenance found in full manifest")
        answer[split]=count
    answer["total"]=sum(answer.values()); return answer
