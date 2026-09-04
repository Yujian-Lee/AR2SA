import argparse
import logging
import os
import warnings
from pathlib import Path

import pandas as pd
import torch
from towhee import pipe, ops


warnings.filterwarnings("ignore", message=".*Converting float dtype.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("vggish").setLevel(logging.ERROR)
logging.getLogger("vggish").propagate = False
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def build_vggish_pipeline():
    return (
        pipe.input("path")
            .map("path", "frame", ops.audio_decode.ffmpeg())
            .map("frame", "vecs", ops.audio_embedding.vggish())
            .output("vecs")
    )


def run_silenced(fn):
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        return fn()
    finally:
        logging.disable(previous_disable_level)


def compute_v1m_embedding(vggish_pipeline, audio_segments_dir):
    embeddings = []
    for idx in range(5):
        wav_path = audio_segments_dir / f"{idx + 1}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing audio segment: {wav_path}")
        value = run_silenced(lambda: vggish_pipeline(str(wav_path)).get()[0])
        embeddings.append(torch.as_tensor(value))
    embedding = torch.stack(embeddings).squeeze(1).float().cpu()
    if embedding.shape != (5, 128):
        raise RuntimeError(f"Unexpected embedding shape {tuple(embedding.shape)} for {audio_segments_dir}")
    return embedding


def save_tensor_atomic(tensor, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(tensor, tmp_path)
    if out_path.exists():
        out_path.unlink()
    tmp_path.replace(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--metadata", type=str, default="./data/metadata.csv")
    parser.add_argument("--task", type=str, default="v1m")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--audio_cache_dir", type=str, default="./data/audio_vggish_cache")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_items", type=int, default=0)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    rows = metadata[(metadata["label"] == args.task) & (metadata["split"].isin(args.splits))].reset_index(drop=True)
    if args.max_items > 0:
        rows = rows.head(args.max_items)

    print(f"[audio-cache] task={args.task}, splits={args.splits}, total={len(rows)}")
    print(f"[audio-cache] output={Path(args.audio_cache_dir).resolve()}")

    vggish_pipeline = build_vggish_pipeline()
    done = 0
    skipped = 0
    failed = 0

    for row_idx, row in rows.iterrows():
        uid = row["uid"]
        split = row["split"]
        audio_segments_dir = Path(args.data_path) / args.task / split / uid / "audio_segments"
        out_path = Path(args.audio_cache_dir) / args.task / split / f"{uid}.pt"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            embedding = compute_v1m_embedding(vggish_pipeline, audio_segments_dir)
            save_tensor_atomic(embedding, out_path)
            done += 1
        except Exception as exc:
            failed += 1
            print(f"[audio-cache][failed] {split}/{uid}: {exc}")
            continue

        if done % 20 == 0:
            print(f"[audio-cache] done={done}, skipped={skipped}, failed={failed}, last={split}/{uid}")

    print(f"[audio-cache] finished: done={done}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
