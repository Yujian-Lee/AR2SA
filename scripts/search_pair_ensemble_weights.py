import argparse
import os
import sys
from collections import OrderedDict

import torch


def _parse_search_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ckpt_a", required=True, help="First checkpoint path.")
    parser.add_argument("--ckpt_b", required=True, help="Second checkpoint path.")
    parser.add_argument("--weight_step", type=float, default=0.05, help="Grid step for weight of ckpt_a.")
    parser.add_argument("--ensemble_epoch", type=int, default=None, help="Epoch index passed to model.forward; default uses args.mask_warmup_epochs.")
    parser.add_argument("--max_batches", type=int, default=0, help="Debug only: evaluate first N batches.")
    search_args, remaining = parser.parse_known_args(argv)
    sys.argv = [sys.argv[0]] + remaining
    return search_args


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _load_model(ckpt_path, args):
    from avs_model import AdaptiveAVS

    model = AdaptiveAVS(args).cuda()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[pair search] loaded {ckpt_path}")
    print(f"[pair search] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def _as_logits_tensor(logits):
    if isinstance(logits, (list, tuple)):
        logits = torch.stack(list(logits), dim=0)
    return logits.detach().squeeze().cpu().half()


def _weight_grid(step):
    if step <= 0 or step > 1:
        raise ValueError("--weight_step must be in (0, 1].")
    n = int(round(1.0 / step))
    values = [round(i / n, 6) for i in range(n + 1)]
    if values[-1] != 1.0:
        values.append(1.0)
    return values


def main():
    search_args = _parse_search_args(sys.argv[1:])

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from config import args
    from dataset import get_val_dataloader
    from utils import pyutils, utility

    ckpt_a = os.path.abspath(search_args.ckpt_a)
    ckpt_b = os.path.abspath(search_args.ckpt_b)
    if not os.path.exists(ckpt_a):
        raise FileNotFoundError(ckpt_a)
    if not os.path.exists(ckpt_b):
        raise FileNotFoundError(ckpt_b)

    eval_epoch = search_args.ensemble_epoch
    if eval_epoch is None:
        eval_epoch = int(getattr(args, "mask_warmup_epochs", 1))
    weights = _weight_grid(search_args.weight_step)
    print(f"[pair search] eval_epoch={eval_epoch}")
    print(f"[pair search] ckpt_a={ckpt_a}")
    print(f"[pair search] ckpt_b={ckpt_b}")
    print(f"[pair search] weights for ckpt_a={weights}")

    test_loader = get_val_dataloader(args, "test", filterlist=None)

    logits_a = OrderedDict()
    targets = OrderedDict()
    model = _load_model(ckpt_a, args)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if search_args.max_batches > 0 and batch_idx >= search_args.max_batches:
                break
            _, logits = model(batch_data, eval_epoch)
            vid = str(batch_data["vid"])
            logits_a[vid] = _as_logits_tensor(logits)
            targets[vid] = batch_data["mask_recs"].squeeze().cpu().to(torch.uint8)
            print(f"[pair search] cached A batch {batch_idx}", end="\r")
    print()
    del model
    torch.cuda.empty_cache()

    meters = {
        w: (pyutils.AverageMeter("miou"), pyutils.AverageMeter("F_score"))
        for w in weights
    }

    model = _load_model(ckpt_b, args)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if search_args.max_batches > 0 and batch_idx >= search_args.max_batches:
                break
            _, logits = model(batch_data, eval_epoch)
            vid = str(batch_data["vid"])
            b = _as_logits_tensor(logits).float()
            a = logits_a[vid].float()
            target = targets[vid].float()
            for w in weights:
                mixed = a * w + b * (1.0 - w)
                miou = utility.mask_iou(mixed, target)
                f_score = utility.Eval_Fmeasure(mixed, target, "./logger", device=args.device)
                meters[w][0].add({"miou": miou.item()})
                meters[w][1].add({"F_score": f_score})
            print(f"[pair search] evaluated B batch {batch_idx}", end="\r")
    print()
    del model
    torch.cuda.empty_cache()

    rows = []
    for w in weights:
        miou = round(meters[w][0].pop("miou"), 6)
        fscore = round(meters[w][1].pop("F_score"), 6)
        rows.append((fscore, miou, w))

    print("\nweight_a,weight_b,miou,fscore")
    for fscore, miou, w in sorted(rows, key=lambda x: x[2]):
        print(f"{w:.6f},{1.0 - w:.6f},{miou:.6f},{fscore:.6f}")

    print("\nTop by F-score:")
    for fscore, miou, w in sorted(rows, reverse=True)[:10]:
        print(f"weight_a={w:.6f}, weight_b={1.0 - w:.6f}, miou={miou:.6f}, fscore={fscore:.6f}")


if __name__ == "__main__":
    main()