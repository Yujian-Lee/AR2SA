import argparse
import os
import re
import sys
from collections import OrderedDict

import torch


def _parse_ensemble_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ensemble_ckpts", nargs="+", required=True, help="Checkpoints to ensemble at probability-map level.")
    parser.add_argument("--ensemble_epoch", type=int, default=None, help="Epoch index passed to model.forward; default uses args.mask_warmup_epochs so saved masks are read.")
    parser.add_argument("--ensemble_save_mask_dir", type=str, default="", help="Optional directory to save ensembled predicted masks.")
    parser.add_argument("--ensemble_max_batches", type=int, default=0, help="Debug only: evaluate first N batches.")
    ens_args, remaining = parser.parse_known_args(argv)
    sys.argv = [sys.argv[0]] + remaining
    return ens_args


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
    print(f"[ensemble] loaded {ckpt_path}")
    print(f"[ensemble] missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model


def _as_logits_tensor(logits):
    if isinstance(logits, (list, tuple)):
        logits = torch.stack(list(logits), dim=0)
    return logits.detach().squeeze().cpu().float()


def _natural_ckpt_label(path):
    name = os.path.basename(path)
    match = re.search(r"epoch_(\d+).*?test([0-9.]+)", name)
    if match:
        return f"epoch={match.group(1)}, test={match.group(2)}"
    return name


def main():
    ens_args = _parse_ensemble_args(sys.argv[1:])

    # Import config after removing ensemble-only args from sys.argv.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from config import args
    from dataset import get_val_dataloader
    from scripts.save_mask import save_batch_mask
    from utils import pyutils, utility

    ckpts = [os.path.abspath(p) for p in ens_args.ensemble_ckpts]
    for path in ckpts:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    eval_epoch = ens_args.ensemble_epoch
    if eval_epoch is None:
        eval_epoch = int(getattr(args, "mask_warmup_epochs", 1))
    print(f"[ensemble] eval_epoch={eval_epoch}")
    print("[ensemble] checkpoints:")
    for path in ckpts:
        print("  ", _natural_ckpt_label(path), path)

    test_loader = get_val_dataloader(args, "test", filterlist=None)
    logit_sums = OrderedDict()
    targets = OrderedDict()

    for ckpt_idx, ckpt_path in enumerate(ckpts):
        model = _load_model(ckpt_path, args)
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                if ens_args.ensemble_max_batches > 0 and batch_idx >= ens_args.ensemble_max_batches:
                    break
                _, logits = model(batch_data, eval_epoch)
                logits = _as_logits_tensor(logits)
                vid = str(batch_data["vid"])
                if vid not in logit_sums:
                    logit_sums[vid] = logits
                    targets[vid] = batch_data["mask_recs"].squeeze().cpu()
                else:
                    logit_sums[vid] += logits
                print(f"[ensemble] ckpt {ckpt_idx + 1}/{len(ckpts)} batch {batch_idx}", end="\r")
        print()
        del model
        torch.cuda.empty_cache()

    avg_meter_miou = pyutils.AverageMeter("miou")
    avg_meter_f = pyutils.AverageMeter("F_score")
    if ens_args.ensemble_save_mask_dir:
        os.makedirs(ens_args.ensemble_save_mask_dir, exist_ok=True)

    for vid, logit_sum in logit_sums.items():
        logits = logit_sum / float(len(ckpts))
        target = targets[vid]
        miou = utility.mask_iou(logits, target)
        f_score = utility.Eval_Fmeasure(logits, target, "./logger", device=args.device)
        avg_meter_miou.add({"miou": miou.item()})
        avg_meter_f.add({"F_score": f_score})
        if ens_args.ensemble_save_mask_dir:
            save_batch_mask(logits, os.path.join(ens_args.ensemble_save_mask_dir, vid))

    res = {
        "miou": round(avg_meter_miou.pop("miou"), 6),
        "fscore": round(avg_meter_f.pop("F_score"), 6),
    }
    print(res)


if __name__ == "__main__":
    main()