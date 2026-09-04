import torch
from utils import utility,pyutils
import cv2,os
import shutil
from pathlib import Path
from scripts.save_mask import save_batch_mask
from scripts.sfd_memory import empty_sfd_stats, add_sfd_stats, format_sfd_stats, update_sfd_memory_batch
N_CLASSES = 2
avg_meter_miou = pyutils.AverageMeter('miou')
avg_meter_F = pyutils.AverageMeter('F_score')


def _copy_saved_masks(src_dir, dst_dir):
    src = Path(src_dir).resolve()
    dst = Path(dst_dir).resolve()
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for file in src.rglob("*.png"):
        rel = file.relative_to(src)
        out_file = dst / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, out_file)


def _safe_remove_tmp_dir(tmp_dir, expected_parent):
    tmp = Path(os.path.abspath(os.fspath(tmp_dir)))
    parent = Path(os.path.abspath(os.fspath(expected_parent)))
    tmp_parent = os.path.normcase(str(tmp.parent))
    expected = os.path.normcase(str(parent))
    if tmp_parent != expected or "__tmp" not in tmp.name:
        raise RuntimeError(f"Refusing to remove unexpected temp directory: {tmp}")
    if tmp.exists():
        shutil.rmtree(tmp)


def test_avs(model, test_loader, args, alpha, max_miou, sfd_split=None, sfd_active_dir=None, sfd_buffer_dir=None):
    logits_list = []
    id_list = []
    # args.save_mask = True
    # save_base_path=os.path.join("v1mtestMASK1000")
    # os.makedirs(save_base_path, exist_ok=True)
    save_base_path = os.environ.get("AR2SA_SAVE_TEST_MASK_DIR")
    sync_mask_policy = os.environ.get("AR2SA_SYNC_MASK_POLICY", "0") in {"1", "true", "True", "on", "ON"}
    if sync_mask_policy:
        save_base_path = None
    save_policy = os.environ.get("AR2SA_SAVE_TEST_MASK_POLICY", "every_epoch").lower()
    save_min_miou = float(os.environ.get("AR2SA_SAVE_TEST_MASK_MIN_MIOU", "0.0"))
    tmp_save_path = None
    active_save_path = None
    if save_base_path and save_policy not in {"0", "false", "off", "none", "disabled"}:
        if save_policy == "best":
            base = Path(save_base_path)
            tmp_save_path = str(base.parent / f"{base.name}__tmp_epoch_{alpha}")
            _safe_remove_tmp_dir(tmp_save_path, base.parent)
            active_save_path = tmp_save_path
        else:
            active_save_path = save_base_path
        os.makedirs(active_save_path, exist_ok=True)
    model.eval()
    id_list = []
    logit = []
    sfd_stats = empty_sfd_stats()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            max_batches = int(os.environ.get("AR2SA_TEST_MAX_BATCHES", "0"))
            if max_batches > 0 and batch_idx == 0:
                print(f"[partial eval] AR2SA_TEST_MAX_BATCHES={max_batches}; this is NOT full test mIoU.")
            if max_batches > 0 and batch_idx >= max_batches:
                break
            _, logits = model(batch_data,alpha)
            print(batch_idx)
            logits = torch.stack(logits, dim=0).squeeze().cpu()  # [5, 720, 1280] = [1*frames, H, W]
            vid_masks_t = batch_data["mask_recs"].squeeze()

            miou = utility.mask_iou(logits, vid_masks_t)
            F_score = utility.Eval_Fmeasure(logits, vid_masks_t, './logger', device=args.device)

            logits_list.append(logits)
            id_list.append(batch_data["vid"])
            if active_save_path:
                save_path = os.path.join(active_save_path, str(batch_data["vid"]))
                save_batch_mask(logits, save_path)

            if sfd_split is not None:
                batch_stats = update_sfd_memory_batch(
                    sfd_split,
                    logits,
                    batch_data["vid"],
                    batch_data["opticalNObg"],
                    sfd_active_dir,
                    sfd_buffer_dir,
                )
                add_sfd_stats(sfd_stats, batch_stats)
            
            avg_meter_miou.add({'miou': miou.item()})
            avg_meter_F.add({'F_score': F_score})
            

    miou = round(avg_meter_miou.pop('miou'),6)
    f_score = round(avg_meter_F.pop('F_score'),6)
    
    res = {
        'miou': miou,
        'fscore': f_score,
    }
    if sfd_split is not None:
        res[f'{sfd_split}_sfd_accepted'] = sfd_stats["accepted"]
        res[f'{sfd_split}_sfd_rejected'] = sfd_stats["rejected"]
        print(format_sfd_stats(sfd_split, sfd_stats))
    if save_base_path and save_policy == "best":
        should_commit = miou > max_miou and miou >= save_min_miou
        if should_commit:
            _copy_saved_masks(tmp_save_path, save_base_path)
            print(f"[test mask] saved best masks to {save_base_path} (miou={miou})")
        else:
            print(f"[test mask] kept previous masks (miou={miou}, best={max_miou}, min={save_min_miou})")
        _safe_remove_tmp_dir(tmp_save_path, Path(save_base_path).parent)
    # if res['miou'] > max_miou:
    #     for i in range(len(id_list)):
    #      os.makedirs(os.path.join(save_base_path,id_list[i]), exist_ok=True)
    #      save_batch_mask(logit[i],os.path.join(save_base_path,id_list[i]))
        

    return res,logits_list,id_list

