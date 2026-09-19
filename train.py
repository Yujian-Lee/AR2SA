import os
import re
import time
import torch
import wandb
import random
import numpy as np
from config import args
from scripts import test_avs
from avs_model import AdaptiveAVS
from scripts.save_mask import save_batch_mask
from dataset import get_train_dataloader,get_val_dataloader

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def get_fineture_model(args,log_dir):
    model = AdaptiveAVS(args).cuda()
    params_dict=torch.load(os.path.join(log_dir,"f_miou_best.pth"))
    new_dict={}
    for i in params_dict.keys():
        if not 'swin_adapt' in i:
            new_dict[i]=params_dict[i]
    model.load_state_dict(new_dict,strict=False)
    return model
def set_params(model, change):
    params_names = [
        'queries_embedder',
        'queries_features',
        'avs_adapt',
        'class_predictor',
        "audio_proj"
    ]

    params=[]
    for name, param in model.named_parameters():
        param.requires_grad = False
        if 'swin_adapt' in name:
            if change:
                param.requires_grad = True  
                params.append(param)
            else:
                torch.nn.init.zeros_(param)
        for _n in params_names:
                if _n in name: 
                    param.requires_grad = True  
                    params.append(param)
    return params        

def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _checkpoint_best_miou(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    for key in ("best_miou", "test_miou", "miou"):
        value = checkpoint.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _infer_resume_epoch(ckpt_path):
    match = re.search(r"epoch_(\d+)", os.path.basename(ckpt_path))
    if match:
        return int(match.group(1)) + 1
    return 1


def load_resume_checkpoint(model, ckpt_path, optimizer=None):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[resume] loaded {ckpt_path}")
    print(f"[resume] missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[resume] missing first:", missing[:10])
    if len(unexpected) > 0:
        print("[resume] unexpected first:", unexpected[:10])
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("[resume] optimizer state loaded")
        except Exception as exc:
            print(f"[resume] optimizer state skipped: {exc}")
    return _checkpoint_best_miou(checkpoint)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def scheduled_lr(args, idx_ep):
    if getattr(args, "warmup_best_init_masks", False):
        warmup_epochs = int(getattr(args, "mask_warmup_epochs", 1))
        if idx_ep < warmup_epochs:
            return args.lr
        stage2_epochs = int(getattr(args, "stage2_epochs", 5))
        if idx_ep < warmup_epochs + stage2_epochs:
            return float(getattr(args, "stage2_lr", 1e-4))
        return float(getattr(args, "stage3_lr", 5e-5))
    return args.lr


def _prepare_logits_for_mask(logits):
    if isinstance(logits, (list, tuple)):
        logits = torch.stack(list(logits), dim=0)
    if torch.is_tensor(logits):
        return logits.detach().squeeze().cpu()
    return torch.as_tensor(logits).squeeze()


def save_logits_masks(mask_root, logits_list, id_list):
    os.makedirs(mask_root, exist_ok=True)
    for logits, vid in zip(logits_list, id_list):
        save_path = os.path.join(mask_root, str(vid))
        os.makedirs(save_path, exist_ok=True)
        save_batch_mask(_prepare_logits_for_mask(logits), save_path)
    print(f"[mask] saved {len(id_list)} videos to {mask_root}")


def collect_logits_for_masks(model, loader, idx_ep, split_name):
    logits_list = []
    id_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            _, logits = model(batch_data, idx_ep)
            logits_list.append(logits)
            id_list.append(batch_data["vid"])
            print(f"[init masks] {split_name} {batch_idx}", end="\r")
    print()
    return logits_list, id_list

if __name__ == '__main__':
    # Fix seed
    #rank=init_dist()
    # warnings.filterwarnings("ignore", message="Converting float dtype from float64 to float64")
    set_seed(219)
    # dir to save checkpoint
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    #if rank==0:
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime('_%Y%m%d-%H%M%S')))
    os.makedirs(log_dir, exist_ok=True)
    wandb.init(
        project="AVSS task",
        config=args,
        name="Adaptive_AVS"
    )
    wandb.require("core")
    
    # Data
    filterlist = None
    testfilterlist = None
    train_loader=get_train_dataloader(args, filterlist)
    val_loader = get_val_dataloader(args,"test",testfilterlist)
    # Model
    model = AdaptiveAVS(args).cuda()
          
    # Optimizer
    params=set_params(model, True)
    optimizer = torch.optim.AdamW(params,lr=args.lr, weight_decay=args.weight_dec, eps=1e-8, betas=(0.9, 0.999))
    message = f'All: {sum(p.numel() for p in model.parameters()) / 1e6}M\n'
    message += f'Train-able: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M\n'
    print(message)
    
    train_losses = []
    miou = []
    fscore = []
    max_miou=0.1

    print(f"Using device {args.device}")    
    for idx_ep in range(args.epochs):
        temp = True
        print("****")
        print(f'[Epoch] {idx_ep}')
        model.train()
        losses = []
       
        if idx_ep==14:
            model=get_fineture_model(args,log_dir)
            params=set_params(model, True)
            optimizer = torch.optim.AdamW(params,lr=1e-4, weight_decay=args.weight_dec, eps=1e-8, betas=(0.9, 0.999))

        model.train()

        logits_list = []
        id_list = []
        for batch_idx, batch_data in enumerate(train_loader):
            # audio_emb,visual_feats,T2Vpos_loss,T2Apos_loss = Bridgemodel(temp, batch_data)
            loss_vid, logits = model(batch_data,idx_ep)
            optimizer.zero_grad()
            loss_vid.backward()
            print(f'loss_{idx_ep}_{batch_idx}: {loss_vid.item()}', end='\r')
            optimizer.step()
            losses.append(loss_vid.item())
            logits_list.append(logits)
            id_list.append(batch_data['vid'])
            
            
        loss = {"total_loss":np.mean(losses),}
        model.eval()
        res,test_logits_list,test_id_list = test_avs(model, val_loader, args, idx_ep, max_miou)

        miou.append(res["miou"])
        fscore.append(res["fscore"])
        if (res["miou"] > max_miou):
            model_save_path = os.path.join(log_dir,"f_miou_best.pth")
            max_miou = res["miou"] 
            torch.save(model.state_dict(), model_save_path)   

        for i in range(len(logits_list)):
            os.makedirs(os.path.join('./v1mtrainMASK1000',id_list[i]), exist_ok=True)
            logits = torch.stack(logits_list[i], dim=0).squeeze().cpu()
            save_batch_mask(logits,os.path.join('./v1mtrainMASK1000',id_list[i]))

                
        wandb.log(res)

        print("mIOU: ", miou)
        print("Fscore: ",fscore)

