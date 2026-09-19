import os
import torch
import random
import numpy as np
from config import args
from dataset import get_val_dataloader
from avs_model import AdaptiveAVS
from scripts import test_avs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == '__main__':
    

    set_seed(219)
    test_loader = get_val_dataloader(args,"test",filterlist=None)
    # if args.save_mask:
    #     os.makedirs("save_masks", exist_ok=True)
    # Model
    model = AdaptiveAVS(args).cuda()
    print(args.ckpt_dir)
    checkpoint = torch.load(args.ckpt_dir, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("[test] loading model_state_dict from epoch checkpoint")
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print("missing:", len(missing), missing[:20])
    print("unexpected:", len(unexpected), unexpected[:20])
    model.eval()
    res,_,_ = test_avs(model, test_loader,args,1,0.1)
    print(res)
