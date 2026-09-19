import os
import cv2
import torch
import random
import torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn import init
from utils.utility import get_loss
from collections import OrderedDict
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
try:
    from torchvision import transforms
except Exception as exc:
    transforms = None
    print(f'[optional import] torchvision.transforms skipped: {exc}')
try:
    import torchvision.transforms.functional as TF
except Exception as exc:
    TF = None
    print(f'[optional import] torchvision.transforms.functional skipped: {exc}')

class AdaptiveAVS(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_processor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-base-ade-semantic",
                                                                         cache_dir=args.model_dir,
                                                                         local_files_only=True)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-ade-semantic",
                                                                         cache_dir=args.model_dir,
                                                                         local_files_only=True,
                                                                         ignore_mismatched_sizes=True).cuda()

        self.audio_proj_vgg = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.learnmask = nn.Parameter(
            torch.full((5, 384, 384),1.0)
        )

    def forward(self, batch_data, idx_ep):
        len_img = len(batch_data["mask_recs"])
        image_sizes = batch_data["image_size"]
        image_sizes = len_img * image_sizes
        img_input = {}
        audio_emb_vgg_ori = batch_data["feat_aud"].cuda().view(1, -1, 128)
        audio_emb = self.audio_proj_vgg(audio_emb_vgg_ori)

        pre_mask = []
        warmup_epochs = int(getattr(self.args, "mask_warmup_epochs",0))
        use_saved_mask = idx_ep == warmup_epochs
        if not use_saved_mask:
            opticalNObg = (batch_data['opticalNObg']!=0).float().cuda() * self.learnmask
        else:
            opticalNObg = (batch_data['opticalNObg']!=0).float().cuda()
            for _idx in range(5):
                if self.training:
                    mask_root = getattr(self.args, "train_mask_dir", "./v1mtrainMASK1000")
                else:
                    mask_root = getattr(self.args, "test_mask_dir", "./v1mtestMASK1000")
                premaskPATH = os.path.join(mask_root, str(batch_data['vid']), f"{_idx}.png")
                pre_img = cv2.imread(premaskPATH)
                if pre_img is None:
                    pre = batch_data['opticalNObg'][_idx].cpu().numpy().astype("uint8") * 255
                else:
                    pre = cv2.cvtColor(cv2.resize(pre_img, (384, 384)), cv2.COLOR_BGR2GRAY)
                pre = torch.as_tensor((pre > 0), dtype=torch.int)
                pre_mask.append(pre)
            pre_mask = (torch.stack(pre_mask)!=0).float().cuda() * opticalNObg * self.learnmask

        if not use_saved_mask:
            img_input["audio_boost_mask"] = opticalNObg
        else:
            img_input["audio_boost_mask"] = pre_mask
        

        img_input['prompt_features_projected'] = audio_emb
        img_input['pixel_mask'] = batch_data['pixel_mask'].squeeze().view(-1, 384, 384).cuda()
        img_input["anchor_audio_boost_mask"] = batch_data["mask_recs"].cuda()
        img_input["mask_labels"] = [i.cuda() for i in batch_data["mask_labels"]]
        img_input["class_labels"] = [i.cuda() for i in batch_data["class_labels"]]
        img_input['pixel_values'] = batch_data['pixel_values'].squeeze().view(-1, 3, 384, 384).cuda()
        img_input['batch_id']=batch_data['vid']

        
        avs_outputs, loss_audio_text = self.model(**img_input)

        with torch.no_grad():
            logits = self.image_processor.post_process_binary_segmentation(avs_outputs, target_sizes=image_sizes)
        loss_frame = get_loss(avs_outputs.loss, batch_data["mask_recs"], batch_data["task"])
        loss_frame = loss_frame + loss_audio_text
        return loss_frame, logits


