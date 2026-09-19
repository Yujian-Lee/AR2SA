import os
import cv2
import json
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from typing import Optional
from towhee import pipe, ops
from torchvision import transforms
from transformers import Mask2FormerImageProcessor
from torch.utils.data import Dataset, ConcatDataset

def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls) # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete

def resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    if not img_is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img

def color_mask_to_label(mask_array, v_pallete):
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    label = np.argmax(semantic_map, axis=-1)
    return label

def load_color_mask_in_PIL_to_Tensor(path, v_pallete, split='train', mode='RGB'):
    color_mask_PIL = Image.open(path).convert(mode)
    color_mask_PIL = resize_img(224, color_mask_PIL, img_is_mask=True)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label) # [H, W]
    return color_label 

def custom_collate(batch):
    # print(batch[0].keys())
    mask_recs = batch[0]['mask_recs']
    image_size = batch[0]['image_size']
    vid = batch[0]['vid']
    feat_aud = batch[0]['feat_aud']
    pixel_values = batch[0]['pixel_values'] 
    pixel_mask = batch[0]['pixel_mask']
    class_labels = batch[0]['class_labels']
    mask_labels = batch[0]['mask_labels']
    task=batch[0]['task']
    optical=batch[0]['optical_pixel_values']
    optical2=batch[0]['optical_pixel_values2']
    audio_inference_text=batch[0]['audio_inference_text']
    visual_inference_text=batch[0]['visual_inference_text']
    unified_text=batch[0]['unified_text']
    NEW_text=batch[0]['NEW_text']
    visualClip = batch[0]['visualClip']
    visualClip_nobg=batch[0]['visualClip_nobg']
    GTtext = batch[0]['GTtext']
    clap_feat_aud=batch[0]['clap_feat_aud']
    onlyAUD=batch[0]['onlyAUD']
    opticalNObg=batch[0]["opticalNObg"]
    res = {
        "mask_recs": mask_recs,
        "image_size": image_size,
        "vid": vid,
        # 替换
        "feat_aud": feat_aud,
        # 替换
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": class_labels,
        "mask_labels": mask_labels,
        "task":task,
        "optical":optical,
        "optical2":optical2,
        "audio_inference_text":audio_inference_text,
        "visual_inference_text":visual_inference_text,
        "unified_text":unified_text,
        "NEW_text":NEW_text,
        "visualClip":visualClip,
        "visualClip_nobg":visualClip_nobg,
        "GTtext":GTtext,
        "clap_feat_aud":clap_feat_aud,
        "onlyAUD":onlyAUD,
        "opticalNObg":opticalNObg,
    }
    return res
def get_train_dataloader(args,filterlist):
    if args.task=="avss":
        return get_avss_train_dataloader(args)
    else:
        return get_avs_train_dataloader(args,filterlist)


def get_avs_train_dataloader(args, filterlist):
    # 改
    train_dataset = AVSBench(args,'train', args.task, filterlist)
    print("++++++++++++++++++++++",len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers,collate_fn=custom_collate)
    return train_loader
def get_val_dataloader(args,split,filterlist):
    if args.task=="avss":
        return get_avss_val_dataloader(args,split)
    else:
        return get_avs_val_dataloader(args,split,filterlist)

def get_avs_val_dataloader(args,split,filterlist):
    # 改
    train_dataset = AVSBench(args,split, args.task,filterlist)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers,collate_fn=custom_collate)
    return train_loader
    
def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class AVSBench(Dataset):
    def __init__(self, args, split, ver, filterlist):
        self.ver = ver
        self.data_base_path = args.data_path
        self.data_path = os.path.join(self.data_base_path,ver,split)
        self.split = split

        meta_path = './data/metadata.csv'
        metadata = pd.read_csv(meta_path, header=0)
        sub_data = metadata[metadata['label'] == ver] 
        self.metadata = sub_data[sub_data['split'] == split] 
        if filterlist is not None:
            self.metadata = self.metadata[~self.metadata["uid"].isin(filterlist)].reset_index(drop=True)
        self.frame_num = 10 if ver == 'v2' else 5  
        self.pallete = get_v2_pallete('./data/label2idx.json')
        if self.ver == 'v1s' and self.split=='train':
            self.frame_num = 5

        
        self.audio_vggish_pipeline = (   # pipeline building - no grad
            pipe.input('path')
                .map('path', 'frame', ops.audio_decode.ffmpeg())
                .map('frame', 'vecs', ops.audio_embedding.vggish())
                .output('vecs')
        )
        
        self.img_process = Mask2FormerImageProcessor.from_pretrained(".models",cache_dir=args.model_dir, local_files_only=True)

    def __len__(self):
        return len(self.metadata)
        
    
    def get_audio_emb(self, wav_path):
        """ wav string path. """ 
        # warnings.filterwarnings("ignore", category=UserWarning)
        # import warnings
        EMB = []
        for i in range(5):
            name = str(i+1)+'.wav'
            jrwav_path = os.path.join(wav_path, name)
            emb = torch.tensor(self.audio_vggish_pipeline(jrwav_path).get()[0])
            # print(emb.shape) [1,1,128]
            EMB.append(emb)
        emb = torch.stack(EMB).squeeze(1)
        return emb
    # def get_audio_emb(self, wav_path):
    #     """ wav string path. """ 
    #     emb = torch.tensor(self.audio_vggish_pipeline(wav_path).get()[0])
    #     return emb


    def get_visual_emb(self, image):
        """
        image: PIL.Image 或者 tensor (C,H,W)，如果是 PIL 就直接 preprocess
        """
        image = Image.fromarray(image.transpose(1,2,0).astype('uint8'))
        image = self.preprocess(image)
        image = torch.stack(image).cuda()

        image_emb = self.clip_model.encode_image(image)
        image_emb = F.normalize(image_emb, p=2, dim=-1)
        return image_emb

    
    def __getitem__(self, idx):
        df_one_video = self.metadata.iloc[idx]
        vid = df_one_video['uid']
        gttext = df_one_video['a_obj']

        clap_rec_audio = f'./{self.data_path}/{vid}/audio_segments'
        feat_aud = self.get_audio_emb(clap_rec_audio)

        if feat_aud.shape[0] != 5 and self.ver == 'v1s':
            while feat_aud.shape[0] != 5:
                feat_aud = torch.concat([feat_aud, feat_aud[-1].view(1, -1)], dim=0)
        
        image_list, label_list, nobg_white_list, onlyAUD = [],[],[],[]
        for _idx in range(self.frame_num):  
            _idx=0 if self.split == 'train' and self.ver == 'v1s' else _idx
            path_mask = f'./{self.data_path}/{vid}/labels_rgb/{_idx}.png'
            mask_cv2 = cv2.imread(path_mask)
            mask_cv2 = cv2.resize(mask_cv2, (384, 384))
            mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
            ground_truth_mask = torch.as_tensor((mask_cv2 > 0) , dtype=torch.float32)
            label_list.append(ground_truth_mask)
            
            imagepath = f'./{self.data_path}/{vid}/frames/{_idx}.jpg'
            normalimage = cv2.cvtColor(cv2.resize(cv2.imread(imagepath),(384,384)),cv2.COLOR_BGR2RGB).transpose(2,0,1)
            image_list.append(normalimage)
            path_nobg_optical = f'./{self.data_path}/{vid}/opticalNObg/{_idx}.png'
            opticalNObg_mask = cv2.imread(path_nobg_optical)
            opticalNObg_mask = cv2.resize(opticalNObg_mask, (384, 384))
            opticalNObg_mask = cv2.cvtColor(opticalNObg_mask, cv2.COLOR_BGR2GRAY)
            opticalNObg_mask = torch.as_tensor((opticalNObg_mask > 0), dtype=torch.float32)
            nobg_white_list.append(opticalNObg_mask)
            

        image_inputs = self.img_process.preprocess(image_list, label_list,return_tensors="pt")

   
        image_inputs['pixel_values'] = image_inputs['pixel_values']
        image_inputs["mask_recs"] = torch.stack(label_list)
        image_inputs["image_size"] = [(384,384)]
        image_inputs["vid"] = vid
        image_inputs["opticalNObg"]= torch.stack(nobg_white_list)
        image_inputs["task"]=self.ver

        
        return image_inputs



