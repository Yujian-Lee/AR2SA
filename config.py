import os
import argparse

parser = argparse.ArgumentParser()
# Path setting
parser.add_argument(
    "--model_dir",
    type=str,
    default='.models',
    help="The path to the pretrained Mask2former to use for mask generation cached locally.",
)
parser.add_argument("--data_path", 
                    type=str,
                    default="/data", 
                    help="The path to the datasets.")
parser.add_argument("--feature_path", 
                    type=str,
                    default="/data", 
                    help="The path to the processed audio feature.")
parser.add_argument(
    "--audio_cache_dir",
    type=str,
    default="./data/audio_vggish_cache",
    help="The directory of precomputed VGGish audio embeddings.",
)

#Training setting
# parser.add_argument("--gpu_id", type=str, default="1", help="The GPU device to run generation on.")
parser.add_argument("--bs",  type=int, default=1, help="batch_size for training")
parser.add_argument("--num_workers",  type=int, default=0, help="the number of workers for training")
parser.add_argument("--lr", type=float, default=1e-3, help='lr to fine tuning adapters.')
parser.add_argument("--weight_dec", type=float, default=0.05, help='weight decay to fine tuning adapters.')
parser.add_argument("--epochs", type=int, default=32, help='epochs to fine tuning adapters.')
parser.add_argument("--device", type=str, default="cuda:1", help="The device to run generation on.")
parser.add_argument("--log_dir", type=str, default="./log_NEW_2025_3_30", help="The path to save checkpoint and mask.")
parser.add_argument("--task", type=str, default="v1m", help="subtask")
parser.add_argument("--mask_path", 
                    type=str,
                    default="/data/avs_mask/", 
                    help="The path to the first stage results of binary mask.")


parser.add_argument(
    "--ckpt_dir",
    type=str,
    default='./pth_avs/_20260315-192007/f_miou_best.pth',
    help="The path to the checkpoint to for testing.",
)
parser.add_argument("--save_mask", default=False, action='store_true', help="whether save the test set mask.")
parser.add_argument("--train_mask_dir", type=str, default="./v1mtrainMASK1000", help="training pre-mask cache directory.")
parser.add_argument("--test_mask_dir", type=str, default="./v1mtestMASK1000", help="test pre-mask cache directory.")
parser.add_argument("--mask_warmup_epochs", type=int, default=1, help="epochs that use only opticalNObg before reading saved pre-masks.")
parser.add_argument("--warmup_best_init_masks", default=False, action="store_true", help="during warmup, keep masks from the best test-mIoU epoch as the initial pre-mask cache.")
parser.add_argument("--update_masks_after_warmup", default=False, action="store_true", help="after warmup, overwrite train/test pre-mask caches every epoch.")
parser.add_argument("--init_masks_from_resume", default=False, action="store_true", help="after loading resume_ckpt, run one no-grad pass to initialize train/test pre-mask caches before training.")
parser.add_argument("--init_mask_epoch", type=int, default=0, help="epoch index passed to the model when initializing masks from resume_ckpt; keep it below mask_warmup_epochs to use opticalNObg.")
parser.add_argument("--stage2_lr", type=float, default=1e-4, help="learning rate after mask warmup.")
parser.add_argument("--stage2_epochs", type=int, default=5, help="number of epochs using stage2_lr after warmup.")
parser.add_argument("--stage3_lr", type=float, default=5e-5, help="learning rate after stage2.")
parser.add_argument(
    "--split_mode",
    type=str,
    default="official",
    choices=["official", "ratio_8_1_1"],
    help="official uses metadata train/val/test; ratio_8_1_1 keeps test fixed and builds a deterministic train/val split from non-test data.",
)
parser.add_argument("--split_seed", type=int, default=219, help="seed for ratio_8_1_1 split.")
parser.add_argument("--val_ratio", type=float, default=0.1, help="validation ratio for ratio_8_1_1 split.")
parser.add_argument("--test_ratio", type=float, default=0.1, help="reported target test ratio for ratio_8_1_1 split.")
parser.add_argument(
    "--eval_fixed_test",
    default=False,
    action="store_true",
    help="evaluate the fixed test split once after training with the best checkpoint.",
)
parser.add_argument(
    "--eval_test_each_epoch",
    default=False,
    action="store_true",
    help="monitor the fixed test split every epoch without using it for checkpoint selection.",
)
parser.add_argument(
    "--sfd_memory",
    default=False,
    action="store_true",
    help="use label-free SFD memory updates for train/val/test masks.",
)
parser.add_argument(
    "--sfd_buffer_dir",
    type=str,
    default="./v1mSFD_euclidean_buffer",
    help="temporary buffer root for SFD memory updates.",
)
parser.add_argument(
    "--save_epoch_pth",
    default=False,
    action="store_true",
    help="save a checkpoint every epoch using val/test mIoU in the file name.",
)
parser.add_argument(
    "--resume_ckpt",
    type=str,
    default="",
    help="model checkpoint path to continue training from.",
)
parser.add_argument(
    "--resume_epoch",
    type=int,
    default=-1,
    help="epoch index to start from after loading resume_ckpt; default infers from epoch_XXX in the checkpoint file name.",
)
parser.add_argument(
    "--resume_best_miou",
    type=float,
    default=-1.0,
    help="best mIoU value to keep when resuming; prevents lower resumed epochs from overwriting the previous best.",
)

args = parser.parse_args()

##os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id



