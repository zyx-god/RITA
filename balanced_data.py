import os, re, glob, json, random
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _sorted_pairs(tp_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    """ManiDataset：Tp 与 Gt 按文件顺序一一对应"""
    tp_files = sorted(glob.glob(os.path.join(tp_dir, '*.*')))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.*')))

    if len(tp_files) != len(gt_files):
        print(f"[WARN] Tp({len(tp_files)}) 和 Gt({len(gt_files)}) 数量不一致，将按最短长度配对")

    min_len = min(len(tp_files), len(gt_files))
    pairs = [(tp_files[i], gt_files[i]) for i in range(min_len)]
    return pairs

def _looks_like_mask_ref(x):
    return isinstance(x, str) and ("/" in x or "\\" in x)

def _load_json_lol(json_path: str) -> List[Tuple[str, str]]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    pairs = []
    for it in data:
        if isinstance(it, (list, tuple)) and len(it) >= 2:
            img, m = it[0], it[1]

            # 只保留看起来像路径的 mask 引用
            if _looks_like_mask_ref(m):
                pairs.append((img, m))

    return pairs



def _find_subdir(root, candidates):

    for name in candidates:
        subdir = os.path.join(root, name)
        if os.path.isdir(subdir):
            return subdir

    raise FileNotFoundError(
        f"Cannot find any of {candidates} under {root}"
    )


class UnifiedNextMaskDatasetEpisodic(Dataset):
    def __init__(self,
                 data_path='/mnt/data0/xuekang/workspace/ar_iml/balanced_data.json',
                 image_size=(512, 512),
                 num_classes: int = 11,
                 num_per_dataset: int = 1840,
                 base_seed: int = 42,
                 common_transforms=None):

        self.image_size = image_size
        self.num_classes = num_classes
        self.eos_token_id = num_classes - 2
        self.padding_id = num_classes - 1

        self.num_per_dataset = num_per_dataset
        self.base_seed = base_seed
        self._epoch = 0
        self.common_transforms = common_transforms


        if os.path.isdir(data_path):
            self.settings_list = [
                ["ManiDataset", data_path]
            ]

        elif os.path.isfile(data_path):
            with open(data_path, 'r') as f:
                self.settings_list = json.load(f)

        else:
            raise FileNotFoundError(
                f"config_json_path does not exist: {data_path}"
            )

        self.pools: List[Dict] = []

        for kind, path in self.settings_list:
            if kind == "ManiDataset":
                tp_dir = _find_subdir(path, ["TP", "Tp", "tp"])
                gt_dir = _find_subdir(path, ["GT", "Gt", "gt"])

                pairs = _sorted_pairs(tp_dir, gt_dir)
                self.pools.append({
                    "kind": "mani",
                    "pairs": pairs
                })

            elif kind == "JsonDataset":
                pairs = _load_json_lol(path)
                self.pools.append({
                    "kind": "json",
                    "pairs": pairs
                })

            else:
                self.pools.append({
                    "kind": "unknown",
                    "pairs": []
                })

        # 本轮展开后的记录
        self.records: List[Dict] = []

        # 变换
        self.to_tensor = T.ToTensor()
        self.resize_img = T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC)
        self.resize_mask = T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST)
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # 初始化一轮
        self.resample(epoch=0)

    def set_epoch(self, epoch: int):
        self.resample(epoch)

    def resample(self, epoch: int = None):
        """对每个数据源各采样 num_per_dataset，并展开为 step 记录。"""
        if epoch is None:
            epoch = self._epoch + 1
        self._epoch = epoch

        rng = random.Random(self.base_seed + epoch)

        self.records = []

        # 统计用
        for pool in self.pools:
            kind = pool["kind"]
            pairs = pool["pairs"]
            if not pairs:
                continue

            n = len(pairs)
            if self.num_per_dataset > n:
                chosen = [pairs[rng.randrange(n)] for _ in range(self.num_per_dataset)]
            else:
                idxs = list(range(n))
                rng.shuffle(idxs)
                chosen = [pairs[i] for i in idxs[:self.num_per_dataset]]

            # 展开
            for img_path, m in chosen:
                if kind == "mani":
                    self.records.append({
                        "kind": "mani", "image": img_path, "mask": m,
                        "step": 0, "i": 0, "final_gt": m
                    })
                    self.records.append({
                        "kind": "mani", "image": img_path, "mask": m,
                        "step": 1, "i": 1, "final_gt": m
                    })
                    self.records.append({
                        "kind": "mani", "image": img_path, "mask": m,
                        "step": 2, "i": 2, "final_gt": m
                    })
                elif kind == "json":
                    self.records.append({
                        "kind": "json_pos", "image": img_path, "mask": m,
                        "step": 0, "i": 0, "final_gt": m
                    })
                    self.records.append({
                        "kind": "json_pos", "image": img_path, "mask": m,
                        "step": 1, "i": 1, "final_gt": m
                    })
                    self.records.append({
                        "kind": "json_pos", "image": img_path, "mask": m,
                        "step": 2, "i": 2, "final_gt": m
                    })

                else:
                    continue  # unknown 跳过


    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        kind = rec["kind"]
        step = rec["step"]

        image = Image.open(rec["image"]).convert("RGB")
        if kind == "mani" or kind == "json_pos":
            raw_mask = Image.open(rec["mask"]).convert("L")
        else:
            H, W = self.image_size
            raw_mask =  torch.zeros((H, W), dtype=torch.long)
            
        if self.common_transforms is not None:
            image = np.array(image)  # H, W, C
            raw_mask = np.array(raw_mask)    # H, W
            res_dict = self.common_transforms(image=image, mask=raw_mask)
            image_np = res_dict['image']
            mask_np = res_dict['mask']

            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)

            if mask_np.dtype != np.uint8:
                mask_np = mask_np.astype(np.uint8)
            raw_mask = Image.fromarray(mask_np)

        image = self.resize_img(image)
        image = self.to_tensor(image)
        image = self.norm(image)
        
        raw_mask = self.resize_mask(raw_mask)

        H, W = self.image_size
        padding_id = self.padding_id
        eos_id = self.eos_token_id

        if kind == "mani":
            m = np.array(raw_mask)
            bin_mask = torch.from_numpy((m > 127).astype(np.uint8)).long() 

            t0 = torch.full((H, W), padding_id, dtype=torch.long)
            t0[bin_mask == 1] = 0
            t1 = t0.clone()
            t1[t1 == padding_id] = 1
            t2 = torch.full((H, W), eos_id, dtype=torch.long)

            if step == 0:
                current = torch.full((H, W), padding_id, dtype=torch.long)
                target = t0
            elif step == 1:
                current = t0
                target = t1
            else:
                current = t1
                target = t2

        elif kind == "json_pos":
            m = np.array(raw_mask)  # uint8
            is_255 = torch.from_numpy((m == 255).astype(np.uint8)).bool()    
            is_not_255 = ~is_255                                          

            t0 = torch.full((H, W), padding_id, dtype=torch.long)
            t0[is_255] = 0

            t1 = torch.zeros((H, W), dtype=torch.long)
            t1[is_not_255] = 1  

            t2 = torch.full((H, W), eos_id, dtype=torch.long)

            if step == 0:
                current = torch.full((H, W), padding_id, dtype=torch.long)
                target = t0
            elif step == 1:
                current = t0
                target = t1
            else:  
                current = t1
                target = t2
        else:
            current = torch.full((H, W), padding_id, dtype=torch.long)
            target = torch.full((H, W), eos_id, dtype=torch.long)

        iter_channel = torch.full((H, W), float(rec["i"]), dtype=torch.float32)

        input_tensor = torch.cat([
            image,                       
            current.unsqueeze(0).float(),
            iter_channel.unsqueeze(0)    
        ], dim=0)

        return input_tensor, target, rec["final_gt"]

