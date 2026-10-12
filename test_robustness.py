import torch
import torch.distributed as dist
import torch.nn.parallel as nnp
import os
from torchvision import transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from metric import PixelF1
from PIL import Image
import numpy as np
import albumentations as albu
import random
import json

class AbstractTransformWrapper:
    def __init__(self, param_list):
        self.param_list = param_list
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __str__(self):
        return self.__class__.__name__[:-7] # 返回类名
    
    def __next__(self):
        if self.index < len(self.param_list):
            param = self.param_list[self.index]
            self.index += 1
            
            if param == 0:
                return param, None
            else:
                return self._get_transform(param)
        else:
            self.index = 0
            raise StopIteration
    
    def _get_transform(self, param):
        raise NotImplementedError
    
        
class GaussianBlurWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.GaussianBlur(
            blur_limit=(param, param),
            always_apply=True,
            p=1.0
        )
        
class GaussianNoiseWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.GaussNoise(
            var_limit=(param, param),
            always_apply=True,
            p=1.0
        )
        
class JpegCompressionWrapper(AbstractTransformWrapper):
    def _get_transform(self, param):
        return param, albu.JpegCompression(
            quality_lower = param-1,
            quality_upper = param,
            p=1.0
        )

class AutoregressiveMaskGenerator:
    def __init__(self, model, image_size=(224,224), max_steps=10):
        self.model = model
        self.image_size = image_size
        self.max_steps = max_steps
        self.num_classes = model.num_classes
        self.eos_token_id = model.eos_token_id
    
    def generate(self, images):
        """
        批量生成 mask
        输入:
        - images: 输入图像 (B, 3, H, W)
        输出:
        - 二维列表，每个子列表是一个样本的生成结果 [B, [mask1, mask2, ...]]
        """
        self.model.eval()
        B, _, H, W = images.shape
        generated_masks = [[] for _ in range(B)]  # 二维列表，存储每个样本的生成结果
    
        with torch.no_grad():
            for b in range(B):  # 遍历批量中的每个样本
                image = images[b]  # 当前样本的图像 (3, H, W)
                current_mask = (self.num_classes-1) * torch.ones((H, W), dtype=torch.long, device=images.device)  # 初始 mask
                iter_val = 0.0
                for step in range(self.max_steps):
                    # 构建输入张量
                    iter_channel = torch.full((H, W), iter_val, dtype=torch.float32, device=images.device)
                    input_tensor = torch.cat([image, current_mask.unsqueeze(0), iter_channel.unsqueeze(0)], dim=0)  # (5, H, W)
                    input_tensor = input_tensor.unsqueeze(0)  # 添加 batch 维度 (1, 5, H, W)
                    
                    # 预测
                    logits = self.model(input_tensor)  # (1, num_classes, H, W)
                    pred_mask = torch.argmax(logits, dim=1).squeeze(0)  # (H, W)
                    
                    # 保存当前预测的 mask
                    generated_masks[b].append(pred_mask)
                    
                    # 检查是否生成 EOS
                    if torch.all(pred_mask == self.eos_token_id):
                        # print(f"样本 {b} 在 step {step} 生成 EOS，终止预测。")
                        break
                    
                    # 更新 current_mask
                    current_mask = pred_mask
                    iter_val += 1.0

        return generated_masks
    
class ImageMaskDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, mask_folder, image_transform, mask_transform, common_transforms=None):
        
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder)])
        self.mask_paths = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder)])
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.common_transforms = common_transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np

        # 读取原图和掩码
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')



        if self.common_transforms is not None:
            image = np.array(image)  # H, W, C
            mask = np.array(mask)    # H, W
            res_dict = self.common_transforms(image=image, mask=mask)
            image_np = res_dict['image']
            mask_np = res_dict['mask']

            # 转换 image
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)

            # 转换 mask
            if mask_np.dtype != np.uint8:
                mask_np = mask_np.astype(np.uint8)
            mask = Image.fromarray(mask_np)

        # 注意这里要返回 transform 之后的结果
        return self.image_transform(image), self.mask_transform(mask)

    

class ImageMaskDatasetJson(torch.utils.data.Dataset):
    def __init__(self, json_path, image_transform=None, mask_transform=None, common_transforms=None):
        """
        :param json_path: JSON file path, format is {"image_path": "mask_path", ...}
        :param image_transform: Image preprocessing function
        :param mask_transform: Mask preprocessing function
        """

        
        with open(json_path, 'r') as f:
            self.path_pairs = json.load(f)  # Load path mapping dictionary

        # Separate image and mask paths
        self.image_paths = []
        self.mask_paths = []
        for pairs in self.path_pairs:
            self.image_paths.append(pairs[0])
            self.mask_paths.append(pairs[1])
        
        # Validate path validity
        self._validate_paths()
        
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.common_transforms = common_transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        if self.common_transforms is not None:
            image = np.array(image)  # H, W, C
            mask = np.array(mask)    # H, W
            res_dict = self.common_transforms(image=image, mask=mask)
            image_np = res_dict['image']
            mask_np = res_dict['mask']

            # 转换 image
            if image_np.dtype != np.uint8:
                image_np = (image_np * 255).astype(np.uint8)
            image = Image.fromarray(image_np)

            # 转换 mask
            if mask_np.dtype != np.uint8:
                mask_np = mask_np.astype(np.uint8)
            mask = Image.fromarray(mask_np)
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        return image, mask

    def _validate_paths(self):
        """Validate all paths"""
        import os
        
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image path not found: {img_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask path not found: {mask_path}")
class DistributedPixelF1:
    def __init__(self, device):
        # 使用两个tensor分别存储f1总和和样本数
        self.f1_sum = torch.tensor(0.0, device=device)
        self.sample_count = torch.tensor(0, device=device)
        self.device = device
    
    def update(self, f1_score, num_samples):
        """仅本地更新，不进行任何通信"""
        self.f1_sum += f1_score
        self.sample_count += num_samples
    
    def compute(self):
        """使用all_reduce进行高效同步"""
        # 将本地结果同步到全局
        dist.all_reduce(self.f1_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.sample_count, op=dist.ReduceOp.SUM)
        
        # 计算平均F1
        avg_f1 = (self.f1_sum / self.sample_count).item() if self.sample_count > 0 else 0.0
        
        return avg_f1,self.f1_sum, self.sample_count.item()

def test_model_distributed(
    model=None, 
    generator=None, 
    image_size=(224, 224), 
    device=None, 
    batch_size=2, 
    world_size=None, 
    rank=None,
    num_class = 11
):
    """
    Distributed testing of a model across multiple GPUs
    
    Args:
        model: The model to test
        generator: Autoregressive mask generator
        image_size: Image resize dimensions
        device: Device to run on
        batch_size: Batch size per GPU
        world_size: Total number of GPUs
        rank: Current GPU rank
    
    Returns:
        dict: Results for each dataset
    """
    # Setup distributed environment if not already set
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    
    # If a generator is not provided, create one using the model
    if generator is None and model is not None:
        generator = AutoregressiveMaskGenerator(model, image_size=image_size, max_steps=10)
    if generator is None and model is None:
        raise ValueError("Either model or generator must be provided")
    
    # Set up transforms
    image_transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mask_transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor()
    ])
    
    # Define datasets to test
    datasets = [
        # CASIA 1.0
        {
            'name': 'CASIA1.0',
            'type': 'folder',
            'image_folder': '/mnt/data0/public_datasets/IML/CASIA1.0/Tp',
            'mask_folder': '/mnt/data0/public_datasets/IML/CASIA1.0/Gt_binary'
        },
        # Columbia
        {
            'name': 'Columbia',
            'type': 'json',
            'json_path': '/mnt/data0/public_datasets/IML/Columbia.json'
        },
        # Coverage
        {
            'name': 'Coverage',
            'type': 'json',
            'json_path': '/mnt/data0/public_datasets/IML/coverage.json'
        },
        # IMD 20
        # {
        #     'name': 'IMD_20',
        #     'type': 'folder',
        #     'image_folder': '/mnt/data0/public_datasets/IML/IMD_20_1024/Tp',
        #     'mask_folder': '/mnt/data0/public_datasets/IML/IMD_20_1024/Gt_binary'
        # },
        # NIST16
        {
            'name': 'NIST16',
            'type': 'folder',
            'image_folder': '/mnt/data0/public_datasets/IML/NIST16_1024/Tp',
            'mask_folder': '/mnt/data0/public_datasets/IML/NIST16_1024/Gt'
        },
        {
            'name': 'Autosplice',
            'type': 'json',
            'json_path':'/mnt/data0/public_datasets/IML/Autosplice/autosplice.json'
        },
        {
            'name': 'CocoGlide',
            'type': 'json',
            'json_path':'/mnt/data0/public_datasets/IML/CocoGlide/cocoglide.json'
        }
    ]
    
    
    robustness_list = [
            GaussianBlurWrapper([0, 3, 7, 11, 15, 19, 23]),
            GaussianNoiseWrapper([3, 7, 11, 15, 19, 23]), 
            JpegCompressionWrapper([50, 60, 70, 80, 90, 100])
    ]

    with torch.no_grad():  # Use no_grad context for testing
        attacks = {}
        for attack_wrapper in robustness_list:
            for attack_param, attack_transform in attack_wrapper:
                results = {}
                for dataset_info in datasets:
                    dataset_name = dataset_info['name']
                    if rank == 0:
                        print(f"Testing on {dataset_name}...")
                        print('attack_name:', attack_param, str(attack_transform))

                    # Create dataset based on type
                    if dataset_info['type'] == 'folder':
                        dataset = ImageMaskDataset(
                            dataset_info['image_folder'], 
                            dataset_info['mask_folder'], 
                            image_transform, 
                            mask_transform,
                            common_transforms=attack_transform
                        )
                    elif dataset_info['type'] == 'json':
                        dataset = ImageMaskDatasetJson(
                            dataset_info['json_path'],
                            image_transform, 
                            mask_transform,
                            common_transforms=attack_transform
                        )
                    
                    # Create distributed sampler
                    sampler = DistributedSampler(
                        dataset, 
                        num_replicas=world_size, 
                        rank=rank, 
                        shuffle=False,
                        drop_last=True
                    )
                    
                    # Create dataloader
                    dataloader = DataLoader(
                        dataset, 
                        batch_size=batch_size, 
                        sampler=sampler,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True
                    )
                    
                    # Distributed F1 metric
                    dist_f1 = DistributedPixelF1(device)
                    
                    # Process each batch
                    local_results = []
                    with tqdm(
                        dataloader, 
                        desc=f"Processing {dataset_name}", 
                        unit="batch", 
                        ncols=100,
                        disable=(rank != 0)  # Only show progress on rank 0
                    ) as pbar:
                        for batch_images, batch_masks in pbar:
                            batch_images = batch_images.to(device)
                            batch_masks = batch_masks.to(device)
                            
                            # Generate masks
                            gen_masks = generator.generate(batch_images)
                            
                            # Process each sample in the batch
                            for i in range(len(gen_masks)):
                                mask_sequence = torch.stack(gen_masks[i], dim=0).unsqueeze(1)
                                
                                # Check for all-254 masks
                                ratio = (mask_sequence == num_class - 2).float().mean(dim=(1, 2))
                                all_254_indices = (ratio >= 0.9).nonzero(as_tuple=True)[0]
                                target_mask = mask_sequence[all_254_indices[0] - 1] if len(all_254_indices) > 0 else mask_sequence[-1]

                                # Process mask logic
                                unique_values = torch.unique(target_mask)
                                unique_values = unique_values[unique_values != (num_class-1)]
                                max_value = unique_values.max() if len(unique_values) > 0 else None
                                processed_mask = torch.where(target_mask == max_value, 0, 1) if max_value else torch.ones_like(target_mask)
                                
                                # Update F1 score
                                f1_score = PixelF1().batch_update(predict=processed_mask.unsqueeze(0), mask=batch_masks[i].unsqueeze(0))
                                local_results.append(f1_score)
                                
                                # Distributed update
                                dist_f1.update(f1_score.sum(), len(f1_score))
                    # 2. 主进程处理剩余样本（如果有）
                    if rank == 0:
                        # 计算总样本数和已处理样本数
                        total_samples = len(dataset)
                        distributed_samples = (total_samples // (batch_size * world_size)) * batch_size * world_size
                        remaining_samples = total_samples - distributed_samples
                        
                        if remaining_samples > 0:
                            print(f"\nProcessing remaining {remaining_samples} samples on main process...")
                            
                            # 创建单进程数据加载器
                            remaining_indices = range(distributed_samples, total_samples)
                            remaining_dataset = torch.utils.data.Subset(dataset, remaining_indices)
                            remaining_loader = DataLoader(
                                remaining_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=True
                            )
                            
                            # 处理剩余样本
                            with tqdm(remaining_loader, desc="Processing remaining samples") as pbar:
                                for batch_images, batch_masks in pbar:
                                    batch_images = batch_images.to(device)
                                    batch_masks = batch_masks.to(device)
                                    gen_masks = generator.generate(batch_images)

                                    for i in range(len(gen_masks)):
                                        mask_sequence = torch.stack(gen_masks[i], dim=0).unsqueeze(1)
                                        # Check for all-254 masks
                                        ratio = (mask_sequence == num_class - 2).float().mean(dim=(1, 2))
                                        all_254_indices = (ratio >= 0.9).nonzero(as_tuple=True)[0]
                                        target_mask = mask_sequence[all_254_indices[0] - 1] if len(all_254_indices) > 0 else mask_sequence[-1]

                                        # Process mask logic
                                        unique_values = torch.unique(target_mask)
                                        unique_values = unique_values[unique_values != (num_class-1)]
                                        max_value = unique_values.max() if len(unique_values) > 0 else None
                                        processed_mask = torch.where(target_mask == max_value, 0, 1) if max_value else torch.ones_like(target_mask)
                                        
                                        # Update F1 score
                                        f1_score = PixelF1().batch_update(predict=processed_mask.unsqueeze(0), mask=batch_masks[i].unsqueeze(0))
                                        local_results.append(f1_score)
                                        dist_f1.update(f1_score.sum(), len(f1_score))
                    
                    # Compute final F1 score
                    mean_f1, total_f1, total_samples = dist_f1.compute()
                    results[dataset_name] = {
                            'mean_f1': mean_f1,
                            'total_f1': total_f1,
                            'total_samples': total_samples
                        }

                    if rank == 0:
                        print('    ')
                        print(f"{dataset_name} Results:")
                        print(f"  Mean F1 Score: {mean_f1:.4f}")
                        print(f"  Total F1 Score: {total_f1:.4f}")
                        print(f"  Total Samples: {total_samples}")
                    dist.barrier()
                mean_f1_list = [data['mean_f1'] for data in results.values()]
                simple_avg = sum(mean_f1_list) / len(mean_f1_list) if mean_f1_list else 0.0
                attacks[str(attack_transform) + str(attack_param)] = simple_avg
                if rank == 0:
                    print()
                    print('Attack:',str(attack_transform) + str(attack_param),':', simple_avg)
    return attacks
from rita import RITA

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ckpt',
        type=str,
        help='Path to checkpoint file'
    )
    return parser.parse_args()

def main():
    # Initialize distributed environment
    dist.init_process_group(backend='nccl')
    
    # Get world size and rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Prepare model
    image_size = (512, 512)
    num_classes = 4
    
    # Load model
    model = RITA(num_classes=num_classes).to(device)

    args = parse_args()
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    # Wrap model in DistributedDataParallel
    model = nnp.DistributedDataParallel(model, device_ids=[local_rank])
    
    # Initialize generator
    generator = AutoregressiveMaskGenerator(model.module, image_size=image_size, max_steps=10)
    
    # Test model
    results = test_model_distributed(
        model=model.module, 
        generator=generator, 
        image_size=image_size, 
        device=device,
        batch_size=2,  # Adjust based on your GPU memory
        world_size=world_size, 
        rank=rank,
        num_class= num_classes
    )
    print(results)
    # Finalize
    dist.destroy_process_group()

if __name__ == '__main__':
    main()


