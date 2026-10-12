import os, glob, math, numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from rita import RITA
from test import test_model_distributed
import builtins
import torch, random, numpy as np
from balanced_data import UnifiedNextMaskDatasetEpisodic
import albumentations as albu
_original_print = builtins.print

def print(*args, **kwargs):
    if not dist.is_available() or not dist.is_initialized():
        return _original_print(*args, **kwargs)
    if dist.get_rank() == 0:
        return _original_print(*args, **kwargs)

builtins.print = print


class EdgeMaskGenerator(torch.nn.Module):
    """generate the 'edge bar' for a 0-1 mask Groundtruth of a image
    Algorithm is based on 'Morphological Dilation and Difference Reduction'
    
    Which implemented with fixed-weight Convolution layer with weight matrix looks like a cross,
    for example, if kernel size is 3, the weight matrix is:
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]

    """
    def __init__(self, kernel_size = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    def _dilate(self, image, kernel_size=3):
        """Doings dilation on the image

        Args:
            image (_type_): 0-1 tensor in shape (B, C, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert image.shape[2] > kernel_size and image.shape[3] > kernel_size, "Image must be larger than kernel size"
        
        kernel = torch.zeros((1, 1, kernel_size, kernel_size)).to(image.device)
        kernel[0, 0, kernel_size // 2: kernel_size//2+1, :] = 1
        kernel[0, 0, :,  kernel_size // 2: kernel_size//2+1] = 1
        kernel = kernel.float()
        # print(kernel)
        res = F.conv2d(image, kernel.view([1,1,kernel_size, kernel_size]),stride=1, padding = kernel_size // 2)
        return (res > 0) * 1.0


    def _find_edge(self, image, kernel_size=3, return_all=False):
        """Find 0-1 edges of the image

        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """

        image = torch.tensor(image).float()
        shape = image.shape
        
        if len(shape) == 2:
            image = image.reshape([1, 1, shape[0], shape[1]])
        if len(shape) == 3:
            image = image.reshape([1, shape[0], shape[1], shape[2]])   
        assert image.shape[1] == 1, "Image must be single channel"
        
        img = self._dilate(image, kernel_size=kernel_size)
        
        erosion = self._dilate(1-image, kernel_size=kernel_size)

        diff = -torch.abs(erosion - img) + 1
        diff = (diff > 0) * 1.0
        # res = dilate(diff)
        diff = diff
        if return_all :
            return diff, img, erosion
        else:
            return diff
    
    def forward(self, x, return_all=False):
        """
        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        return self._find_edge(x, self.kernel_size, return_all=return_all)


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print('Not using distributed mode')
        rank = 0
        world_size = 1
        local_rank = 0
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank





def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """denormalize image with mean and std
    """
    image = image.clone().detach().cpu()
    image = image * torch.tensor(std).view(3, 1, 1)
    image = image + torch.tensor(mean).view(3, 1, 1)
    return image

def process_and_visualize_mask(mask, save_path=None):
    processed_mask = mask.copy()
    processed_mask *= 10
    plt.figure(figsize=(8, 8))
    plt.imshow(processed_mask, cmap="viridis", vmin=0, vmax=100)
    plt.colorbar(label="Mask Value")
    plt.title("Processed Mask")
    plt.axis("off")
    plt.show()

    if save_path:
        processed_image = Image.fromarray(processed_mask.astype(np.uint8))
        processed_image.save(save_path)
        print(f"Processed mask saved to {save_path}")

def calculate_accuracy(output, target):
    pred = torch.argmax(output, dim=1)
    correct = (pred == target).sum()
    total = target.numel()
    accuracy = correct.float() / total
    return accuracy

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_loader_for_epoch(dataset, world_size, rank, epoch, batch_size=8, num_workers=12):
    dataset.set_epoch(epoch)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )
    sampler.set_epoch(epoch) 

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )
    return loader, sampler



def train_model(model, dataset,batch_szie, criterion, optimizer, num_epochs, device, rank, scheduler, num_classes, mvss_protocal):
    best_acc = -torch.inf
    scaler = torch.amp.GradScaler()  
    best_f1 = -torch.inf
    best_gen = -torch.inf
    best_count = 0
    edge_generator = EdgeMaskGenerator(7).to(model.device)
    world_size = dist.get_world_size() 
    for epoch in range(num_epochs):
        dataloader, sampler = build_loader_for_epoch(
        dataset, world_size, rank, epoch,
        batch_size=batch_szie, num_workers=12
    )
        model.train()
        total_loss = 0.0
        total_edge_loss = 0.0
        total_acc = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") if rank == 0 else dataloader
        for inputs, targets, final_gt in pbar:
            inputs = inputs.to(device, non_blocking=True)
            iter_values = inputs[:, 4, 0, 0]
            targets = targets.to(device, non_blocking=True)
            B,_, H, W = inputs.shape
            optimizer.zero_grad(set_to_none=True)

            indices = iter_values.view(B, 1, 1, 1).expand(B, 1, H, W)
            iter_mask = (targets.unsqueeze(1)==indices).float()
            edge_mask = edge_generator(iter_mask)
            with autocast(): 
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                selected_logits = torch.gather(outputs, dim=1, index=indices.long())
                edge_loss = F.binary_cross_entropy_with_logits(
                    input=selected_logits,
                    target=iter_mask,
                    weight=edge_mask
                ) * 20
                loss  = loss + edge_loss
                acc = calculate_accuracy(outputs, targets)
            scaler.scale(loss).backward() 
            scaler.step(optimizer)  
            scaler.update()  
            dist.all_reduce(acc, op=dist.ReduceOp.SUM)
           
            acc /= world_size
            acc = acc.item()
            total_acc += acc
            total_loss += loss.item()
            total_edge_loss += edge_loss.item()
            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix(loss=loss.item(),edge_loss=edge_loss.item(),acc=acc)

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        avg_edge_loss = total_edge_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        generator = AutoregressiveMaskGenerator(model.module, image_size=(image_size, image_size), max_steps=10)
        results = test_model_distributed(generator=generator, image_size=(image_size, image_size), device=device,num_class=num_classes, is_mvss_protocal=mvss_protocal)
        if rank == 0:
            keys = ['CASIA1.0', 'Columbia', 'Coverage', 'NIST16', 'Autosplice', 'CocoGlide', 'IMD_20']
            valid_values = [results[k]['mean_f1'] for k in keys if k in results]  
            mean = sum(valid_values) / len(valid_values) if valid_values else 0
            keys_gen = ['Columbia', 'Coverage', 'NIST16', 'Autosplice', 'CocoGlide', 'IMD_20']
            gen_values = [results[k]['mean_f1'] for k in keys_gen if k in results] 
            mean_gen = sum(gen_values) / len(gen_values) if gen_values else 0
            if mean > best_f1:
                best_f1 = mean
                torch.save(model.module.state_dict(), f"{ckpt_path}/best.pth")
                print('Best Average F1:', mean)
            if mean_gen > best_gen:
                best_gen = mean_gen
                torch.save(model.module.state_dict(), f"{ckpt_path}/best_gen.pth")
                print('Best Gen Average F1:', mean_gen)
            print('Average F1:', mean)
            torch.save(model.module.state_dict(), f"{ckpt_path}/last.pth")
            print(f"Epoch {epoch+1}/{num_epochs}, Total_Loss: {avg_loss:.4f}, Edge_Loss: {avg_edge_loss:.4f}, ACC:  {avg_acc:.4f}, LR: {current_lr:.7f}")
    return model

class AutoregressiveMaskGenerator:
    def __init__(self, model, image_size=(224,224), max_steps=10):
        self.model = model
        self.image_size = image_size
        self.max_steps = max_steps
        self.num_classes = model.num_classes
        self.eos_token_id = model.eos_token_id
        self.padding_id = self.eos_token_id + 1
    def generate(self, images):
        self.model.eval()
        B, _, H, W = images.shape
        generated_masks = [[] for _ in range(B)] 

        with torch.no_grad():
            for b in range(B):  
                image = images[b] 
                current_mask = self.padding_id * torch.ones((H, W), dtype=torch.long, device=images.device)  
                iter_val = 0.0
                for step in range(self.max_steps):
                    iter_channel = torch.full((H, W), iter_val, dtype=torch.float32, device=images.device)
                    input_tensor = torch.cat([image, current_mask.unsqueeze(0), iter_channel.unsqueeze(0)], dim=0) 
                    input_tensor = input_tensor.unsqueeze(0) 
                    logits = self.model(input_tensor)  
                    pred_mask = torch.argmax(logits, dim=1).squeeze(0) 
                    generated_masks[b].append(pred_mask)
                    if torch.all(pred_mask == self.eos_token_id):
                        break
                    current_mask = pred_mask
                    iter_val += 1.0

        return generated_masks


def visualize_generation(model, images, gt_mask_paths, mask_sequences, save_path=None):

    B = len(images) 
    num_steps = max(len(seq) for seq in mask_sequences) 

    assert len(gt_mask_paths) == B, "gt_mask_paths 的长度必须与 B 一致"

    plt.figure(figsize=(4 * (num_steps + 2), 4 * B))  # 设置画布大小

    for b in range(B):
        plt.subplot(B, num_steps + 2, b * (num_steps + 2) + 1)
        image = denormalize(images[b]) 
        img_np = image.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)  
        plt.imshow(img_np)
        plt.title(f"Input Image {b}")
        plt.axis("off")

        gt_mask = np.array(Image.open(gt_mask_paths[b]).convert("L")) 
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).float() 

        # 将 GT 掩码通过最近邻插值调整到输入图像的尺寸
        gt_mask_resized = F.interpolate(gt_mask, size=images.shape[2:], mode="nearest")  
        gt_mask_resized = gt_mask_resized.squeeze(0).squeeze(0).numpy().astype(np.uint8) 

        # 可视化 GT 掩码
        plt.subplot(B, num_steps + 2, b * (num_steps + 2) + 2)
        plt.imshow(gt_mask_resized, cmap="gray", vmin=0, vmax=100) 
        plt.title(f"GT Mask {b}")
        plt.axis("off")

        seq = mask_sequences[b]
        for i in range(num_steps):
            plt.subplot(B, num_steps + 2, b * (num_steps + 2) + i + 3)
            if i < len(seq): 
                m_np = seq[i].squeeze(0).cpu().numpy() 
                m_np = np.clip(m_np * 10, 0, 100).astype(np.uint8) 
                plt.imshow(m_np, cmap="gray", vmin=0, vmax=100) 
            else:  
                plt.imshow(np.zeros_like(gt_mask_resized), cmap="gray", vmin=0, vmax=100)
            plt.title(f"Step {i}")
            plt.axis("off")

    plt.tight_layout()

    if save_path:
        folder_path = os.path.dirname(save_path)
        os.makedirs(folder_path, exist_ok=True)
        plt.savefig(save_path)


def main(image_size):
    rank, world_size, local_rank = init_distributed_mode()
    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    is_main = (rank == 0)
    device_local = torch.device("cuda", local_rank)
    writer = SummaryWriter(log_dir=runs_path) if is_main else None
    image_size = (image_size,image_size)
    num_classes = 4
    train_transform = albu.Compose([
            albu.RandomScale(scale_limit=0.2, p=1), 
            # Flips
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            # Brightness and contrast fluctuation
            albu.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=0.1,
                p=1
            ),
            albu.ImageCompression(
                quality_lower = 70,
                quality_upper = 100,
                p = 0.2
            ),
            # Rotate
            albu.RandomRotate90(p=0.5),
            # Blur
            albu.GaussianBlur(
                blur_limit = (3, 7),
                p = 0.2
            ),
    ])

    if os.path.isdir(data_path):
         mvss_protocal = True
         dataset = UnifiedNextMaskDatasetEpisodic(data_path=data_path, num_classes=num_classes, common_transforms=train_transform,num_per_dataset=5123)
    else:
        mvss_protocal = False
        dataset = UnifiedNextMaskDatasetEpisodic(data_path=data_path, num_classes=num_classes, common_transforms=train_transform)

    if is_main:
        print("Loaded dataset, total samples:", len(dataset))
    model = RITA().to(device_local)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if is_main:
        total_params = sum(p.numel() for p in model.module.parameters())
        print(f"Total parameters: {total_params:,}")
    weights = torch.ones(num_classes).to(local_rank)
    weights[0] = 1
    weights[1] = 0.2
    weights[num_classes-1] = 0.1
    if rank == 0:
        print('weights：',weights[:2],weights[num_classes-2],weights[num_classes-1])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = epoch
    T_max = num_epochs  
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=5e-7)  
    batch_szie = args.batch_size
    model = train_model(model, dataset,batch_szie, criterion, optimizer, num_epochs, device_local, rank, scheduler,num_classes,mvss_protocal)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        '--path', 
        type=str, 
        required=True, 
        help='Path to the output directory (e.g., results/mamba)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        required=False,
        default=512
    )
    parser.add_argument(
        '--epoch',
        type=int,
        required=False,
        default=512
    )
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        default=512
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8
    )
    args = parser.parse_args()
    output_dir = args.path
    image_size = args.image_size
    epoch = args.epoch
    data_path = args.data_path
    images_path = os.path.join(output_dir,'images')
    ckpt_path = os.path.join(output_dir,'ckpts')
    runs_path = os.path.join(output_dir,'runs')
    main(image_size)
