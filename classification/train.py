import argparse
import os
import random
import time
import math
import pprint
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ==== project imports ====
from datasets import CIFAR10_LT
from datasets import CIFAR100_LT
from datasets import Places_LT
from datasets import ImageNet_LT
from datasets import iNa2018
from models import resnet, resnet_places, resnet_cifar
from utils import get_config
import losses
from report import summarize

# ============================================================
# Trainer Class
# ============================================================
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._set_seed()
        self._setup_logger()
 
        self._setup_device()
        
        self._build_dataloaders()
        self._count_samples_per_class()
        self._build_model()
        self._build_optimizer_loss()
        
    # -------------------------------
    def _setup_device(self):      
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.cfg.gpu}" if self.cfg.gpu is not None else "cuda:0")
        else:
            self.device = torch.device("cpu")
            
    # -------------------------------
    def _set_seed(self):
        if getattr(self.cfg, "deterministic", False):
            seed = 0
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # -------------------------------
    def _build_model(self):
        ds, bb = self.cfg.dataset.lower(), self.cfg.backbone.lower()
        self.block = None
        print(bb)
        if ds in ('cifar10', 'cifar100'):
            map_cifar = {
                "resnet20": resnet_cifar.resnet20_fe,
                "resnet32": resnet_cifar.resnet32_fe,
                "resnet44": resnet_cifar.resnet44_fe,
                "resnet56": resnet_cifar.resnet56_fe,
                "resnet110": resnet_cifar.resnet110_fe,
            }
            self.model = map_cifar.get(bb, resnet_cifar.resnet32_fe)()
            self.classifier = resnet_cifar.Classifier(feat_in=64, num_classes=self.cfg.num_classes, num_pem = self.cfg.num_pem)
        elif ds in ('imagenet', 'ina2018'):
            map_imagenet = {
                "resnet18": resnet.resnet18_fe,
                "resnet34": resnet.resnet34_fe,
                "resnet50": resnet.resnet50_fe,
                "resnet101": resnet.resnet101_fe,
                "resnet152": resnet.resnet152_fe,
            }
            self.model = map_imagenet.get(bb, resnet.resnet50_fe)()
            self.classifier = resnet.Classifier(feat_in=2048, num_classes=self.cfg.num_classes, num_pem = self.cfg.num_pem)
        elif ds == 'places':
            map_places = {
                "resnet50": resnet_places.resnet50_fe,
                "resnet101": resnet_places.resnet101_fe,
            }
            self.model = map_places.get(bb, resnet_places.resnet50_fe)(pretrained=True)
            self.classifier = resnet_places.Classifier(feat_in=2048, num_classes=self.cfg.num_classes, num_pem = self.cfg.num_pem)
            self.block = resnet_places.Bottleneck(2048, 512, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d)
        else:
            raise ValueError(f"Unsupported dataset: {self.cfg.dataset}")

        # Move to GPU (DataParallel if multi-GPU)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            self.classifier = nn.DataParallel(self.classifier)
            if self.block:
                self.block = nn.DataParallel(self.block)

        self.model = self.model.to(self.device)
        self.classifier = self.classifier.to(self.device)
        if self.block:
            self.block = self.block.to(self.device)

    # -------------------------------
    def _build_dataloaders(self):
        ds = self.cfg.dataset.lower()
        if ds == 'cifar10':
            dataset = CIFAR10_LT(False, root=self.cfg.data_path, imb_factor=self.cfg.imb_factor,
                                 batch_size=self.cfg.batch_size, num_works=self.cfg.workers)
        elif ds == 'cifar100':
            dataset = CIFAR100_LT(False, imb_type = self.cfg.imb_type, root=self.cfg.data_path, imb_factor=self.cfg.imb_factor,
                                  batch_size=self.cfg.batch_size, num_works=self.cfg.workers)
        elif ds == 'places':
            dataset = Places_LT(False, root=self.cfg.data_path, batch_size=self.cfg.batch_size, num_works=self.cfg.workers)
        elif ds == 'imagenet':
            dataset = ImageNet_LT(False, root=self.cfg.data_path, batch_size=self.cfg.batch_size, num_works=self.cfg.workers)
        elif ds == 'ina2018':
            dataset = iNa2018(False, root=self.cfg.data_path, batch_size=self.cfg.batch_size, num_works=self.cfg.workers)
        else:
            raise ValueError(f"Unsupported dataset: {self.cfg.dataset}")

        self.train_loader = dataset.train_instance
        self.val_loader = dataset.eval

    # -------------------------------
    def _build_optimizer_loss(self):
        lf = self.cfg.loss_function
        
        if lf != 'CE' and self.cfg.num_pem > 0:
            print("\033[5m" + "WARNING:" + "\033[0m" + f" the loss function {lf} may not be compatible with llr" )
        
        self.criterion_bce = nn.BCEWithLogitsLoss().to(self.device)
        if lf == 'CE': 
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            self.criterion_val = nn.CrossEntropyLoss().to(self.device)
        elif lf == 'LA': 
            self.criterion = losses.logit_adjustment.LogitAdjustment(cls_num_list = self.class_count).to(self.device)
            self.criterion_val = nn.CrossEntropyLoss().to(self.device)
        
        params = [{"params": self.model.parameters()}, {"params": self.classifier.parameters()}]
        if self.block:
            params.append({"params": self.block.parameters()})
        self.optimizer = optim.SGD(params, lr=self.cfg.lr,
                                   momentum=self.cfg.momentum, weight_decay=self.cfg.weight_decay)


    # -------------------------------
    def _setup_logger(self):
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        
        os.makedirs(os.path.join(self.cfg.log_dir, 'ckpt'), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.log_dir, 'result'), exist_ok=True)
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(self.cfg.log_dir, 'result', f'run_{self.run_id}.npy')

        self.exp_log = {
            "timestamp": self.run_id,
            "backbone_path": None,
            "block_path": None,
            "train_losses": np.zeros(self.cfg.num_epochs),
            "pem_losses": np.zeros(self.cfg.num_epochs),
            "val_losses": np.zeros(self.cfg.num_epochs),
            "val_accs": np.zeros(self.cfg.num_epochs),
            "per_class_accs": np.zeros([self.cfg.num_epochs, self.cfg.num_classes]),
            "config": {k: v for k, v in vars(self.cfg).items() if not k.startswith("__")},
        }

    # -------------------------------
    def _save_checkpoint(self):
        bb_file = os.path.join(self.cfg.log_dir, 'ckpt', f'run_{self.run_id}_backbone' )
        torch.save(self.model.state_dict(), bb_file)
        self.exp_log['backbone_path'] = bb_file
        if self.block:
            block_file = os.path.join(self.cfg.log_dir, 'ckpt', f'run_{self.run_id}_block' )
            torch.save(self.model.state_dict(), block_file)
            self.exp_log['block_path'] = block_file
        
        print(f'checkpoints saved at epoch {self.current_epoch+1}')
    # -------------------------------
    def adjust_lr(self):
        e = self.current_epoch + 1
        if getattr(self.cfg, "sch_mode", False) == 'cosine':
            lr_min, lr_max = 0, self.cfg.lr
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(e / self.cfg.num_epochs * math.pi))
        else:
            lr = self.cfg.lr

            milestones = sorted(getattr(self.cfg, "milestone", []))
            decay = getattr(self.cfg, "decay_rate", 0.1)

            # Apply decay once for epem milestone passed
            for m in milestones:
                if e > m:
                    lr *= decay
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    # -------------------------------
    def _count_samples_per_class(self):
        print("🔍 Counting samples per class in training dataset...")

        # Try to access the underlying dataset object
        # (support both raw Dataset and DataLoader-like wrappers)
        if hasattr(self.train_loader, "dataset"):
            dataset = self.train_loader.dataset
        else:
            dataset = self.train_loader  # some custom loaders are already datasets

        num_classes = getattr(self.cfg, "num_classes", None)
        if num_classes is None:
            # Try to infer automatically
            all_labels = []
            for _, label in dataset:
                all_labels.append(label)
            num_classes = max(all_labels) + 1
            print(f"⚠️ Inferred num_classes = {num_classes}")

        class_counts = [0] * num_classes

        # iterate once through dataset (no need for DataLoader)
        for i in range(len(dataset)):
            _, label = dataset[i]
            if isinstance(label, torch.Tensor):
                label = int(label.item())
            class_counts[label] += 1
        self.class_count = torch.tensor(class_counts, device = self.device)
        print(self.class_count)
    # -------------------------------
    def train_one_epoch(self):
        self.model.train()
        self.classifier.train()
        if self.block: self.block.train()

        total_loss, loss_pem_total, total_samples = 0.0, 0.0, 0
        start = time.time()
        for i, (images, target) in enumerate(self.train_loader):
            images, target = images.to(self.device), target.to(self.device)
            feats = self.model(images)
            if self.block:
                feats = self.block(feats)
            outputs, pems = self.classifier(feats)
            loss_main = self.criterion(outputs, target)
            
            loss_pem = torch.tensor(0.0).to(self.device)
            
            for i in pems:
                pems_true = i.gather(1, target.unsqueeze(1))
                targets_pem = torch.zeros_like(pems_true)
                loss_pem += self.criterion_bce(pems_true, targets_pem)
            
                            
            loss = loss_main  +  loss_pem
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            loss_pem_total += loss_pem.item() * bs
            total_samples += bs
            
        loss_pem_avg = loss_pem_total / max(1, total_samples)
        avg_loss = total_loss / max(1, total_samples)
        print(f"[Epoch {self.current_epoch+1}/{self.cfg.num_epochs}] Train loss={avg_loss:.4f} ({time.time()-start:.1f}s)")
        self.exp_log["train_losses"][self.current_epoch] = avg_loss
        self.exp_log["pem_losses"][self.current_epoch] = loss_pem_avg

    # -------------------------------
    def validate(self):
        self.model.eval()
        self.classifier.eval()
        if self.block:
            self.block.eval()
        total_loss, correct, total = 0.0, 0, 0
        num_classes = getattr(self.cfg, "num_classes", None)
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)

        with torch.no_grad():
            
            
            for images, target in self.val_loader:
                images, target = images.to(self.device), target.to(self.device)
                feats = self.model(images)
                if self.block:
                    feats = self.block(feats)
                outputs, pems = self.classifier(feats)
                
                if pems:
                    dimentions = tuple(range(1,outputs.ndim))
                    pem_sum = sum(h - h.mean(dim = dimentions, keepdim = True) for h in pems)
                    outputs = outputs + pem_sum / self.cfg.num_pem 
                    mask = (target == 10)                 
                loss = self.criterion_val(outputs, target)
                
                total_loss += loss.item() * images.size(0)
                total += images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(target).sum().item()

                # update per-class stats
                for t, p in zip(target.view(-1), preds.view(-1)):
                    class_total[int(t)] += 1
                    if p == t:
                        class_correct[int(t)] += 1

        # aggregate results
        avg_loss = total_loss / max(1, total)
        
        per_class_acc = 100 * class_correct / class_total
        acc = per_class_acc.mean().item()
        print(f"Validation - loss: {avg_loss:.4f}, acc@1: {acc:.2f}%")
        if (self.current_epoch + 1) % self.cfg.print_freq == 0:
            print(per_class_acc)
        self.exp_log["val_losses"][self.current_epoch] = avg_loss
        self.exp_log["val_accs"][self.current_epoch] = acc
        self.exp_log["per_class_accs"][self.current_epoch, :] = per_class_acc

    # -------------------------------
    def train(self):
        print(f"🚀 Training on {self.device} ({self.cfg.backbone}@{self.cfg.dataset})")
        for i, epoch in enumerate(range(self.cfg.num_epochs)):
            
            self.current_epoch = i
            self.adjust_lr()
            self.train_one_epoch()
            self.validate()
        
        if self.cfg.save_checkpoint:
            self._save_checkpoint()
        np.save(self.log_path, self.exp_log)
        summarize(self.log_path)
        print(f"✅ Done. Log saved at {self.log_path}")


# ============================================================
# Entry point
# ============================================================
def main():
    config, args = get_config()
    config.imb_type = 'exp'
    #print("Active Config:", config)
    print(pprint.pformat({k: v for k, v in vars(config).items() if not k.startswith('__')}))

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
