"""
KrossFuse: Feature Fusion via Kronecker Product and Random Projection

This module implements efficient feature fusion between CLIP and DINOv2 models
using Kronecker products and random projection techniques for image classification.

"""

import torch
import torchvision
import torchvision.transforms as transforms
import open_clip
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import argparse
from typing import Tuple, Optional
from timeit import default_timer as timer


class KrossFuse:
    """
    KrossFuse: Efficient feature fusion using CLIP and DINOv2 models.
    
    This class extracts features from both CLIP and DINOv2 vision transformers
    and fuses them using Kronecker products with random projection for
    dimensionality reduction.
    
    Args:
        device (str): Device to run models on ('cuda' or 'cpu')
        clip_model_name (str): CLIP model variant (default: 'ViT-B/32')
        dino_model_name (str): DINOv2 model variant (default: 'dinov2_vitb14')
    """
    
    def __init__(
        self, 
        device: str = "cuda",
        clip_model_name: str = 'ViT-B/32',
        dino_model_name: str = 'dinov2_vitb14'
    ):
        self.device = device
        
        # Initialize CLIP model
        print(f"Loading CLIP model: {clip_model_name}...")
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained='openai'
        )
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()
        
        # CLIP preprocessing
        self.clip_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])
        
        # Initialize DINOv2 model
        print(f"Loading DINOv2 model: {dino_model_name}...")
        self.dino_model = torch.hub.load("facebookresearch/dinov2", dino_model_name)
        self.dino_model.eval()
        self.dino_model.to(device)
        
        # DINOv2 preprocessing
        self.dino_preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            ),
        ])

    @torch.no_grad()
    def extract_features(
        self, 
        dataset, 
        batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from dataset using both CLIP and DINOv2 models.
        
        Args:
            dataset: PyTorch dataset to extract features from
            batch_size: Batch size for feature extraction
            
        Returns:
            Tuple of (clip_features, dino_features, labels)
        """
        clip_features_list = []
        dino_features_list = []
        labels_list = []
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=lambda x: x, 
            num_workers=4
        )
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            images = [item[0] for item in batch]
            labels = torch.tensor([item[1] for item in batch]).to(self.device)
            
            # Extract and normalize CLIP features
            clip_images = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
            clip_features = self.clip_model.encode_image(clip_images)
            clip_features = torch.nn.functional.normalize(clip_features, p=2, dim=1)
            clip_features_list.append(clip_features.cpu())
            
            del clip_images, clip_features
            torch.cuda.empty_cache()
            
            # Extract and normalize DINOv2 features
            dino_images = torch.stack([self.dino_preprocess(img) for img in images]).to(self.device)
            dino_features = self.dino_model(dino_images)
            dino_features = torch.nn.functional.normalize(dino_features, p=2, dim=1)
            dino_features_list.append(dino_features.cpu())
            
            del dino_images, dino_features
            torch.cuda.empty_cache()
            
            labels_list.append(labels.cpu())
        
        # Concatenate all batches
        clip_features = torch.cat(clip_features_list, dim=0)
        dino_features = torch.cat(dino_features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        return clip_features, dino_features, labels

    def hadamard_product_fusion(
        self, 
        clip_features: torch.Tensor, 
        dino_features: torch.Tensor, 
        n_components: int = 1000, 
        batch_size: int = 100, 
        random_state: int = 42
    ) -> torch.Tensor:
        """
        Fuse features using Hadamard product (element-wise multiplication) 
        after random projection.
        
        Args:
            clip_features: CLIP features tensor
            dino_features: DINOv2 features tensor
            n_components: Target dimensionality after projection
            batch_size: Batch size for processing
            random_state: Random seed for reproducibility
            
        Returns:
            Fused and projected features
        """
        clip_features = clip_features.to(self.device).float()
        dino_features = dino_features.to(self.device).float()
        
        n_samples = clip_features.shape[0]
        clip_dim = clip_features.shape[1]
        dino_dim = dino_features.shape[1]
        
        # Create random projection matrices
        torch.manual_seed(random_state)
        clip_random_matrix = (torch.rand(clip_dim, n_components, device=self.device, dtype=torch.float32) * 2 * np.sqrt(3) - np.sqrt(3)) / np.sqrt(n_components)
        dino_random_matrix = (torch.rand(dino_dim, n_components, device=self.device, dtype=torch.float32) * 2 * np.sqrt(3) - np.sqrt(3)) / np.sqrt(n_components)
        projected_features = torch.zeros((n_samples, n_components), device=self.device)
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            clip_batch = clip_features[i:batch_end]
            dino_batch = dino_features[i:batch_end]
            
            # Random projection then Hadamard product
            clip_projected = clip_batch @ clip_random_matrix
            dino_projected = dino_batch @ dino_random_matrix
            projected_features[i:batch_end] = clip_projected * dino_projected
            
            del clip_projected, dino_projected
            torch.cuda.empty_cache()
        
        return projected_features


def load_dataset(dataset_name: str, split: str = 'train', data_root: str = '.cache'):
    """
    Load dataset by name.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'cifar10', 'cifar100')
        split: Dataset split ('train' or 'test')
        data_root: Root directory for dataset storage
        
    Returns:
        PyTorch dataset object
    """
    dataset_name = dataset_name.lower()
    is_train = (split == 'train')
    
    if dataset_name == 'cifar10':
        return torchvision.datasets.CIFAR10(root=data_root, train=is_train, download=True, transform=None)
    elif dataset_name == 'cifar100':
        return torchvision.datasets.CIFAR100(root=data_root, train=is_train, download=True, transform=None)
    elif dataset_name == 'flowers102':
        return torchvision.datasets.Flowers102(root=data_root, split=split, download=True, transform=None)
    elif dataset_name == 'food101':
        return torchvision.datasets.Food101(root=data_root, split=split, download=True, transform=None)
    elif dataset_name == 'dtd':
        return torchvision.datasets.DTD(root=data_root, split=split, download=True, transform=None)
    elif dataset_name == 'aircraft':
        return torchvision.datasets.FGVCAircraft(root=data_root, split=split, download=True, transform=None)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported")


def main(args):
    """Main training and evaluation pipeline."""
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Define feature paths
    train_features_path = save_dir / f"{args.dataset}_train_features.pt"
    test_features_path = save_dir / f"{args.dataset}_test_features.pt"
    
    # Initialize KrossFuse
    fusion = KrossFuse(device=device, clip_model_name=args.clip_model, dino_model_name=args.dino_model)
    
    # Load datasets
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset = load_dataset(args.dataset, split='train', data_root=args.data_root)
    test_dataset = load_dataset(args.dataset, split='test', data_root=args.data_root)
    
    # Extract or load training features
    if not train_features_path.exists() or args.recompute_features:
        print("\n[1/4] Extracting training features...")
        train_clip_features, train_dino_features, train_labels = fusion.extract_features(
            train_dataset, batch_size=args.batch_size
        )
        torch.save({
            'clip_features': train_clip_features,
            'dino_features': train_dino_features,
            'labels': train_labels
        }, train_features_path)
        print(f"    Saved to: {train_features_path}")
    else:
        print("\n[1/4] Loading saved training features...")
        train_data = torch.load(train_features_path)
        train_clip_features = train_data['clip_features']
        train_dino_features = train_data['dino_features']
        train_labels = train_data['labels']
    
    print(f"    CLIP feature dim: {train_clip_features.shape[1]}")
    print(f"    DINO feature dim: {train_dino_features.shape[1]}")
    
    # Extract or load test features
    if not test_features_path.exists() or args.recompute_features:
        print("\n[2/4] Extracting test features...")
        test_clip_features, test_dino_features, test_labels = fusion.extract_features(
            test_dataset, batch_size=args.batch_size
        )
        torch.save({
            'clip_features': test_clip_features,
            'dino_features': test_dino_features,
            'labels': test_labels
        }, test_features_path)
        print(f"    Saved to: {test_features_path}")
    else:
        print("\n[2/4] Loading saved test features...")
        test_data = torch.load(test_features_path)
        test_clip_features = test_data['clip_features']
        test_dino_features = test_data['dino_features']
        test_labels = test_data['labels']
    
    start = timer()
    
    # Fuse features
    print(f"\n[3/4] Fusing features (n_components={args.n_components})...")
    train_fused = fusion.hadamard_product_fusion(
        train_clip_features, 
        train_dino_features, 
        n_components=args.n_components,
        batch_size=100
    )
    print(f"    Fused feature dim: {train_fused.shape}")
    
    test_fused = fusion.hadamard_product_fusion(
        test_clip_features, 
        test_dino_features, 
        n_components=args.n_components,
        batch_size=100
    )
    
    # Convert to numpy for sklearn
    train_fused = train_fused.cpu().numpy()
    test_fused = test_fused.cpu().numpy()
    train_labels = train_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()
    
    # Train classifier
    print(f"\n[4/4] Training logistic regression (C={args.C})...")
    classifier = LogisticRegression(
        random_state=42, 
        C=args.C, 
        max_iter=args.max_iter, 
        verbose=1 if args.verbose else 0
    )
    classifier.fit(train_fused, train_labels)
    
    # Evaluate
    train_predictions = classifier.predict(train_fused)
    train_accuracy = np.mean((train_labels == train_predictions).astype(float)) * 100.
    
    test_predictions = classifier.predict(test_fused)
    test_accuracy = np.mean((test_labels == test_predictions).astype(float)) * 100.
    
    end = timer()
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Training Accuracy:   {train_accuracy:.2f}%")
    print(f"Test Accuracy:       {test_accuracy:.2f}%")
    print(f"Total Runtime:       {end - start:.2f} seconds")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KrossFuse: Feature Fusion via Kronecker Product and Random Projection"
    )
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar100',
                       choices=['cifar10', 'cifar100', 'flowers102', 'food101', 'dtd', 'aircraft'],
                       help='Dataset name')
    parser.add_argument('--data_root', type=str, default='.cache',
                       help='Root directory for dataset storage')
    parser.add_argument('--save_dir', type=str, default='features_save',
                       help='Directory to save extracted features')
    
    # Model arguments
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                       choices=['ViT-B/32', 'ViT-L/14', 'ViT-L-14-336'],
                       help='CLIP model variant')
    parser.add_argument('--dino_model', type=str, default='dinov2_vitb14',
                       choices=['dinov2_vitb14', 'dinov2_vitl14'],
                       help='DINOv2 model variant')
    
    # Feature extraction arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for feature extraction')
    parser.add_argument('--recompute_features', action='store_true',
                       help='Force recompute features even if cached')
    
    # Fusion arguments
    parser.add_argument('--n_components', type=int, default=3000,
                       help='Target dimensionality after random projection')
    
    # Classifier arguments
    parser.add_argument('--C', type=float, default=0.316,
                       help='Regularization parameter for logistic regression')
    parser.add_argument('--max_iter', type=int, default=1000,
                       help='Maximum iterations for logistic regression')
    
    # Other arguments
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    main(args)