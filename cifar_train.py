import yaml
import wandb
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
import argparse
from models import resnet50, resnet50bn
import random
import numpy as np
from utils import compute_stage_grad_norms

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model using YAML configuration.")
parser.add_argument('--config', type=str, required=True, help="Path to YAML configuration file")
args = parser.parse_args()

# Load the YAML configuration
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Set seed for reproducibility
seed = config['training']['seed'] 
print(f'Using seed: {seed}')
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Ensure deterministic behavior in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data augmentations for training
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
    transforms.RandomHorizontalFlip(),     # Horizontal flip
    ToTensor(),                            # Convert to tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR-10 normalization
])

# Transforms for testing (no augmentations)
test_transforms = transforms.Compose([
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# Datasets
if config['training']['dataset']=='cifar10':
    train_dataset = CIFAR10(root="./data_cifar", train=True, download=True, transform=train_transforms)
    test_dataset = CIFAR10(root="./data_cifar", train=False, download=True, transform=test_transforms)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

elif config['training']['dataset']=='cifar100':
    # Load CIFAR-100 dataset
    train_dataset = CIFAR100(root="./data_cifar100", train=True, download=True, transform=train_transforms)
    test_dataset = CIFAR100(root="./data_cifar100", train=False, download=True, transform=test_transforms)

else:
    raise ValueError(f"Unsupported dataset: {config['training']['dataset']}")


# Hyperparameters
batch_size = config['training']['batch_size']
epochs = config['training']['epochs']
lr = config['training']['lr']
weight_decay = config['training']['weight_decay']
scheduler_type = config['model_params']['skip_scheduler']
final_skip_values = config['model_params']['final_skip_values']
update_per_batch = config['model_params'].get('update_per_batch', False)

# Extended training parameters
extended_training = config.get("extended_training", {}).get("extend", False)
if extended_training:
    extended_final_skip_values = config.get("extended_training", {}).get("final_skip_values", []) 
    extended_scheduler_type = config['extended_training']['skip_scheduler']
    extended_epochs = config['extended_training']['epochs']
    layers_to_freeze = config.get("extended_training", {}).get("layers_to_freeze", [])

print(f"Extended training: {extended_training}")

# Initialize W&B if enabled
if config['wandb']['enable']:
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        name=config['wandb']['run_name'],
        group=config['wandb'].get('group', "default"),  # Default to "default" if group is not provided
        config={
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "model": config['training']['model'],
            "scheduler_type": scheduler_type,
            "final_skip_values": final_skip_values,
            "extended_training": extended_training,
            "extended_final_skip_values": extended_final_skip_values,
            "extended_scheduler_type": extended_scheduler_type ,
            "extended_epochs": extended_epochs ,
            "layers_to_freeze": layers_to_freeze ,

        }
    )

# Dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize the model
print(f"Initializing model: {config['training']['model']}...")

# Extract model parameters from the YAML file
model_name = config['training']['model']
num_classes = config['training']['num_classes']

if model_name == "resnet50":
    model = resnet50(num_classes=num_classes, scheduler_type=scheduler_type, total_epochs=epochs, final_skip_values=final_skip_values)

elif model_name == "resnet50_bn":
    model = resnet50bn(num_classes=num_classes, scheduler_type=scheduler_type, total_epochs=epochs, final_skip_values=final_skip_values, use_bn=True)

else:
    raise ValueError(f"Unsupported model: {model_name}")

model = model.to(device)

print(model)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")


# Optimizer and learning rate schedule
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_updates = len(train_dataloader) * epochs
warmup_updates = int(total_updates * 0.1)
lrs = torch.cat([
    torch.linspace(0, lr, warmup_updates),  # Linear warmup
    torch.linspace(lr, 0, total_updates - warmup_updates)  # Linear decay
])

# Training loop
train_losses, train_accuracies, test_accuracies = [], [], []
update = 0


print("Starting training...")
total_batches = len(train_dataloader) * epochs

for epoch in range(1, config['training']['epochs'] + 1):
    print(f"Starting epoch {epoch}...")



    # Update skip connection scaling
    if not update_per_batch:
        model.update_skip_scale(epoch, total_epochs=config['training']['epochs'])

    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    epoch_stage_gradients = {"stage1": [], "stage2": [], "stage3": [], "stage4": []}

    # Training loop for batches
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Update skip scalar per epoch 
        if update_per_batch:
            global_step = batch_idx + (epoch - 1) * len(train_dataloader)
            model.update_skip_scale(global_step, total_batches)

        # Update learning rate if needed
        for param_group in optimizer.param_groups:
            param_group["lr"] = lrs[batch_idx + (epoch - 1) * len(train_dataloader)]

        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # To compute grandient norms per stage
        stage_grad_norms = compute_stage_grad_norms(model)
        for stage in epoch_stage_gradients:
            epoch_stage_gradients[stage].append(stage_grad_norms[stage])

        optimizer.step()

        # Accumulate training metrics
        running_loss += loss.item() * inputs.size(0)
        total_samples += targets.size(0)
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()

        

        # Print batch-level metrics
        batch_loss = loss.item()
        if batch_idx % 10 == 0:  # Adjust batch print frequency
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {batch_loss:.4f}")


    skip_scales = model.get_skip_scales()
    print("Current Skip Scales Per Stage:")
    for stage, scale in skip_scales.items():
        print(f"{stage}: {scale:.4f}")
    

    # Calculate epoch-level metrics
    epoch_loss = running_loss / total_samples
    epoch_acc = 100. * total_correct / total_samples
    avg_stage_grad_norms = {stage: sum(epoch_stage_gradients[stage]) / len(epoch_stage_gradients[stage]) for stage in epoch_stage_gradients}


    # Validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            val_correct += outputs.argmax(dim=1).eq(targets).sum().item()

    val_loss /= len(test_dataset)
    val_acc = 100. * val_correct / len(test_dataset)

    # Log epoch-level metrics to WandB
    if config['wandb']['enable']:
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc, 
            # "skip_scale": model.layer1[0].skip_scale,
            **avg_stage_grad_norms,
            **{f"skip_scale/{stage}": scale for stage, scale in model.get_skip_scales().items()}
            
        })

    # Print epoch-level metrics
    print(f"Epoch {epoch} completed: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

print("Training complete!")
print(f"Final Test Accuracy: {val_acc:.2f}%")

# Specify the path to save the model weights
save_path = f"{config['wandb']['run_name']}.pth"

# Save the model weights
if config.get("training", {}).get("save_weights", False):
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

# Extended training phase

def count_frozen_parameters(model):
    """Prints the number of frozen parameters in the model."""
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}")
    print(f"Trainable Parameters: {total_params - frozen_params:,}")

if extended_training:
    print("Starting extended training phase...")

    # Capture last skip_scale values before extended training
    extended_start_values = list(model.get_skip_scales().values())

    model.total_epochs += extended_epochs

    print(f'first extended skip values: {extended_start_values}')

    total_epochs = config["training"]["epochs"] + extended_epochs

    for layer_name in layers_to_freeze: 
        for param in getattr(model, layer_name).parameters():
            param.requires_grad = False

    count_frozen_parameters(model)

    # model.final_skip_values = extended_final_skip_values


    print(f'Final skip values: {model.final_skip_values}')

    for epoch in range(config["training"]["epochs"] + 1, total_epochs + 1):
        print(f"Starting extended epoch {epoch}...")

        model.update_skip_scale(epoch - config["training"]["epochs"], extended_epochs, extended_start_values, extended_final_skip_values)
        skip_scales = model.get_skip_scales()

        print("Current Skip Scales Per Stage:")
        for stage, scale in skip_scales.items():
            print(f"{stage}: {scale:.4f}")

        model.train()
        running_loss, total_correct, total_samples = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total_samples += targets.size(0)
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = 100. * total_correct / total_samples

        # Validation loop
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        val_loss /= len(test_dataset)
        val_acc = 100. * val_correct / len(test_dataset)

        # Log metrics
        if config['wandb']['enable']:
            wandb.log({
                "epoch": epoch,
                "extended_train_loss": epoch_loss,
                "extended_train_accuracy": epoch_acc,
                "extended_val_loss": val_loss,
                "extended_val_accuracy": val_acc,
                **{f"extended_skip_scale/{stage}": scale for stage, scale in model.get_skip_scales().items()}
            })
            
        print(f"Extended Epoch {epoch} completed: Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

print("Training complete!")
