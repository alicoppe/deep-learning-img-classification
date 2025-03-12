"""
Basic Usage:
python main.py --model <model_name> --model_config <path_to_json> --logdir <result_dir> ...
Please see config.py for other command line usage.
"""
import warnings

import torch
from torch import optim
from torchvision.datasets import CIFAR10
from torchvision import transforms
from utils import seed_experiment, to_device, cross_entropy_loss, compute_accuracy
from config import get_config_parser
import json
from mlp import MLP
from resnet18 import ResNet18
from mlpmixer import MLPMixer
from tqdm import tqdm
from torch.utils.data import DataLoader
import time
import os
import datetime

def train(epoch, model, dataloader, optimizer, args):
    model.train()
    epoch_accuracy = 0
    epoch_loss = 0
    start_time = time.time()

    with tqdm(total=len(dataloader), desc=f"Epoch {epoch}", unit="batch") as pbar:
        for idx, batch in enumerate(dataloader):
            pbar.update(1)
            batch = to_device(batch, args.device)
            optimizer.zero_grad()
            imgs, labels = batch
            logits = model(imgs)
            loss = cross_entropy_loss(logits, labels)
            acc = compute_accuracy(logits, labels)

            loss.backward()
            optimizer.step()
            epoch_accuracy += acc.item() / len(dataloader)
            epoch_loss += loss.item() / len(dataloader)

            pbar.set_postfix(loss=loss.item(), acc=acc.item())

    pbar.close()
    print(f"== [TRAIN] Epoch: {epoch}, Accuracy: {epoch_accuracy:.3f}, Training Time: {time.time() - start_time:.3f} ==")

    return epoch_loss, epoch_accuracy, time.time() - start_time



def evaluate(epoch, model, dataloader, args, mode="val"):
    model.eval()
    epoch_accuracy=0
    epoch_loss=0
    total_iters = 0
    start_time = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, args.device)
            imgs, labels = batch
            logits = model(imgs)
            loss = cross_entropy_loss(logits, labels)
            acc = compute_accuracy(logits, labels)
            epoch_accuracy += acc.item() / len(dataloader)
            epoch_loss += loss.item() / len(dataloader)
            total_iters += 1
            if idx % args.print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}"
                )
        tqdm.write(
            f"=== [{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Accuracy: {epoch_accuracy:.3f} ===>"
        )
    return epoch_loss, epoch_accuracy, time.time() - start_time

        
if __name__ == "__main__":
    parser = get_config_parser()
    args = parser.parse_args()

    # Check for the device
    if args.device == "cuda":
        if not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Falling back to 'mps' if available, otherwise 'cpu'.")
            args.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            print("Using CUDA")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            warnings.warn("MPS is not available on this device. Falling back to 'cpu'.")
            args.device = "cpu"
        else:
            print("Using MPS")

    if args.device == "cpu":
        warnings.warn("Running on CPU. You may run out of memory; consider using a smaller batch size.")

    # Seed the experiment, for repeatability
    seed_experiment(args.seed)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
    ])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
    ])

    # Load datasets
    train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root='./data', train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    test_set = CIFAR10(root='./data', train=False, transform=test_transform, download=True)

    # Load model
    print(f'Build model {args.model.upper()}...')
    if args.model_config is not None:
        print(f'Loading model config from {args.model_config}')
        with open(args.model_config) as f:
            model_config = json.load(f)
    else:
        raise ValueError('Please provide a model config json')

    model_cls = {'mlp': MLP, 'resnet18': ResNet18, 'mlpmixer': MLPMixer}[args.model]
    model = model_cls(**model_config)
    model.to(args.device)

    # Optimizer
    optimizer = {
        "adamw": optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        "adam": optim.Adam(model.parameters(), lr=args.lr),
        "sgd": optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
        "momentum": optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    }[args.optimizer]

    print(
        f"Initialized {args.model.upper()} model with {sum(p.numel() for p in model.parameters())} "
        f"total parameters, of which {sum(p.numel() for p in model.parameters() if p.requires_grad)} are learnable."
        f'\n Using {args.optimizer} optimizer with learning rate {args.lr}.'
    )

    # DataLoaders
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Training Loop
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    
    # Load best model validation accuracies if the file exists
    best_valid_accs = None
    if os.path.exists(args.best_model_path):
        print(f'\n A best model path was specified at {args.best_model_path}, using this for early stopping criterion...\n')
        with open(args.best_model_path, "r") as f:
            best_model_data = json.load(f)
        best_valid_accs = best_model_data.get("valid_accs", None)

    patience = args.patience
    threshold = args.threshold
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(args.epochs):
        tqdm.write(f"====== Epoch {epoch} ======>")

        # Train and validate
        loss, acc, _ = train(epoch, model, train_dataloader, optimizer, args)
        train_losses.append(loss)
        train_accs.append(acc)

        valid_loss, valid_acc, _ = evaluate(epoch, model, valid_dataloader, args)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Determine the best accuracy for this epoch from the previous best model
        if best_valid_accs is not None and epoch < len(best_valid_accs):
            best_valid_acc = best_valid_accs[epoch]
        else:
            best_valid_acc = 0  # Default if no previous best exists

        if best_valid_acc > 0:
            # Check if the validation accuracy drops significantly
            if valid_acc <= best_valid_acc - threshold:
                epochs_no_improve += 1
                tqdm.write(f'Validation accuracy of {valid_acc} at epoch {epoch} is less than the best validation {best_valid_acc} - threshold of {threshold}')
                tqdm.write(f"No improvement for {epochs_no_improve} epoch(s).")
                
            else:
                tqdm.write(f'Validation accuracy is within the threshold of the best validation accuracy at epoch {epoch} of {best_valid_acc}')

        # Early stopping condition
        if epochs_no_improve >= patience:
            early_stop = True
            tqdm.write(f"Early stopping triggered after {epoch + 1} epochs due to lack of improvement.")
            break  # Stop training

    test_loss, test_acc, _ = evaluate(epoch, model, test_dataloader, args, mode="test")
    print(f"===== Best validation Accuracy: {max(valid_accs):.3f} =====>")

    # Save results
    if args.logdir is not None and not early_stop:
        print(f'Writing training logs to {args.logdir}...')
        os.makedirs(args.logdir, exist_ok=True)

        # Set filename based on argument or default
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{args.filename}_results.json' if args.filename else f'results_{args.model}_{timestamp}.json'

        with open(os.path.join(args.logdir, filename), 'w') as f:
            json.dump({
                "train_losses": train_losses,
                "valid_losses": valid_losses,
                "train_accs": train_accs,
                "valid_accs": valid_accs,
                "test_loss": test_loss,
                "test_acc": test_acc
            }, f, indent=4)

        print(f'Saved results to {filename}')
        
        # Visualize
        if args.visualize and args.model in ['resnet18', 'mlpmixer']:
            model.visualize(args.logdir)
