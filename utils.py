import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def plot_results(logdir, filenames, plotnames=None, save_path=None, title=None):
    """
    Creates a 2x2 grid of subplots for train accuracy, validation accuracy,
    train loss, and validation loss. Each subplot will display data from all
    specified filenames, with a legend identifying each.

    Args:
        logdir (str): Directory where the result JSON files are located.
        filenames (list[str]): List of filenames (JSON files) to plot.
        plotnames (list[str], optional): Names to display in the legend for each
                                         file. Must be same length as filenames.
                                         If None, filenames (minus "_results.json")
                                         will be used in the legend.
        save_path (str, optional): Path to save the generated plot as a PNG file.
                                   If None, the plot is not saved.
        title (str, optional): Title for the entire plot. If None, no title is added.

    Raises:
        ValueError: If filenames are not provided or plotnames length doesn't
                    match filenames length.
    """
    if not filenames:
        raise ValueError("Filenames must be specified (non-empty list).")

    if plotnames is not None and len(plotnames) != len(filenames):
        raise ValueError("Number of plotnames must match number of filenames.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey='row')
    ax_train_acc, ax_val_acc = axes[0]
    ax_train_loss, ax_val_loss = axes[1]

    for i, fname in enumerate(filenames):
        path = os.path.join(logdir, fname)
        with open(path, 'r') as f:
            data = json.load(f)

        label = plotnames[i] if plotnames is not None else fname.replace("_results.json", "")
        
        epochs_acc = range(1, len(data["train_accs"])+1)
        epochs_loss = range(1, len(data["train_losses"])+1)

        ax_train_acc.plot(epochs_acc, data["train_accs"], label=label)
        ax_val_acc.plot(epochs_acc, data["valid_accs"], label=label)
        ax_train_loss.plot(epochs_loss, data["train_losses"], label=label)
        ax_val_loss.plot(epochs_loss, data["valid_losses"], label=label)
    
    # Set titles for subplots
    ax_train_acc.set_title("Train Accuracy")
    ax_val_acc.set_title("Validation Accuracy")
    ax_train_loss.set_title("Train Loss")
    ax_val_loss.set_title("Validation Loss")
    
    # Enable y-axis labels
    for ax in [ax_train_acc, ax_val_acc, ax_train_loss, ax_val_loss]:
        ax.yaxis.set_tick_params(labelsize=10)
    
    # Add legends to subplots
    ax_train_acc.legend()
    ax_val_acc.legend()
    ax_train_loss.legend()
    ax_val_loss.legend()
    
    # Add an overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    
    # Save plot if save_path is provided
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}.png")
    
    plt.show()



def generate_plots(list_of_dirs, legend_names, save_path):
    """ Generate plots according to log 
    :param list_of_dirs: List of paths to log directories
    :param legend_names: List of legend names
    :param save_path: Path to save the figs
    """
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have same length"
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(os.path.join(logdir, 'results.json')), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[name] = json.load(f)
    
    for yaxis in ['train_accs', 'valid_accs', 'train_losses', 'valid_losses']:
        fig, ax = plt.subplots()
        for name in data:
            ax.plot(data[name][yaxis], label=name)
        ax.legend()
        ax.set_xlabel('epochs')
        ax.set_ylabel(yaxis.replace('_', ' '))
        fig.savefig(os.path.join(save_path, f'{yaxis}.png'))
        

def seed_experiment(seed):
    """Seed the pseudorandom number generator, for repeatability.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    elif isinstance(tensors, list):
        return list(
            (to_device(tensors[0], device), to_device(tensors[1], device)))
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor):
    """ Return the mean loss for this batch
    :param logits: [batch_size, num_class]
    :param labels: [batch_size]
    :return loss 
    """

    exp_logits = torch.exp(logits)
    sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
    softmax_probs = exp_logits / sum_exp_logits
    
    selected_probs = softmax_probs[torch.arange(len(labels)), labels]
    loss = -torch.mean(torch.log(selected_probs + 1e-9))  # Adding epsilon to prevent log(0)
    
    return loss

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor):
    """ Compute the accuracy of the batch """
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return acc
