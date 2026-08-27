import csv
from utils import gen_random_data
from model import AutoNet
from dataset import ImgRadarDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def main():
    args = parse_arg()
    print("========== Train Config =========")
    for key, val in vars(args).items():
        print(f"{key}: {val}")
    print("=================================")
    if args.test:
        args.epoches = 1
        args.data_dir = "./test/data"
        args.save_dir = "./test/train"
        gen_random_data()
        train(args)
    else:
        train_losses, val_losses = train(args)
        plot_fig(train_losses, val_losses, args.save_dir)

def train(args):
    batch_size = args.batch
    root_dir = args.data_dir
    epoches = args.epoches
    eval_step = args.eval_step
    num_workers = args.num_workers
    lr = args.lr
    radar_length = args.radar_length
    save_dir = args.save_dir
    save_step = args.save_step
    device = args.device
    resume = args.resume
    train_ratio = args.ratio
    val_ratio = 1 - train_ratio
    weight_decay = args.weight_decay
    gamma = args.gamma
    step_size = args.reduce_step

    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    total_dataset = ImgRadarDataset(root_dir)
    train_set, val_set = random_split(total_dataset, [train_ratio, val_ratio])
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    model = AutoNet(radar_length)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)

    best_loss = float("inf")
    train_losses = []
    val_losses = []

    print("Start Train:")
    print(f"Train Data Len = {len(train_set)}, Val Data Len = {len(val_set)}")

    if resume is not None:
        if os.path.exists(resume):
            print(f"Resume training from {resume}")
            model.load_state_dict(torch.load(resume))
        else:
            print("Resume file not exist, train from empty weight")

    csv_path = os.path.join(save_dir, "training_log.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])

    model.to(device)
    for epoch in range(1, epoches + 1):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epoches}")

        # train loop
        for img, radar, label in loop:
            img = img.to(device)
            radar = radar.to(device)
            label = label.to(device)

            output = model((img, radar))
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * img.size(0)
            loop.set_postfix({
                'Train Loss': f"{train_loss:.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        train_loss /= len(train_set)
        train_losses.append(train_loss)
        scheduler.step()

        # eval loop
        if eval_step is None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img, radar, label in val_loader:
                    img = img.to(device)
                    radar = radar.to(device)
                    label = label.to(device)

                    output = model((img, radar))
                    loss = loss_func(output, label)
                    val_loss += loss.item() * img.size(0)

                val_loss /= len(val_set)
                val_losses.append(val_loss)
                print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, \
                    LR = {scheduler.get_last_lr()[0]:.6f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                    print(f"\t-> Save best model with loss: {val_loss}")
        elif epoch % eval_step == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img, radar, label in val_loader:
                    img = img.to(device)
                    radar = radar.to(device)
                    label = label.to(device)

                    output = model((img, radar))
                    loss = loss_func(output, label)
                    val_loss += loss.item() * img.size(0)

                val_loss /= len(val_set)
                for _ in range(eval_step):
                    val_losses.append(val_loss)
                print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}, \
                    LR = {scheduler.get_last_lr()[0]:.6f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                    print(f"\t-> Save best model with loss: {val_loss}")


        if save_step is not None and epoch % save_step == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"checkpoints/{epoch:03d}.pt"))

        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_losses[-1] if val_losses else 'NaN', scheduler.get_last_lr()[0]])

    print(f"Training Finished! Best Val Loss: {best_loss}")
    torch.save(model.state_dict(), os.path.join(save_dir, "last.pt"))

    return train_losses, val_losses



def plot_fig(train_losses, val_losses, save_dir):
    fig = plt.figure(figsize=(10, 6))
    axes = fig.add_subplot(111)
    epoches = range(1, len(train_losses) + 1)
    axes.plot(epoches, train_losses, label="Train Loss", color="blue")
    axes.plot(epoches, val_losses, label="Validation Loss", color="orange")
    axes.set_xlabel("Epochs")
    axes.set_ylabel("Loss")
    axes.set_title("Training and Validation Loss")
    axes.legend()
    axes.grid()
    plt.savefig(os.path.join(save_dir, "loss_plot.png" ))


def parse_arg():
    parser = argparse.ArgumentParser("Trainning Script")

    parser.add_argument( "-d", "--data-dir", type=str, default="./data", help="Path to the dataset directory" )
    parser.add_argument( "--ratio", type=float, default=0.7, help="train and validation ration")
    parser.add_argument( "-b", "--batch", type=int, default=32, help="Batch size for training" )
    parser.add_argument( "-n", "--num-workers", type=int, default=16, help="Number of workers for data loading" )
    parser.add_argument( "-e", "--epoches", type=int, default=100, help="Number of epochs for training" )
    parser.add_argument( "--eval_step", type=int, default=None, help="Number of step to apply validation" )
    parser.add_argument( "--lr", type=float, default=1e-3, help="Learning rate for the optimizer" )
    parser.add_argument( "--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer" )
    parser.add_argument( "--reduce-step", type=int, default=10, help="Step to reduce lr for scheduler" )
    parser.add_argument( "--gamma", type=float, default=0.5, help="Reduce ratio for scheduler" )
    parser.add_argument( "--radar-length", type=int, default=360, help="Length of the radar input" )
    parser.add_argument( "-s", "--save-dir", type=str, default="./train", help="Directory to save model checkpoints" )
    parser.add_argument( "--save-step", type=int, default=None, help="Epoch step to save the checkpoint" )
    parser.add_argument( "--resume", type=str, default=None, help="Path to a checkpoint to resume training from" )
    parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (cuda or cpu)" )

    parser.add_argument( "--test", action="store_true", help="Run a test training with random data" )

    return parser.parse_args()

if __name__ == '__main__':
    main()
