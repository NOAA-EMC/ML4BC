'''
Description: This script is to train UNet model.
8/23/2024, Linlin Cui (linlin.cui@noaa.gov)
'''

import os
from datetime import datetime, timedelta
import time

from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler


from models.SmaAt_UNet import SmaAt_UNet
from utils.dataset import NetCDFDataset

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def get_random_subset_dataloader(dataset, subset_fraction=0.9, batch_size=8, seed=42):

    torch.manual_seed(seed)

    num_samples = len(dataset)
    indices = torch.randperm(num_samples)
    subset_size = int(num_samples * subset_fraction)

    train_indices = indices[:subset_size]
    sampler = SubsetRandomSampler(train_indices)


    train_dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,  
        num_workers=0,
        pin_memory=True,
    )

    valid_indices = indices[subset_size:]
    sampler = SubsetRandomSampler(valid_indices)
    valid_dl = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,  
        num_workers=0,
        pin_memory=True,
    )

    return train_dl, valid_dl

def trainer(
    epochs,
    batch_size,
    seed,
    train_percent,
    model,
    loss_func,
    opt,
    dataset,
    device,
    save_every,
    tensorboard: bool = False,
    earlystopping=None,
    lr_scheduler=None,
):

    writer =None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(comment=f"{model.__class__.__name__}")

    start_time = time.time()

    best_valid_loss = 1.e6
    earlystopping_counter = 0
    for epoch in tqdm(range(epochs), desc='Epochs', leave=True):

        train_dl, valid_dl = get_random_subset_dataloader(dataset, train_percent, batch_size, seed)

        model.train()
        train_loss = 0.0
        for i, (xb, yb) in enumerate(tqdm(train_dl, desc='Batches', leave=False)):
            loss = loss_func(model(xb.to(device)), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        #Cal validation loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in tqdm(valid_dl, desc='validation', leave=False):
                y_pred = model(xb.to(device))
                loss = loss_func(y_pred, yb.to(device))
                val_loss += loss.item()
                #pred_class = torch.argmax(nn.functional.softmax(y_pred, dim=1), dim=1)
                #iou_metric.add(pred_class, target=yb)

            #iou_class, mean_iou = iou_metric.value()
            val_loss /= len(valid_dl)

        # Save the model with the best mean IOU
        if val_loss < best_valid_loss: 
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    'model': model,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                },
                f"checkpoints/best_model_{model.__class__.__name__}.pt",
            )
            best_valid_loss = val_loss
            earlystopping_counter = 0

        else:
            if earlystopping is not None:
                earlystopping_counter += 1
                if earlystopping_counter >= earlystopping:
                    print(f"Stopping early --> valid_loss has not decreased over {earlystopping} epochs")
                    break

        print(
            f"Epoch: {epoch:5d}, Time: {(time.time() - start_time) / 60:.3f} min,"
            f"Train_loss: {train_loss:2.10f}, Val_loss: {val_loss:2.10f},",
            f"lr: {get_lr(opt)},",
            f"Early stopping counter: {earlystopping_counter}/{earlystopping}" if earlystopping is not None else "",
        )

        if writer:
            # add to tensorboard
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Parameters/learning_rate", get_lr(opt), epoch)
        if save_every is not None:
            if epoch % save_every == 0:
                # save model
                torch.save(
                    {
                        "model": model,
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict(),
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                    },
                    f"checkpoints/model_{model.__class__.__name__}_epoch_{epoch}.pt",
                )
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)


if __name__ == "__main__":

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001
    epochs = 50
    earlystopping = 30
    save_every = 1
    batch_size = 32 
    train_percent = 0.9
    seed = 42

    startdate = datetime(2021, 3, 23)
    enddate = datetime(2024, , 1)
    #print(startdate)
    #print(enddate)

    data_dir = '/scratch1/NCEPDEV/nems/Linlin.Cui/Tests/ML4BC/run27/data'
    bbox = [230, 300, 25, 50]
    dataset = NetCDFDataset(data_dir, startdate, enddate, bbox)

    model = SmaAt_UNet(n_channels=1, n_classes=1)
    if torch.cuda.device_count() > 1:
        nn.DataParallel(model).to(device)
    else:
        model.to(device)

    opt = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss().to(device)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=4)

    trainer(
        epochs = epochs,
        batch_size = batch_size,
        seed = seed,
        train_percent = train_percent,
        model = model,
        loss_func = loss_fn,
        opt = opt,
        dataset = dataset,
        device = device,
        save_every = save_every,
        tensorboard = False,
        earlystopping = earlystopping,
        lr_scheduler = lr_scheduler,
    )
