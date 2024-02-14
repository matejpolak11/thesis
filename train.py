from models.unet import UNet

import os
import time
import random
import pickle

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.io

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

mask_to_rgb_dict = {
    0: [0, 0, 0],  # frame
    1: [0, 0, 255],  # water
    2: [255, 0, 255],  # blocks
    3: [0, 255, 255],  # non-built
    4: [255, 255, 255]  # road network
}


class Runner:
    def __init__(self,
        model_name: str,
        n_epochs: int,
        learning_rate: float,
        batch_size: int,
        use_augmentation: bool,
        leaky_relu: bool):

        self.model = None
        if model_name == "unet":
            self.model = UNet(len(mask_to_rgb_dict))
        elif model_name == "transunet":
            pass

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation

    def fetch_data(self):
        self.train_DS = SegmentationDataset(split_name="train", use_augmentation=self.use_augmentation)
        self.val_DS = SegmentationDataset(split_name="val", use_augmentation=self.use_augmentation)

        if torch.cuda.is_available():
            device = torch.device('cuda')
            pin_mem = True
        else:
            device = torch.device('cpu')
            pin_mem = False

        self.model = self.model.to(device)

        # create the training and test data loaders
        self.train_loader = DataLoader(self.train_DS,
                                  shuffle=True,
                                  batch_size=self.batch_size,
                                  pin_memory=pin_mem)
        self.val_loader = DataLoader(self.val_DS,
                                shuffle=True,
                                batch_size=self.batch_size,
                                pin_memory=pin_mem)

    def training_loop(self):
        start_time = time.time()
        loss = {"train": [], "val": []}

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        opt = Adam(self.model.parameters(), lr=self.learning_rate)
        loss_func = nn.CrossEntropyLoss()

        train_steps = len(self.train_DS) // self.batch_size
        val_steps = len(self.val_DS) // self.batch_size

        for epoch in range(self.n_epochs):
            print('Starting Epoch # ' + str(epoch))
            self.model.train()
            self.model.to(device)

            total_train_loss = 0
            total_val_loss = 0

            for (i, (x, y)) in enumerate(self.train_loader):
                # print('Train batch # ' + str(i))
                # send the input to the device
                # print (x.shape)
                # print (y.shape)
                (x, y) = (x.to(device), y.to(device))
                # print (x.shape)
                # print (y.shape)
                # perform a forward pass and calculate the training loss
                pred = self.model(x)
                loss = loss_func(pred, y)
                # first, zero out any previously accumulated gradients, then
                # perform backpropagation, and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                total_train_loss += loss
            # switch off autograd
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for (x, y) in self.val_loader:
                    # send the input to the device
                    (x, y) = (x.to(device), y.to(device))
                    # make the predictions and calculate the validation loss
                    pred = self.model(x)
                    total_val_loss += loss_func(pred, y)
            # calculate the average training and validation loss
            avg_train_loss = total_train_loss / train_steps
            avg_val_loss = total_val_loss / val_steps
            # update our training history
            loss["train"].append(avg_train_loss.cpu().detach().numpy())
            loss["val"].append(avg_val_loss.cpu().detach().numpy())
            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(epoch + 1, self.n_epochs))
            print("Train loss: {:.4f}, Validation loss: {:.4f}".format(
                avg_train_loss, avg_val_loss))

            #(a, b) = train_DS[1]
            #a = a.to(device)
            #b = b.to(device)
            #a = a.unsqueeze(0)
            #p = model(a)
            #print(a)
            #print(b)
            #print(torch.unique(p))
            #print(torch.unique(torch.argmax(p, dim=1), return_counts=True))
            #if e == 10 or e == 25 or e == 40 or e == 50 or e == 60 or e == 75 or e == 100 or e == 150 or e == 200:
            #    torch.save(model, "UNet_augm_" + str(e))

            # torch.save(model, "unet")
        # display the total time needed to perform the training
        end_time = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            end_time - start_time))

        with open('H', 'wb') as f:
            pickle.dump(loss, f)


class SegmentationDataset(Dataset):
    def __init__(self, split_name, use_augmentation):

        self.img_paths = []
        self.label_paths = []

        for dataset_name in ["paris", "world"]:
            if use_augmentation:
                self.img_paths.extend(self.list_dir(os.path.join("data", "augmented", dataset_name, split_name, "img")))
                self.label_paths.extend(self.list_dir(os.path.join("data", "augmented", dataset_name, split_name, "label")))
            else:
                self.img_paths.extend(self.list_dir(os.path.join("data", "img", dataset_name, split_name, "img")))
                self.label_paths.extend(self.list_dir(os.path.join("data", "img", dataset_name, split_name, "label")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(self.img_paths[idx])
        img = img / 255

        label = torchvision.io.read_image(self.label_paths[idx])
        label = label.numpy()
        mask = np.zeros(label.shape[1:])
        for cat in mask_to_rgb_dict.keys():
            color = np.array(mask_to_rgb_dict[cat]).reshape(3, 1, 1)
            mask[np.all(label == color, axis=0)] = cat
        mask = torch.from_numpy(mask)

        #mean = image.mean([1, 2])
        #std = image.std([1, 2])
        #norm = torchvision.transforms.Normalize(mean, std)
        #image = norm(image)

        return img, mask

    def list_dir(self, dir: str):
        dir_path = os.path.abspath(dir)
        res = []
        for file_name in os.listdir(dir):
            res.append(os.path.join(dir_path, file_name))
        return res

if __name__ == "__main__":
    seed = 343

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    runner = Runner("unet", 200, 0.001, 4, False, False)
    runner.fetch_data()
    runner.train_DS.__getitem__(4)
    runner.training_loop()