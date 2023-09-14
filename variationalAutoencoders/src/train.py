import scipy.io
import torch
import torch.optim as optim
import torch.nn as nn
import vaemodel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.utils import save_image
import argparse
import os
import shutil
# learning parameters
batch_size = 64
lr = 0.001
epochs = 20
# constructing the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=20, type=int, 
                    help='number of epochs to train the VAE for')
args = vars(parser.parse_args())
epochs = args['epochs']
parser.add_argument('-l', '--learning rate', default=0.0001, type=float, 
                    help='Learning Rate')
args = vars(parser.parse_args())
lr = args['learning rate']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# getting the data into NumPy format
mat_data = scipy.io.loadmat('../Dataset/frey_rawface.mat')
path = "../outputImages"
if not os.path.exists(path):
                   os.mkdir(path)
else:
                               
                 shutil.rmtree(path) 
                 os.makedirs(path)
data = mat_data['ff'].T.reshape(-1, 1, 28, 20)
data = data.astype('float32') / 255.0
print(f"Number of instances: {len(data)}")
# dividing the data into train and validation set
x_train = data[:-300]
x_val = data[-300:]
print(f"Training instances: {len(x_train)}")
print(f"Validation instances: {len(x_val)}")

# preparing the torch Dataset
class FreyFaceDataset(Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return (len(self.X))
    def __getitem__(self, index):
        data = self.X[index]
        return torch.tensor(data, dtype=torch.float)

train_data = FreyFaceDataset(x_train)
val_data = FreyFaceDataset(x_val)
# iterable data loader
train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)

vaemodel = vaemodel.Conv_VAE().to(device)
optimizer = optim.Adam(vaemodel.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

def final_loss(recontruction_loss, mu, logvar):
    BCE = recontruction_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
def fit(model, dataloader):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data= data
        data = data.to(device)
        data = data
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        recontruction_loss = criterion(reconstruction, data)
        loss = final_loss(recontruction_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss

def validate(model, dataloader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data= data
            data = data.to(device)
            data = data
            reconstruction, mu, logvar = model(data)
            recontruction_loss = criterion(reconstruction, data)
            loss = final_loss(recontruction_loss, mu, logvar)
            running_loss += loss.item()            
        
            # input and output of every epoch
            if i == int(len(val_data)/dataloader.batch_size) - 1:
                num_rows = 10
                both = torch.cat((data[:10], 
                                  reconstruction[:10]))
                
                save_image(both.cpu(), f"../outputImages/output_image{epoch}.png", nrow=num_rows)
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss

train_loss = []
val_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(vaemodel, train_loader)
    val_epoch_loss = validate(vaemodel, val_loader)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")