import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

import pandas
from matplotlib import pyplot as plt

from matplotlib.ticker import FormatStrFormatter

torch.manual_seed(626)

data = pandas.read_csv('./gan_photoz/stripe82_galaxies.csv')

y = torch.tensor(data['z'].values, dtype=torch.float32).reshape(-1, 1)
X = torch.tensor(data[['MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z']].values, dtype=torch.float32)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

Z_DIM = 20
D_DIM = 32
G_DIM = 32
lr = 0.0001
gamma = 0.2
step = 2000
EPOCHS = 10000

# Define the neural network
class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(4 + Z_DIM, G_DIM)
        self.fc2 = nn.Linear(G_DIM, G_DIM)
        self.fc3 = nn.Linear(G_DIM, 1)
        self.bn1 = nn.BatchNorm1d(G_DIM)
        self.bn2 = nn.BatchNorm1d(G_DIM)

    def forward(self, x,z):
        x = torch.cat((x,z),1)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(4 + 1, D_DIM)
        self.fc2 = nn.Linear(D_DIM, D_DIM)
        self.fc3 = nn.Linear(D_DIM, 1)
        self.bn1 = nn.BatchNorm1d(D_DIM)
        self.bn2 = nn.BatchNorm1d(D_DIM)

    def forward(self, x,y):
        x = torch.cat((x,y),1)
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
    
class LossKLDiscriminator(nn.Module):

    def __init__(self):
        super(LossKLDiscriminator, self).__init__()

    def forward(self, real_preds, fake_preds):
        return -(real_preds - torch.exp(fake_preds - 1)).mean()+1.
    
class LossKLGenerator(nn.Module):

    def __init__(self):
        super(LossKLGenerator, self).__init__()

    def forward(self, fake_preds):
        return -fake_preds.mean() + 1.

# Initialize the model, loss function, and optimizer
model = FeedForwardNN()
criterion = LossKLGenerator()
optimizer = optim.Adam(model.parameters(), lr=lr)

discriminator = Discriminator()
criterion_d = LossKLDiscriminator()
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

steplr_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=step, gamma=gamma)
steplr = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

# Train the model
num_epochs = EPOCHS
losses = []
losses_d = []
losses_test = []
losses_test_d = []
mse = []
mse_test = []

for epoch in range(num_epochs):

    discriminator.train()
    optimizer_d.zero_grad()
    real_preds = discriminator(X_train, y_train)
    fake_preds = discriminator(X_train, model(X_train, torch.rand(X_train.shape[0], Z_DIM)))
    loss_d = criterion_d(real_preds, fake_preds)
    loss_d.backward()
    optimizer_d.step()


    z = torch.rand(X_train.shape[0], Z_DIM)

    model.train()
    optimizer.zero_grad()
    outputs = model(X_train, z)
    loss = criterion(discriminator(X_train, outputs))
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        z = torch.rand(X_test.shape[0], Z_DIM)
        predictions = model(X_test, z)
        test_loss_g = criterion(discriminator(X_test, y_test))

        real_preds = discriminator(X_test, y_test)
        fake_preds = discriminator(X_test, model(X_test, torch.rand(X_test.shape[0], Z_DIM)))
        test_loss_d = criterion_d(real_preds, fake_preds)

        mse.append(torch.mean((y_test - predictions)**2))
        mse_test.append(torch.mean((y_test - predictions)**2))


    losses_test.append(test_loss_g.item())
    losses_test_d.append(test_loss_d.item())


    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    losses.append(loss.item())
    losses_d.append(loss_d.item())

    steplr.step()
    steplr_d.step()

torch.save(model, 'CGAN_photoz_KL_divergence.pth')

training_epochs = pandas.DataFrame({'losses': losses, 'losses_d': losses_d, 'losses_test': losses_test, 'losses_test_d': losses_test_d, 'mse': mse, 'mse_test': mse_test})
training_epochs.to_csv('training_losses.csv')

# Evaluate the model
model.eval()
with torch.no_grad():
    z = torch.rand(X_test.shape[0], Z_DIM)
    predictions = model(X_test, z)
    test_loss = criterion(predictions)
    print(f'Test Loss: {test_loss.item():.4f}')

    import matplotlib.pyplot as plt

    # Inverse transform the predictions and actual values to original scale
    predictions = scaler_y.inverse_transform(predictions.numpy())
    y_test = scaler_y.inverse_transform(y_test.numpy())

    # Plot the predictions vs the actual values

    svm = RandomForestRegressor()
    svm.fit(X_train, y_train)
    y_svm = svm.predict(X_test)
    y_svm = scaler_y.inverse_transform(y_svm.reshape(-1, 1))
    plt.scatter(y_test, predictions, color='blue', alpha=0.5, label='CGAN')
    plt.scatter(y_test, y_svm, color='red', alpha=0.5, label='Random Forest')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Spectroscopic redshift')
    plt.ylabel('Photometric redshift')
    plt.legend()
    plt.savefig('scatter.png')

    plt.clf()
    plt.hist2d(y_test.flatten(), predictions.flatten(),bins=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'w--', lw=1)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Observed')
    plt.legend()
    plt.savefig('hist2d.png')

    counts1,ybins1,xbins1,image1 = plt.hist2d(y_test.flatten(), y_svm.flatten(),bins=50)
    counts2,ybins2,xbins2,image2 = plt.hist2d(y_test.flatten(), predictions.flatten(),bins=50)
    plt.clf()
    plt.contour(counts1, extent=[xbins1.min(), xbins1.max(), ybins1.min(), ybins1.max()], linewidths=1, label='SVM', colors='red')
    plt.contour(counts2, extent=[xbins2.min(), xbins2.max(), ybins2.min(), ybins2.max()], linewidths=1, label='GAN', colors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'w--', lw=1)
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Observed')
    plt.legend()
    plt.savefig('contour.png')

    plt.clf()
    plt.plot(y_test, label='Real')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.savefig('line.png')

    counts, xbins, image = plt.hist(y_test, label='Real spectroscopic redshift', bins=50, alpha=0.5, density=False)
    plt.clf()
    plt.hist(predictions, label='CGAN photometric redshift', bins=50, range=(-0.1, 0.8), alpha=0.5, density=False)  
    plt.hist(y_svm, label='Random Forest photometric redshift', bins=50, range=(-0.1, 0.8), alpha=0.5, density=False)
    plt.plot(xbins[:-1], counts, label='Real Spectroscopic', range=(-0.1, 0.8), color='black')
    plt.xlabel('Redshift')
    plt.ylabel('Number of galaxies')
    plt.legend()
    plt.savefig('hist.png')

    plt.clf()
    fig, ax = plt.subplots(3)
    fig.subplots_adjust(hspace=0)
    ax[0].plot(losses, color='blue', label='Train Generator')
    ax[0].plot(losses_test, color='red', linestyle='--',label='Test Generator')
    ax[1].plot(losses_d, color='blue',label='Train Discriminator')
    ax[1].plot(losses_test_d, color='red', linestyle='--',label='Test Discriminator')
    ax[2].plot(mse, color='blue',label='Train MSE')
    ax[2].plot(mse_test, color='red', linestyle='--',label='Test MSE')
    #ax[0].set_yscale('log')
    #ax[1].set_yscale('log')
    #ax[2].set_yscale('log')
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')
    ax[2].set_ylabel('MSE')
    ax[1].set_xlabel('Training Epoch')
    plt.savefig('losses.png')



    plt.clf()
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(6, 6))
    for i in range(9):
        x_display = X_test[i:i+1].repeat((1000, 1)) 
        z = torch.rand(1000, Z_DIM)
        y_display = model(x_display, z)
        x_display = scaler_X.inverse_transform(x_display.numpy())
        y_display = scaler_y.inverse_transform(y_display.numpy())
        ax[i].hist(y_display, bins=50, density=True, alpha=0.5, color='blue', label='Probability density')
        ax[i].axvline(x=y_test[i], color='r', linestyle='-', label='Real redshift')

    fig.legend(loc='upper right') 
    fig.text(0.5, 0.04, 'Spectroscopic redshift', ha='center')
    fig.text(0.04, 0.5, 'Probability density', va='center', rotation='vertical')
    plt.savefig('hist_generated.png')