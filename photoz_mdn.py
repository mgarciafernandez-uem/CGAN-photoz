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


class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(MDN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 22)
        self.fc2 = nn.Linear(22, num_gaussians * (2 * output_dim + 1))
        self.prelu = nn.PReLU()
        self.num_gaussians = num_gaussians
        self.output_dim = output_dim

    def forward(self, x):
        x = self.prelu(self.fc1(x))
        x = self.fc2(x)
        pi = nn.functional.softmax(x[:, :self.num_gaussians], dim=1)
        mu = x[:, self.num_gaussians:self.num_gaussians + self.num_gaussians * self.output_dim].view(-1, self.num_gaussians, self.output_dim)
        sigma = torch.exp(x[:, self.num_gaussians + self.num_gaussians * self.output_dim:]).view(-1, self.num_gaussians, self.output_dim)
        return pi, mu, sigma

def mdn_loss(pi, mu, sigma, y):
    m = torch.distributions.Normal(mu, sigma)

    loss = torch.exp(m.log_prob(y.unsqueeze(1).expand_as(mu)))

    loss = torch.sum(pi.unsqueeze(2) * loss, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
num_gaussians = 30

model = MDN(input_dim, output_dim, num_gaussians)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5000

loss_train_list = []
loss_test_list = []
mse_train_list = []
mse_test_list = []
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    pi, mu, sigma = model(X_train)
    loss = mdn_loss(pi, mu, sigma, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    loss_train_list.append(loss.item())

    # Compute the loss for the test dataset
    model.eval()
    with torch.no_grad():
        pi_test, mu_test, sigma_test = model(X_test)
        test_loss = mdn_loss(pi_test, mu_test, sigma_test, y_test)
    
    # Compute the MSE for both train and test datasets
    with torch.no_grad():
        y_train_pred = torch.sum(pi.unsqueeze(2) * mu, dim=1)
        y_test_pred = torch.sum(pi_test.unsqueeze(2) * mu_test, dim=1)
        
        train_mse = torch.mean((y_train_pred - y_train) ** 2).item()
        test_mse = torch.mean((y_test_pred - y_test) ** 2).item()
    
    loss_test_list.append(test_loss.item())
    mse_train_list.append(train_mse)
    mse_test_list.append(test_mse)
    

# Save the model weights
torch.save(model, 'MDN_photoz.pth')
# Save the losses and MSE to a CSV file
losses_df = pd.DataFrame({
    'Epoch': range(1, num_epochs + 1),
    'Training Loss': loss_train_list,
    'Test Loss': loss_test_list,
    'Training MSE': mse_train_list,
    'Test MSE': mse_test_list
})

losses_df.to_csv('training_losses_MDN.csv', index=False)

# Plotting the loss
plt.figure(figsize=(10, 5))
plt.plot(loss_train_list, label='Training Loss')
plt.plot(loss_test_list, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(mse_train_list, label='Training MSE')
plt.plot(mse_test_list, label='Test MSE')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
model.eval()
with torch.no_grad():
    pi, mu, sigma = model(X_test)
    y_pred = torch.sum(pi.unsqueeze(2) * mu, dim=1)

# Inverse transform the predictions and actual values
y_pred = scaler_y.inverse_transform(y_pred.numpy())
y_test = scaler_y.inverse_transform(y_test.numpy())

# Plot the predicted versus the actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.grid(True)
plt.show()

# Plot the probability density for a single record and compare with the actual value of that record
single_record_index = 0  # Index of the record to plot
single_X_test = X_test[single_record_index].unsqueeze(0)
single_y_test = y_test[single_record_index]

model.eval()
with torch.no_grad():
    pi, mu, sigma = model(single_X_test)
    pi = pi.squeeze().numpy()
    mu = mu.squeeze().numpy()
    sigma = sigma.squeeze().numpy()

x = np.linspace(single_y_test - 3, single_y_test + 3, 1000)
y = np.zeros_like(x)

for i in range(num_gaussians):
    y += pi[i] * (1 / (sigma[i] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu[i]) / sigma[i]) ** 2)

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Predicted Density')
plt.axvline(single_y_test, color='r', linestyle='--', label='Actual Value')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Probability Density for a Single Record')
plt.legend()
plt.grid(True)
plt.show()