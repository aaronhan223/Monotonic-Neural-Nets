import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import inv
import warnings
import torch.optim as optim
import torch.nn.functional as F
import InputDataset
from torch.utils.data import DataLoader
from plot import plot_fig
import math
import pdb

true_fn = lambda x: .05 * x + .3 * np.sqrt(x) + np.random.normal(0, .1, 1)
upper, lower, sample_size = 20, 0, 1000


class MLP(nn.Module):
    '''
    Simple feedforward network for classification.
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
            # nn.Tanh(),
            # nn.Linear(hidden_size, output_size),
            # nn.Tanh(),
            # nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


def train(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0

    for batch_data in data_loader:
        inputs = torch.Tensor(batch_data['InputVector'].float())
        results = torch.Tensor(batch_data['Result'].float())
        optimizer.zero_grad()
        predictions = model(inputs)
        predictions = torch.squeeze(predictions)
        loss = F.mse_loss(predictions, results)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    return total_loss


def f(model, W, X):
    return model(torch.tensor(X).float()).detach().numpy() + np.squeeze(W) * X


def rand_smoothing(model, X, W, sigma):
    cnt, res, pred = 0, 0, np.zeros_like(X, dtype=np.float32)
    for i in range(1000):
        rand_noise = np.random.normal(0, sigma, X.shape)
        # FIXME: random smoothing should not include linear part: does not matter here
        pred += f(model, W, X + rand_noise)
    return pred / 1000
    # while True:
    #     rand_noise = np.random.normal(0, sigma, X.shape)
    #     # res += np.sum(np.square(model(torch.tensor(X + rand_noise).float()).detach().numpy())) / X.shape[0]
    #     pred += f(model, W, X + rand_noise)
    #     cnt += 1
    #     # final_res, final_data = res / cnt, pred / cnt
    #     print('W: {}, NN: {}.'.format(W, math.sqrt(final_res) / sigma))
    #     if W >= math.sqrt(final_res) / sigma:
    #         return final_data


def get_sigma(model, W, sigma, unif_tensor, x_search_space, N=500):
    res = 0
    for i in range(N):
        z = torch.tensor(np.random.uniform(lower, upper, sample_size), dtype=torch.float32)
        unweighted = torch.squeeze(torch.pow(model(torch.unsqueeze(z, 1)), 2), 1) / unif_tensor
        gauss_tensor = 1 / (sigma * math.sqrt(2 * math.pi)) * torch.exp(-0.5 * torch.pow(((z - x_search_space) / sigma), 2))
        pdb.set_trace()
        res += torch.dot(gauss_tensor, unweighted)
    res = math.sqrt(res / N)
    if res / sigma <= W:
        return sigma


def main():
    X = np.expand_dims(np.sort(np.random.uniform(lower, upper, sample_size)), axis=1)
    y = true_fn(X)
    W = np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)
    y_hat = np.squeeze(W) * X
    residual = y - y_hat

    num_epoch, lr, batch_size = 100, 1e-3, 64
    model = MLP(X.shape[1], 256, 1)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    dataset = InputDataset.InputDataset(inputs=X, results=residual)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    print('Fitting residual...\n')
    for i in range(1, num_epoch + 1):
        train_loss = train(data_loader, model, optimizer)
        if i % 10 == 0:
            print('Epoch [%d]: Train loss: %.3f.' % (i, train_loss))

    sigma_space, sigma = [1, .5, .1, .05, .01], 0
    x_search_space = torch.tensor(np.arange(lower, upper, 1 / sample_size)).float()
    unif_tensor = torch.ones(sample_size) * 1 / ((upper - lower) * sample_size)
    print('Find appropriate sigma...\n')
    for s in sigma_space:
        sigma = get_sigma(model, W, s, unif_tensor, x_search_space)

    smoothed = rand_smoothing(model, X, W, sigma)
    Y = {'linear': y_hat, 'w/o smooth': model(X), 'smooth': smoothed, 'gt': y}
    plot_fig(X=X, Y=Y)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
