import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import inv
import warnings
import torch.optim as optim
import torch.nn.functional as F
from old_proj import InputDataset
from torch.utils.data import DataLoader
from old_proj.plot import plot_fig
from tqdm import tqdm
import math
from lib.simulated_data import load_data
import pdb

cuda = torch.cuda.is_available()


class MLP(nn.Module):
    '''
    Simple feedforward network for classification.
    '''
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size))

    def forward(self, x):
        if cuda:
            x = x.cuda()
        return self.model(x)


def train(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0

    for batch_data in data_loader:
        inputs = torch.Tensor(batch_data['InputVector'].float())
        results = torch.Tensor(batch_data['Result'].float())
        if cuda:
            inputs, results = inputs.cuda(), results.cuda()
        optimizer.zero_grad()
        predictions = model(inputs)
        predictions = torch.squeeze(predictions)
        loss = F.mse_loss(predictions, results)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    return total_loss


def f(model, W, X):
    return model(torch.tensor(X).float()).cpu().detach().numpy() + np.squeeze(W) * X


def rand_smoothing(model, X, W, sigma, n):
    cnt, res, pred = 0, 0, np.zeros_like(X, dtype=np.float32)
    for i in range(n):
        rand_noise = np.random.normal(0, sigma, X.shape)
        pred += f(model, W, X + rand_noise)
    return pred / n


def get_monotonicity(model, W, X, sigma, n, upper, lower):
    z = torch.tensor(np.random.uniform(lower, upper, n), dtype=torch.float32)
    const = (upper - lower) / n * math.sqrt(2 * math.pi) * sigma
    nn_output = torch.squeeze(torch.pow(model(torch.unsqueeze(z, 1)), 2), 1)
    monotonicity = np.empty(X.shape)
    for j, x_i in enumerate(X):
        res = 0
        for i, z_i in enumerate(z):
            res += torch.exp(-0.5 * torch.pow(((z_i - x_i) / sigma), 2)) * nn_output[i]
        res = const * res
        monotonicity[j] = res.item() <= W
    return monotonicity


def get_monotonicity_bounded(W, sigma):
    if W >= 1 / sigma:
        return True
    return False


def main():
    for fn in range(4):
        data, n, lower, upper = load_data(n=1000, lower=-2, upper=2, fn=fn)

        X_train = data['data_train'][:, data['X_cols']].astype(np.number)
        X_test = data['data_test'][:, data['X_cols']].astype(np.number)
        Y_train = data['data_train'][:, data['Y_col']:data['Y_col'] + 1].astype(np.number)
        Y_test = data['data_test'][:, data['Y_col']:data['Y_col'] + 1].astype(np.number)

        W = np.dot(np.dot(inv(np.dot(X_train.T, X_train)), X_train.T), Y_train)
        y_hat = np.squeeze(W) * X_train
        residual = Y_train - y_hat

        num_epoch, lr, batch_size = 500, 1e-3, 64
        model = MLP(X_train.shape[1], 1024, 1)
        if cuda:
            model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        dataset = InputDataset.InputDataset(inputs=X_train, results=residual)
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False)

        print('Fitting residual...\n')
        for i in range(1, num_epoch + 1):
            train_loss = train(data_loader, model, optimizer)
            if i % 10 == 0:
                print('Epoch [%d]: Train loss: %.3f.' % (i, train_loss))

        sigma_space, sigma, bounded = np.logspace(0, -2), 0, True
        print('Finding appropriate sigma...\n')
        for s in tqdm(sigma_space):
            if bounded:
                monotonicity = get_monotonicity_bounded(W[0][0], s)
                if monotonicity:
                    sigma = s
                    break
            else:
                monotonicity = get_monotonicity(model, W[0][0], np.squeeze(X), s, n, upper, lower)
                if np.average(monotonicity) > .8:
                    sigma = s
                    break
        pdb.set_trace()
        smoothed = rand_smoothing(model, X_test, W, sigma, n)
        Y = {'linear': y_hat, 'w/o smooth': f(model, W, X_test), 'smooth': smoothed, 'gt': y,
             'nn': model(torch.tensor(X, dtype=torch.float32)).cpu().detach().numpy()}
        plot_fig(X=X, Y=Y)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
