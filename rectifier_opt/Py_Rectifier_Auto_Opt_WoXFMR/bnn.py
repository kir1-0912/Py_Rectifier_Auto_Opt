import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from Allocate_EMX import  Rect_Opt_Flow
import numpy as np

class Linear_BBB(nn.Module):
    def __init__(self, input_features, output_features, prior_var=1.):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.w_mu = nn.Parameter(torch.zeros(output_features, input_features))
        self.w_rho = nn.Parameter(torch.zeros(output_features, input_features))
        self.b_mu = nn.Parameter(torch.zeros(output_features))
        self.b_rho = nn.Parameter(torch.zeros(output_features))
        self.w = None
        self.b = None
        self.prior = Normal(0, prior_var)

    def forward(self, input):
        w_epsilon = Normal(0, 1).sample(self.w_mu.shape)
        self.w = self.w_mu + torch.log(1 + torch.exp(self.w_rho)) * w_epsilon
        b_epsilon = Normal(0, 1).sample(self.b_mu.shape)
        self.b = self.b_mu + torch.log(1 + torch.exp(self.b_rho)) * b_epsilon
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)
        self.w_post = Normal(self.w_mu.data, torch.log(1 + torch.exp(self.w_rho)))
        self.b_post = Normal(self.b_mu.data, torch.log(1 + torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(self.w).sum() + self.b_post.log_prob(self.b).sum()
        return F.linear(input, self.w, self.b)

class MLP_BBB(nn.Module):
    def __init__(self, input_dim, hidden_units, prior_var=1.):
        super().__init__()
        self.hidden = Linear_BBB(input_dim, hidden_units, prior_var=prior_var)
        self.out = Linear_BBB(hidden_units, 1, prior_var=prior_var)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.out(x)
        return x

    def log_prior(self):
        return self.hidden.log_prior + self.out.log_prior

    def log_post(self):
        return self.hidden.log_post + self.out.log_post

    def sample_elbo(self, input, target, samples):
        outputs = torch.zeros(samples, target.shape[0])
        log_priors = torch.zeros(samples)
        log_posts = torch.zeros(samples)
        log_likes = torch.zeros(samples)

        for i in range(samples):
            outputs[i] = self(input).reshape(-1)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
            log_likes[i] = Normal(outputs[i], 0.1).log_prob(target.reshape(-1)).sum()

        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        loss = log_post - log_prior - log_like
        return loss


x_train = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
])
y_train = Rect_Opt_Flow(x_train).reshape(-1, 1)

net = MLP_BBB(input_dim=x_train.shape[1], hidden_units=32, prior_var=10)
optimizer = optim.Adam(net.parameters(), lr=0.1)
epochs = 2000

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = net.sample_elbo(x_train, y_train, samples=1)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {loss.item()}')

x_test = torch.tensor([
    [1.5, 2.5, 3.5],
    [4.5, 5.5, 6.5]
])
with torch.no_grad():
    y_pred = net(x_test).numpy()

print("Predicted output:", y_pred)