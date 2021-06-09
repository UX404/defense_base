import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def drawgrad(grad):
    grad = grad.cpu().numpy()[0][0]
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection='3d')
    ax.view_init(60, 35)
    x = np.linspace(0, 32, 32)
    y = np.linspace(0, 32, 32)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, grad, rstride=1, cstride=1, cmap='viridis', edgecolor='black')
    plt.show()


def pgd_attack(model, x, y, step_size, epsilon, perturb_steps,
                random_start=None, distance='l_inf'):
    model.eval()
    batch_size = len(x)
    if random_start:
        x_adv = x.detach() + random_start * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

class PGDAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        if self.random_start:
            x_adv = x.detach() + self.random_start * torch.randn(x.shape).cuda().detach()
        else:
            x_adv = x.detach()
        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            # grad ([128, 3, 32, 32])

            # drawgrad(grad)
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            
            
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv        
