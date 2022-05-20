import numpy as np
import torch
from collections import defaultdict
PRINT_LOSS_N = 100


def train(model, optimizer, loader, device='cuda'):
    losses = []
    model.train()
    for i, data in enumerate(loader):
        anchor, positive, negative, _ = data
        optimizer.zero_grad()
        loss = model.loss(anchor.to(device), positive.to(device), negative.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
        if i % PRINT_LOSS_N == 0:
            print(f"Iter: {i}, Mean Loss: {np.mean(losses):.3f}")
    return np.mean(losses)


def compute_representations(model, loader, identities_count, emb_size=32, device='cuda'):
    model.eval()
    representations = defaultdict(list)
    for i, data in enumerate(loader):
        anchor, id = data[0], data[-1]
        with torch.no_grad():
            repr = model.get_features(anchor.to(device))
            repr = repr.view(-1, emb_size)
        for i in range(id.shape[0]):
            representations[id[i].item()].append(repr[i])
    averaged_repr = torch.zeros(identities_count, emb_size).to(device)
    for k, items in representations.items():
        r = torch.cat([v.unsqueeze(0) for v in items], 0).mean(0)
        averaged_repr[k] = r / torch.linalg.vector_norm(r)
    return averaged_repr


def make_predictions(representations, r):
    return ((representations - r)**2).sum(1) # predictions based on L2 distance


def evaluate(model, repr, loader, device):
    model.eval()
    total = 0
    correct = 0
    for i, data in enumerate(loader):
        anchor, id = data
        id = id.to(device)
        with torch.no_grad():
            r = model.get_features(anchor.to(device))
            r = r / torch.linalg.vector_norm(r)
        pred = make_predictions(repr, r)
        top1 = pred.min(0)[1]
        correct += top1.eq(id).sum().item()
        total += 1
    return correct/total
