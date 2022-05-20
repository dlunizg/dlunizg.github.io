---
layout: page
mathjax: true
permalink: /lab4_metric/
---

- [Metričko ugrađivanje](#cnn)
- [Vježba](#vjezba)
  - [1. zadatak](#1zad)
  - [2. zadatak](#2zad)
  - [3. zadatak](#3zad)
  - [4. zadatak](#4zad)
- [Dodatni materijali](#add)


<a name='cnn'></a>

## 4. vježba: Metričko ugrađivanje

<a name='vjezba'></a>
## Vježba


<a name='1zad'></a>

### 1. zadatak: Učitavanje podataka
Potrebno je prilagoditi učitavanje podataka tako da se omogući učenje metričkim ugrađivanjem.
Da bi smo to napravili, potrebno je prilagoditi MNIST dataset tako da se prilikom dohvata primjera za
treniranje (sidra), vraćaju i odgovarajući pozitivan i negativan primjer.

```python
from torch.utils.data import Dataset
from collections import defaultdict
from random import choice
import torchvision


class MNISTMetricDataset(Dataset):
    def __init__(self, root="/tmp/mnist/", split='train'):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        # YOUR CODE HERE


    def _sample_positive(self, index):
        # YOUR CODE HERE


    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)
```

Implementirajte metode `_sample_positive` i `_sample_negative`.

<a name='2zad'></a>


### 2. zadatak: Definicija mreže
Zadan je kod kojim se definira struktura sijamske mreže. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE

class SimpleSiamese(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = ...
        return x

    def loss(self, anchor, positive, negative):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        loss = ...
        return loss
```
Nadopunite kod:
#### a) gubitak
Implementirajte trojni gubitak po uzoru na pytorchev [`TripletMarginLoss`](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html).
#### b) blok BNReLUConv
U praksi je često praktično grupirati sekvence slojeva koje se često ponavljaju u zajednički gradivni blok. Napravite BNReLUConv blok u kojem se najprije izvodi normalizacija po podacima, zatim ReLU i konačno kovolucija.
#### c) mreža
Definirajte konačnu arhitekturu mreže. Mreža se sastoji od 3 BNReLUConv bloka (veličina jezgre je 3, broj konvolucijskih jezgara jednak je emb_size) između kojih se koristi sažimanje maksimumom s veličinom jezgre 3 i korakom 2. Konačne značajke za sliku dobiju se globalnim sažimanjem prosjekom.

<a name='3zad'></a>

### 3. zadatak: Eksperimenti
Zadan je kod za klasifikaciju MNIST podataka.

```python
import time
import torch.optim
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from model import SimpleSiamese
import torch.nn as nn

from utils import compute_representations, evaluate

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = ...
        return feats


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    model = # IdentityModel or SimpleSiamese

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./mnist/"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    emb_size = 28 * 28

    print("Computing mean representations for evaluation...")
    representations = compute_representations(model, train_loader, num_classes, emb_size, device)
    if EVAL_ON_TRAIN:
        print("Evaluating on training set...")
        acc1 = evaluate(model, representations, traineval_loader, device)
        print(f"Train Top1 Acc: {round(acc1 * 100, 2)}%")
    if EVAL_ON_TEST:
        print("Evaluating on test set...")
        acc1 = evaluate(model, representations, test_loader, device)
        print(f"Test Accuracy: {acc1 * 100:.2f}%")
```

#### a) Klasifikacija na temelju udaljenosti u prostoru slike
Klasificirajte slike iz MNIST skupa za validaciju na temelju udaljenosti od primjera za treniranje. Klasifikaciju napravite u prostoru slike. Dopunite kod za mrežu `IdentityModel` koja kao značajke vraća vektoriziranu sliku. Izmjerite točnost takve klasifikacije.

#### b) Klasifikacija na temelju metričkog ugrađivanja
Zadan je kod za treniranje i evaluaciju sijamske mreže

```python

import time
import torch.optim
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from model import SimpleSiamese
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = True
EVAL_ON_TRAIN = False


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")

    # CHANGE ACCORDING TO YOUR PREFERENCE
    mnist_download_root = "./mnist/"
    ds_train = MNISTMetricDataset(mnist_download_root, split='train')
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')

    num_classes = 10

    print(f"> Loaded {len(ds_train)} training images!")
    print(f"> Loaded {len(ds_test)} validation images!")

    train_loader = DataLoader(
        ds_train,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    traineval_loader = DataLoader(
        ds_traineval,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1
    )

    emb_size = 32
    model = SimpleSiamese(1, emb_size).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        t0 = time.time_ns()
        train_loss = train(model, optimizer, train_loader, device)
        print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
        if EVAL_ON_TEST or EVAL_ON_TRAIN:
            print("Computing mean representations for evaluation...")
            representations = compute_representations(model, train_loader, num_classes, emb_size, device)
        if EVAL_ON_TRAIN:
            print("Evaluating on training set...")
            acc1 = evaluate(model, representations, traineval_loader, device)
            print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
        if EVAL_ON_TEST:
            print("Evaluating on test set...")
            acc1 = evaluate(model, representations, test_loader, device)
            print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
        t1 = time.time_ns()
        print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}")

```

Na MNIST skupu za treniranje naučite sijamsku mrežu iz zadatka 2.c. Klasificirajte slike iz MNIST skupa za validaciju, ovaj puta u prostoru značajki.

#### c) Pohranjivanje parametara modela
U praksi je praktično pohraniti parametre naučenog modela, za kasnije korištenje u fazi zaključivanja. Modificirajte skriptu za treniranje tako da pohranite naučene parametre korištenjem funkcije ['torch.save'](https://pytorch.org/docs/stable/generated/torch.save.html).

#### d) Klasifikacija neviđenih razreda
Jedna od prednosti učenja metričkim ugrađivanjem nad standardnim klasifikacijskim modelima jest mogućnost dodavanja novih klasa u skup za evaluaciju.
Modificirajte konstruktor `MNISTMetricDataset` tako da se omogući uklanjanje primjera odabrane klase iz skupa za učenje:

```python
    def __init__(self, root="/tmp/mnist/", split='train', remove_class=None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            # Filter out images with target class equal to remove_class
            # YOUR CODE HERE

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]
```

Iz MNIST skupa za treniranje uklonite razred 0 te istrenirajte novu sijamsku mrežu iz zadatka 2. Klasificirajte sve slike (uključujući i razred 0) iz MNIST skupa za validaciju na temelju sličnosti u prostoru značajki.



<a name='4zad'></a>

### 4. zadatak: Vizualizacija podataka


<a name='add'></a>

### Dodatni materijali
