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
#### a) Klasifikacija na temelju udaljenosti u prostoru slike
Klasificirajte slike iz MNIST skupa za validaciju na temelju udaljenosti primjera 
#### b) Klasifikacija na temelju metričkog ugrađivanja
#### c) Klasifikacija neviđenih razreda
#### d) Pohranjivanje parametara modela

<a name='4zad'></a>

### 4. zadatak: Vizualizacija podataka


<a name='add'></a>

### Dodatni materijali
