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
Izvedite učitavanje podataka tako da se omogući 
učenje modela za metričko ugrađivanje trojnim gubitkom.
Da bismo to napravili, potrebno je prilagoditi MNIST dataset 
tako da se prilikom dohvata primjera za
treniranje (sidra), dohvaćaju i odgovarajući pozitivan i negativan primjer.

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


Implementirajte metode `_sample_positive` i `_sample_negative` 
tako da njihove povratne vrijednosti odgovaraju 
indeksima uzorkovanih slika u listi `self.images`.
Za potrebe ove vježbe dovoljno je implementirati jednostavno uzorkovanje 
koje će za pozitivni primjer uzorkovati slučajnu sliku 
koja pripada istom razredu kao sidro,
a za negativni primjer slučajnu sliku koja pripada 
bilo kojem razredu različitom od razreda sidra.

<a name='2zad'></a>


### 2. zadatak: Definicija modela za metričko ugrađivanje

Zadan je kod koji definira grubu strukturu 
modela za metričko ugrađivanje. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE

class SimpleMetricEmbedding(nn.Module):
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
Nadopunite predloženi kod prema sljedećim uputama.

#### a) gubitak
Implementirajte trojni gubitak po uzoru na pytorchev [`TripletMarginLoss`](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html).

#### b) blok BNReLUConv
U praksi je često praktično grupirati sekvence slojeva 
koje se često ponavljaju u zajednički gradivni blok. 
Oblikujte gradivni blok `BNReLUConv` koji se sastoji od 
normalizacije po grupi, aktivacije ReLU i konačno konvolucije.
Primijetite da naš predložak nasljeđuje razred 
[Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html).
To znači da za dodavanje slojeva u konstruktoru 
možete koristiti metodu `append`.

#### c) metričko ugrađivanje
Dovršite definiciju modela za metričko ugrađivanje. 
Neka se taj model sastoji od 3 `BNReLUConv` bloka 
(neka veličina jezgre bude 3, a broj mapa značajki - `emb_size`) 
između kojih postavite sažimanje maksimumom 
s veličinom jezgre 3 i korakom 2. 
Konačne ugrađivanje slike dobivamo globalnim sažimanjem prosjekom.
Pripazite da izlazni tenzor u metodi `get_features` zadrži prvu dimenziju koja označava veličinu minigrupe, čak i kada je ona jednaka 1. 

<a name='3zad'></a>

### 3. zadatak: Učenje i vrednovanje

Zadan je kod za učenje modela za metričko ugrađivanje na MNIST podatacima. Ovaj kod koristi pomoćnu skriptu `utils.py` dostupnu [ovdje](https://github.com/dlunizg/dlunizg.github.io/tree/master/data/lab4/utils.py).

```python

import time
import torch.optim
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from model import SimpleMetricEmbedding
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
    model = SimpleMetricEmbedding(1, emb_size).to(device)
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
#### a) Uvod
Proučite skriptu za treniranje i `utils.py`. Kako se računaju reprezentacije razreda? Kako se provodi klasifikacija primjera? Probajte smisliti alternativne pristupe za klasifikaciju primjera.

#### b) Klasifikacija na temelju metričkog ugrađivanja
Na MNIST skupu za treniranje naučite model za metričko ugrađivanje iz zadatka 2.c. Klasificirajte slike iz MNIST skupa za validaciju i izmjerite točnost.

#### c) Klasifikacija na temelju udaljenosti u prostoru slike
Klasificirajte slike iz MNIST skupa za validaciju, ali ovaj put u prostoru slike. Značajke u prostoru slike možete dobiti definiranjem jednostavne mreže za vektorizaciju slike.

```python

class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def get_features(self, img):
        # YOUR CODE HERE
        feats = ...
        return feats
```

Implementirajte klasu `IdentityModel` koja sliku pretvara u vektor i modificirajte skriptu za učenje tako da provodi klasifikaciju u prostoru slike. Primjetite da se `IdentityModel` ne može trenirati. Izmjerite točnost klasifikacije na MNIST skupu za validaciju.


#### d) Pohranjivanje parametara modela
U praksi je praktično pohraniti parametre naučenog modela, za kasnije korištenje u fazi zaključivanja. Modificirajte skriptu za treniranje tako da pohranite naučene parametre korištenjem funkcije ['torch.save'](https://pytorch.org/docs/stable/generated/torch.save.html). Iznova istrenirajte model za metričko ugrađivanje i pohranite dobivene parametre.

#### e) Klasifikacija neviđenih razreda
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

Iz MNIST skupa za treniranje uklonite razred 0 te istrenirajte novi model za metričko ugrađivanje iz zadatka 2. Klasificirajte sve slike (uključujući i razred 0) iz MNIST skupa za validaciju na temelju sličnosti u prostoru značajki. Primjetite da ćete trebati imati dva loadera MNIST skupa za treniranje, jedan bez razreda 0 koji ćete koristiti za treniranje modela i drugi sa svim klasama koji ćete iskoristiti za dobivanje prosječne reprezentacije za sve klase. Pohranite parametre naučenog modela.


<a name='4zad'></a>

### 4. zadatak: Vizualizacija podataka
Zanimljivo je pogledati razmještaj podataka u prostoru značajki i prostoru slike. S obzirom da je visokodimenzionalne podatke nemoguće vizualizirati u originalnom prostoru, potrebno je primjere prebaciti u 2D prostor značajki. Ovo možemo napraviti [analizom glavnih komponenti](https://en.wikipedia.org/wiki/Principal_component_analysis). U torchu je dostupna gotova implementacija [pca_lowrank](https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html).

```python
import numpy as np
import torch

from dataset import MNISTMetricDataset
from model import SimpleMetricEmbedding
from matplotlib import pyplot as plt


def get_colormap():
    # Cityscapes colormap for first 10 classes
    colormap = np.zeros((10, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    return colormap


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"= Using device {device}")
    emb_size = 32
    model = SimpleMetricEmbedding(1, emb_size).to(device)
    # YOUR CODE HERE
    # LOAD TRAINED PARAMS

    colormap = get_colormap()
    mnist_download_root = "./mnist/"
    ds_test = MNISTMetricDataset(mnist_download_root, split='test')
    X = ds_test.images
    Y = ds_test.targets
    print("Fitting PCA directly from images...")
    test_img_rep2d = torch.pca_lowrank(ds_test.images.view(-1, 28 * 28), 2)[0]
    plt.scatter(test_img_rep2d[:, 0], test_img_rep2d[:, 1], color=colormap[Y[:]] / 255., s=5)
    plt.show()
    plt.figure()

    print("Fitting PCA from feature representation")
    with torch.no_grad():
        model.eval()
        test_rep = model.get_features(X.unsqueeze(1))
        test_rep2d = torch.pca_lowrank(test_rep, 2)[0]
        plt.scatter(test_rep2d[:, 0], test_rep2d[:, 1], color=colormap[Y[:]] / 255., s=5)
        plt.show()
```

Modificirajte kod tako da učitava parametre naučene u prethodnom zadatku. Više o pohrani i učitavanju parametara možete naći na [pytorch stranicama](https://pytorch.org/tutorials/beginner/saving_loading_models.html). Vizualizirajte primjere u prostoru značajki (za model koji je učen sa svi klasama i model koji je učen bez da je vidio 0) i prostoru slike.

<a name='add'></a>

### Dodatni materijali
