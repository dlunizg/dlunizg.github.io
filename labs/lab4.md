---
layout: page
mathjax: true
permalink: /lab4/
---
- [Generativni modeli](#gm)
- [Ograničeni Boltzmanov stroj](#rbm)
  - [1. zadatak](#1zad)
  - [2. zadatak](#2zad)
  - [3. zadatak](#3zad)
- [Varijacijski autoenkider](#vae)
  - [4. zadatak](#4zad)
- [Generative adversarial networks](#gan)
  - [5. zadatak](#5zad)



<a name='gm'></a>

## 4. vježba: Generativni modeli (GM)
_v1.3-2020_

NAPOMENA: Ukoliko brzina izvršavanja zadataka predstavlja problem zbog nedostpunosti GPUa, napominjemo da Google unutar svoje Google Colab usluge nudi besplatno GPU na korištenje.

U ovoj vježbi pozabavit ćete se s generativnim modelima. Njihova glavna razlika u odnosu na diskriminativne modele je u tome što su predviđeni za generiranje uzoraka karakterističnih za distribuciju uzoraka korištenih pri treniranju. Da bi to mogli raditi na odgovarajući način, nužno je da mogu naučiti bitne karakteristike uzoraka iz skupa za treniranje. Jedna moguća reprezentacija tih bitnih karakteristika je distribucija ulaznih vektora, a model bi uz pomoć takve informacije mogao generirati više uzoraka koji su vjerojatniji (više zastupljeni u skupu za treniranje), a manje uzoraka koji su manje vjerojatni.

Distribucija uzoraka iz skupa za treniranje može se opisati distribucijom vjerojatnosti više varijabli
$$p(\mathbf x)$$. Vjerojatnost uzoraka za treniranje $$\mathbf x^{(i)}$$ trebala bi biti visoka dok bi vjerojatnost ostalih uzoraka trebala biti niža. Nasuprot tome, diskriminativni modeli se, na više ili manje izravne načine, fokusiraju na aposteriornu vjerojatnost razreda $$d$$

$$ 
p(d\vert \mathbf{x})=\frac{p(d)p(\mathbf{x}\vert d)}{p(\mathbf{x})}
$$

Gornji izraz sugerira da bi poznavanje $$p(\mathbf x)$$ mogla biti korisna informacija i za diskriminativne modele, iako je oni u pravilu ne koriste direktno. Ipak, logična je pretpostavka da bi preciznije poznavanje $$p(\mathbf x)$$ moglo pomoći u boljoj procjeni $$p(d \vert \mathbf{x})$$. Tu ideju dodatno podupire i razumna pretpostavka, da su i ulazni uzorci i odgovarajući razred $$d$$ (izlaz), posljedica istih bitnih značajki. Ulazni uzorci sadrže u sebi znatnu količinu bitnih informacija, ali često sadrže i šum koji dodatno otežava modeliranje direktne veze sa izlazom. Model veze izlaza i bitnih značajki, očekivano je jednostavniji nego direktna veza ulaza i izlaza.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/bitneZ.svg" width="70%">
</div>

Ovakva razmišljanja upućuju na upotrebu generativnih modela za ekstrakciju bitnih značajki. Njihova primarna namjena - generiranje uzoraka - tada je u drugom planu. Nakon treniranja, na njih se može nadograditi dodatni diskriminativni model (npr. MLP) koji na temelju bitnih značajki "lako" određuje željeni izlaz. Ova vježba fokusira se na treniranje generativnih modela.

<a name='rbm'></a>

### Ograničeni Boltzmanov stroj (RBM)

Boltzmanov stroj (BM) je [stohastička](https://en.wikipedia.org/wiki/Stochastic_neural_network) [rekurzivna](https://en.wikipedia.org/wiki/Recursive_neural_network) [generativna](https://en.wikipedia.org/wiki/Generative_model) mreža koja treniranjem nastoji maksimizirati $$p(\mathbf x^{(i)})$$, a temelji se na Boltrmanovoj distribuciji prema kojoj je vjerojatnost stanja $$\mathbf x$$ to manja, što je veća njegova energija $$E(\mathbf x)$$ prema sljedećem izrazu

$$
p(\mathbf{x})\propto
e^{\frac{-{E(\mathbf{x})}}{\mathit{kT}}}
$$

Umnožak Boltzmanove konstanta $$k$$ i termodinamičke temperature $$T$$ se ignorira, odnosno postavlja na 1.

Pojedini elementi stanja BM-a $$x_j$$ su binarni i mogu poprimiti vrijednosti 0 i 1. Energetska funkcija $$E(\mathbf x)$$ kod BM-a određena je elementima stanja $$x_j$$ i težinama veza $$w_{ji}$$ između njih te pripadajućim pomacima $$b_j$$.

$$
E(\mathbf{x})=-\left(\frac{1}{2}\sum _{i=1}^{N}\sum
_{\substack{j=1 \\ j\neq i}}^{N}w_{\mathit{ji}}x_{j}x_{i}+\sum
_{j=1}^{N}b_{j}x_{j}\right)=-\left(\frac{1}{2}\mathbf{x^{T}Wx}+\mathbf{b^{T}x}\right)
$$

Matrica $$\mathbf{W}$$ je simetrična te ima nule na glavnoj dijagonali. Vjerojatnost pojedinog uzorka definiramo kao 

$$
p(\mathbf{x};\mathbf{W},\mathbf{b})=\frac{e^{-E(\mathbf{x})/T}}{\sum_{\mathbf{x}}e^{-E(\mathbf{x})/T}}=\frac{e^{\frac{1}{2}\mathbf{x^{T}Wx}+\mathbf{b^{T}x}}}{Z(\mathbf{W},\mathbf{b})}
$$

$$Z(\mathbf W)$$ se naziva particijska funkcija, a uloga joj je normaliziranje vjerojatnosti kako bi

$$
\sum_{\mathbf{x}}p(\mathbf{x};\mathbf{W},\mathbf{b})=1
$$

Prema odabranoj energetskoj funkciji i Boltzmanovoj distribuciji određena je vjerojatnost da pojedini element mreže ima vrijednost 1

$$
p(x_{j}=1)=\frac{1}{1+e^{-\sum
_{i=1}^{N}w_{\mathit{ji}}x_{i}-b_{j}}}=\sigma \left(\sum
_{i=1}^{N}w_{\mathit{ji}}x_{i}+b_{j}\right)
$$

Kako bismo energetskom funkcijom BM-a mogli opisati korelacije višeg reda, odnosno kompleksnije međusobne veze pojedinih elemenata vektora podataka, uvodimo tzv. skrivene varijable $$h$$. Stvarne podatke nazivamo vidljivim slojem i označavamo s $$\mathbf v$$, dok skrivene varijable čine skriveni sloj $$\mathbf h$$.

$$
\mathbf{x}=(\mathbf v,\mathbf h)
$$

RBM je mreža u kojoj nisu dozvoljene međusobne povezanosti unutar istog sloja. To ograničenje (od tuda ime Restricted Boltzman Machine) omogućuje jednostavno osvježavanje stanja mreže. Iako ima poznatu svrhu, skriveni sloj $$\mathbf h$$ i njegova distribucija $$p(\mathbf h)$$ nisu poznati. 

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/rbm.svg" width="20%">
</div>

Energija mreže sada postaje

$$
E(\mathbf{v},\mathbf{h})=-\mathbf{v^{T}Wh}-\mathbf{b^{T}h}-\mathbf{a^{T}v}
$$

Matrica $$\mathbf W$$ sadrži težine veza između vidljivog i skrivenog sloja i više nije simetrična, a vektori $$\mathbf a$$ i $$\mathbf b$$ sadrže pomake vidljivog i skrivenog sloja.
Prema novoj strukturi i prethodnoj jednadžbi za vjerojatnost pojedinog elementa dobivamo dvije jednadžbe za osvježavanje stanja RBM-a. 

$$p(v_{i}=1)=\sigma \left(\sum
_{j=1}^{N}w_{\mathit{ji}}h_{j}+a_{i}\right)$$ za vidljivi sloj

$$p(h_{j}=1)=\sigma \left(\sum
_{i=1}^{N}w_{\mathit{ji}}v_{i}+b_{j}\right)$$ za skriveni sloj

Uzorkovanje vrijednosti pojedine varijable provodi se prema gornje dvije jednadžbe i pomoću generatora slučajnih brojeva.

```python
sampled_tensor = probability_tensor.bernoulli()
```

**Učenje RBM-a**

Prisjetimo se da želimo maksimizirati vjerojatnost svih uzoraka za učenje (stvaranih podatka) koji su u RBM-u predstavljeni vidljivim slojem. Stoga maksimiziramo umnožak svih $$p(\mathbf{v}^{(j)})$$ gdje je 

$$
p(\mathbf{v};\mathbf{W},\mathbf{a},\mathbf{b})=\sum
_{\mathbf{h}}p(\mathbf{v},\mathbf{h};\mathbf{W},\mathbf{a},\mathbf{b})=\sum
_{\mathbf{h}}{\frac{e^{\mathbf{v}^{T}\mathbf{W}\mathbf{h}+\mathbf{b^{T}h}+\mathbf{a^{T}v}}}{Z(\mathbf{W},\mathbf{a, b})}}
$$

Maksimiziramo logaritam umnoška vjerojatnosti svih vidljivih vektora.

$$
\ln \left[\prod
_{n=1}^{N}p(\mathbf{v}^{(n)};\mathbf{W},\mathbf{a},\mathbf{b})\right]
$$

Da bismo to postigli trebamo odrediti parcijalne derivacije s obzirom na parametre mreže

$$\frac{\partial }{\partial w_{\mathit{ij}}}\ln \left[\prod
_{n=1}^{N}p(\mathbf{v}^{(n)};\mathbf{W},\mathbf{a},\mathbf{b})\right]=\sum
_{n=1}^{N}\left[v_{i}^{(n)}h_{j}^{(n)}-\sum
_{\mathbf{v,h}}v_{i}h_{j}p(\mathbf{v,h};\mathbf{W},\mathbf{a},\mathbf{b})\right]=N\left[\langle
v_{i}h_{j}\rangle
_{P(\mathbf{h}\vert \mathbf{v}^{(n)};\mathbf{W},\mathbf{b})}-\langle
v_{i}h_{j}\rangle
_{P(\mathbf{v},\mathbf{h};\mathbf{W},\mathbf{a},\mathbf{b})}\right]$$

$$\frac{\partial }{\partial b_{j}}\ln \left[\prod
_{n=1}^{N}p(\mathbf{v}^{(n)};\mathbf{W},\mathbf{a},\mathbf{b})\right]=\sum
_{n=1}^{N}\left[h_{j}^{(n)}-\sum
_{\mathbf{v,h}}h_{j}p(\mathbf{v,h};\mathbf{W},\mathbf{a},\mathbf{b})\right]=N\left[\langle
h_{j}\rangle
_{P(\mathbf{h}\vert \mathbf{v}^{(n)};\mathbf{W},\mathbf{b})}-\langle
h_{j}\rangle
_{P(\mathbf{v},\mathbf{h};\mathbf{W},\mathbf{a},\mathbf{b})}\right]$$

$$\frac{\partial }{\partial a_{j}}\ln \left[\prod
_{n=1}^{N}p(\mathbf{v}^{(n)};\mathbf{W},\mathbf{a},\mathbf{b})\right]=\sum
_{n=1}^{N}\left[v_{j}^{(n)}-\sum
_{\mathbf{v,h}}v_{j}p(\mathbf{v,h};\mathbf{W},\mathbf{a},\mathbf{b})\right]=N\left[\langle
v_{j}\rangle -\langle v_{j}\rangle
_{P(\mathbf{v},\mathbf{h};\mathbf{W},\mathbf{a},\mathbf{b})}\right]$$

Konačni izrazi sve tri jednadžbe sadrže po dvije komponenete u kojima $$\langle \rangle$$ zagrade označavaju usrednjene vrijednosti za $$N$$ ulaznih uzoraka (obično je to veličina mini grupe). 
Prvi pribrojnici u konačnim izrazima odnose se na stanja mreže kada su ulazni uzorci fiksirani u vidljivom sloju. Za određivanje odgovarajućih stanja skrivenog sloja $$\mathbf{h}$$ dovoljno je svaki element $$h_j$$ odrediti prema izrazu za $$p(h_j = 1)$$.
Drugi pribrojnici odnose se na stanja mreže bez fiksnog vidljivog sloja pa se ta stanja mogu interpretirati kao nešto što mreža zamišlja na temelju trenutne konfiguracije parametara ($$\mathbf W$$, $$\mathbf a$$ i $$\mathbf b$$). Da bi došli do tih stanja trebamo iterativno naizmjence računati nova stanja slojeva ([Gibssovo uzorkovanje](https://en.wikipedia.org/wiki/Gibbs_sampling)) prema izrazima za $$p(h_j = 1)$$ i $$p(v_i = 1)$$. Zbog izostanka međusobnih veza elemenata istog sloja, u jednoj iteraciji se prvo paralelno uzorkuju svi skriveni elementi, a nakon toga svi elementi vidljivog sloja. Teoretski, broj iteracija treba biti velik kao bi dobili ono što mreža "stvarno" misli, odnosno kako bi došli do stacionarne distribucije. Tada je svejedno koje početno stanje vidljivog sloja uzmemo. Praktično rješenje ovog problema je Contrastive Divergenvce (CD) algoritam gdje je dovoljno napraviti svega $$k$$ iteracija (gdje je $$k$$ mali broj, često i samo 1), a za početna stanja vidljivog sloja uzimamo ulazne uzorke. Iako je ovo odmak od teorije, u praksi se pokazalo da dobro funkcionira. Vizualizacija CD-1 algoritma dana je na slici.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/CD.svg" width="50%">
</div>

Korekcija težina i pomaka za ulazni uzorak vi, tada se može realizirati na sljedeći način:

$$\Delta w_{\mathit{ij}}= \eta \left[\langle v_{i}h_{j}\rangle ^{0}-\langle
v_{i}h_{j}\rangle ^{1}\right]$$, 
$$\Delta b_{j}=\eta \left[\langle h_{j}\rangle ^{0}-\langle h_{j}\rangle
^{1}\right]$$, 
$$\Delta a_{i}=\eta \left[\langle v_{i}\rangle ^{0}-\langle v_{i}\rangle
^{1}\right]$$, 

Faktor učenja $$\eta$$ obično se postavlja na vrijednost manju od 1. Prvi pribrojnik u izrazu za $$\Delta w_{\mathit{ij}}$$ često se naziva pozitivna faza, a drugi pribrojnik, negativna faza.

U zadacima će se koristiti MNIST baza. Iako pikseli u MNIST slikama mogu poprimiti realne vrijednosti iz raspona [0, 1], svaki piksel možemo promatrati kao vjerojatnost binarne varijable $$p(v_i = 1)$$.
Ulazne varijable tada možemo tretirati kao stohastičke binarne varjable s [Bernoulijevom razdiobom](https://en.wikipedia.org/wiki/Bernoulli_distribution) i zadanom vjerojatnosti distribucijom $$p(v_i = 1)$$.

<a name='1zad'></a>

### 1. zadatak

Implementirajte RBM koji koristi CD-1 za učenje. Ulazni podaci neka su MNIST brojevi. Vidljivi sloj tada mora imati 784 elementa, a skriveni sloj neka ima 100 elemenata. Kako su vrijednosti ulaznih uzoraka (slika) realne u rasponu [0 1], oni mogu poslužiti kao $$p(v_i = 1)$$ pa za inicijalne vrijednosti vidljivog sloja trebate provesti uzorkovanje. Koristitie mini grupe veličine 100 uzoraka, a treniranje neka ima 100 epoha.

**Podzadaci:**

1. Vizualizirajte težine $$\mathbf W$$ ostvarene treniranjem te pokušajte interpretirati ostvarene težine pojedinih skrivenih neurona.
2. Vizualizirajte rezultate rekonstrukcije prvih 20 testnih uzoraka MNIST baze. Kao rezultat rekonstukcije koji ćete prikazati, koristite $$p(v_{i}=1)=\sigma \left(\sum_{j=1}^{N}w_{\mathit{ji}}h_{j}+a_{i}\right)$$, umjesto binarnih vrijednosti dobivenih uzorkovanjem.
3. Ispitajte učestalost uključenosti elemenata skrivenog sloja te vizualizirajte naučene težine $$\mathbf W$$ soritrane prema učestalosti
4. Preskočite inicijalno uzorkovanje/binarizaciju na temelju ulaznih uzoraka, već ulazne uzorke (realne u rasponu [0 1]) koristiti kao ualzne vektore $$\mathbf v$$. Koliko se tako dobijeni RBM razlikuje od prethodnog?
5. Povećajte broj Gibsovih uzorkovanja k u CD-k. Koje su razlike?
6. Ispitajte efekt variranja koeficijenta učenja.
7. Slučajno inicijalizirjte skriveni sloj, provedite nekoliko Gibbsovih uzorkovanje te vizualizirajte generirane uzorke vidljivog sloja
8. Provedite prethodne eksperimente za manji i veći broj skrivenih neurona. Što opažate kod težina i rekonstrukcija? 

**NAPOMENA**: Osim nadopunjavanja koda koji nedostaje, predložak se treba prilagođavati prema potrebi, a može i prema vlastitim preferencijama. Stoga **budite oprezni s tvrdnjama da vam neki dio koda ne radi!**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torchvision import datasets, transforms
import tqdm
from torchvision.utils import make_grid

import torch.distributions as tdist

import numpy as np
import tqdm

import matplotlib.pyplot as plt

BATCH_SIZE = 100
EPOCHS = 100
VISIBLE_SIZE = 784
HIDDEN_SIZE = 100

def visualize_RBM_weights(weights, grid_width, grid_height, slice_shape=(28, 28)):
    for idx in range(0, grid_width * grid_height):
        plt.subplot(grid_height, grid_width, idx+1)
        plt.imshow(weights[..., idx].reshape(slice_shape))
        plt.axis('off')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def draw_rec(inp, title, size, Nrows, in_a_row, j):
    plt.subplot(Nrows, in_a_row, j)
    plt.imshow(inp.reshape(size), vmin=0, vmax=1, interpolation="nearest")
    plt.title(title)
    plt.axis('off')
    
    
def reconstruct(ind, states, orig, weights, biases, h1_shape=(10, 10), v_shape=(28,28)):
    j = 1
    in_a_row = 6
    Nimg = states.shape[1] + 3
    Nrows = int(np.ceil(float(Nimg+2)/in_a_row))
    
    plt.figure(figsize=(12, 2*Nrows))
       
    draw_rec(states[ind], 'states', h1_shape, Nrows, in_a_row, j)
    j += 1
    draw_rec(orig[ind], 'input', v_shape, Nrows, in_a_row, j)
    
    reconstr = biases.copy()
    j += 1
    draw_rec(sigmoid(reconstr), 'biases', v_shape, Nrows, in_a_row, j)
    
    for i in range(h1_shape[0] * h1_shape[1]):
        if states[ind,i] > 0:
            j += 1
            reconstr = reconstr + weights[:,i]
            titl = '+= s' + str(i+1)
            draw_rec(sigmoid(reconstr), titl, v_shape, Nrows, in_a_row, j)
    plt.tight_layout()

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=BATCH_SIZE)

class RBM():
    
    def __init__(self, visible_size, hidden_size, cd_k=1):
        self.v_size = visible_size
        self.h_size = hidden_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.W = torch.Tensor(normal_dist.sample(sample_shape=(self.v_size, self.h_size)))
        self.v_bias = torch.Tensor(torch.zeros(self.v_size))
        self.h_bias = torch.Tensor(torch.zeros(self.h_size))

    
    def forward(self, batch):
        return self._cd_pass(batch)
    
    
    def __call__(self, batch):
        return self.forward(batch)
    
    
    def _cd_pass(self, batch):
        batch = batch.view(-1, 784)
        h0_prob = 
        h0 = 

        h1 = h0

        for step in range(0, self.cd_k):
            v1_prob = 
            v1 = 
            h1_prob = 
            h1 = 
            
        return h0_prob, h0, h1_prob, h1, v1_prob, v1
    
    def reconstruct(self, h, gibbs_steps=None):
        h1 = h
        
        steps_to_do = self.cd_k
        if gibbs_steps is not None:
            steps_to_do = gibbs_steps

        for step in range(0, steps_to_do):
            v1_prob = 
            v1 = 
            h1_prob = 
            h1 = 

        return h1_prob, h1, v1_prob, v1

    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h0_prob, h0, h1_prob, h1, v1_prob, v1 = self._cd_pass(batch)

        w_positive_grad = 
        w_negative_grad = 

        dw = (w_positive_grad - w_negative_grad) / batch.shape[0]

        self.W = self.W + 
        self.v_bias = self.v_bias + 
        self.h_bias = self.h_bias + 


model = RBM(visible_size=VISIBLE_SIZE, hidden_size=HIDDEN_SIZE, cd_k=1)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        model.update_weights_for_batch(sample, 0.1)


plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(model.W.data, 10, 10)


sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20): 
    h0_prob, h0, h1_prob, h1, v1_prob, v1 = model(sample)


    plt.figure(figsize=(8, 4), facecolor='w')
    plt.subplot(1, 3, 1)
    plt.imshow(sample[idx, ...].view(28, 28).cpu())
    if idx == 0:
        plt.title("Original image")

    plt.subplot(1, 3, 2)
    recon_image = v1_prob[idx, ...].view(28, 28)
    plt.imshow(recon_image.cpu().data)
    if idx == 0:
        plt.title("Reconstruction")
    
    plt.subplot(1, 3, 3)
    state_image = h1[idx, ...].view(10, 10)
    plt.imshow(state_image.cpu().data)
    if idx == 0:
        plt.title("Hidden state")

sample, _ = next(iter(test_loader))
sample = sample[0, ...].view(-1, 784)

h0_prob, h0, h1_prob, h1, v1_prob, v1 = model(sample)

reconstruct(0, h1.numpy(), sample.numpy(), model.W.numpy(), model.v_bias.numpy())


sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

h0_prob, h0, h1_prob, h1, v1_prob, v1 = model(sample)

h0_prob, h0, h1_prob, h1, v1_prob, v1, model_weights, model_v_biases = list(map(lambda x: x.numpy(), [h0_prob, h0, h1_prob, h1, v1_prob, v1, model.W, model.v_bias]))


# Vjerojatnost da je skriveno stanje uključeno kroz Nu ulaznih uzoraka
plt.figure(figsize=(9, 4))
tmp = (h1.sum(0)/h1.shape[0]).reshape((10, 10))
plt.imshow(tmp, vmin=0, vmax=1, interpolation="nearest")
plt.axis('off')
plt.colorbar()
plt.title('vjerojatnosti (ucestalosti) aktivacije pojedinih neurona skrivenog sloja')

# Vizualizacija težina sortitranih prema učestalosti
plt.figure(figsize=(16, 16))
tmp_ind = (-tmp).argsort(None)
visualize_RBM_weights(model_weights[:, tmp_ind], 10, 10)
plt.suptitle('Sortirane matrice tezina - od najucestalijih do najmanje koristenih')


r_input = np.random.rand(100, HIDDEN_SIZE)
r_input[r_input > 0.9] = 1 # postotak aktivnih - slobodno varirajte
r_input[r_input < 1] = 0
r_input = r_input * 20 # pojačanje za slučaj ako je mali postotak aktivnih

s = 10
i = 0
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s

h1_prob, h1, v1_prob, v1 = model.reconstruct(torch.from_numpy(r_input).float(), 19)

plt.figure(figsize=(16, 16))
for idx in range(0, 19):
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(r_input[idx, ...].reshape(10, 10))
    if idx == 0:
        plt.title("Set state")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(h1[idx, ...].view(10, 10))
    if idx == 0:
        plt.title("Final state")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(v1_prob[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    plt.axis('off')
```

<a name='2zad'></a>

### 2. zadatak

Deep belief Network (DBN) je duboka mreža koja se dobije slaganjem više RBM-ova jednog na drugi, pri čemu se svaki sljedeći RBM pohlepno trenira pomoću skrivenog ("izlaznog") sloja prethodnog RBM-a (osim prvog RBM-a koji se trenira direktno s ulaznim uzorcima). Teoretski, tako izgrađen DBN trebao bi povećati $$p(\mathbf v)$$ što nam je i cilj. Korištenje DBN, odnosno rekonstrukcija ulaznog uzorka provodi se prema donjoj shemi. U prolazu prema gore određuju se skriveni slojevi iz vidljivog sloja dok se ne dođe do najgornjeg RBM-a, zatim se na njemu provede CD-k algoritam, nakon čega se, u prolasku prema dolje, određuju niži skriveni slojevi dok se ne dođe do rekonstruiranog vidljivog sloja. Težine između pojedinih slojeva su iste u prolazu gore kao i u prolazu prema dolje. Implementirajte troslojni DBN koji se sastoji od dva pohlepno pretrenirana RBM-a. Prvi RBM neka je isit kao i u 1. zadatku, a drugi RBM neka ima skriveni sloj od 100 elemenata.

**Podzadaci:**

1. Vizualizirajte težine $$\mathbf W_2$$ ostvarene treniranjem.
2. Vizualizirajte rezultate rekonstrukcije prvih 20 testnih uzoraka MNIST baze. 
3. Slučajno inicijalizirjte krovni skriveni sloj, provedite nekoliko Gibbsovih uzorkovanje te vizualizirajte generirane uzorke vidljivog sloja - usporedite s prethodnim zadatkom.
4. Postavite broj skrivenih varijabli gornjeg RBM-a jednak broju elemenata vidljivog sloja donjeg RBM-a, a inicijalne težine $$\mathbf W_2$$ postavite na $$\mathbf W_1^T$$. Koji su efekti promjene? Vizualizirajte uzorke krovnog skrivenog sloja kao matrice 28x28.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/DBN1.svg" width="100%">
</div>

Koristite sljedeći predložak uz predložak 1. zadatka:

**NAPOMENA**: Osim nadopunjavanja koda koji nedostaje, predložak se treba prilagođavati prema potrebi, a može i prema vlastitim preferencijama. Stoga **budite oprezni s tvrdnjama da vam neki dio koda ne radi!**

```python
class DBN():

    def __init__(self, first_rbm: RBM, second_hidden_size, cd_k=1):
        self.v_size = first_rbm.v_size
        self.h1_size = first_rbm.h_size
        self.h2_size = second_hidden_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.W1 = first_rbm.W
        self.v_bias = first_rbm.v_bias.clone()
        self.h1_bias = first_rbm.h_bias.clone()
        
        self.W2 = torch.Tensor(normal_dist.sample(sample_shape=(self.h1_size, self.h2_size)))
        self.h2_bias = torch.Tensor(torch.zeros(self.h2_size))
    
    
    def forward(self, batch, steps=None):
        batch = batch.view(-1, 784)
        
        h1up_prob = 
        h1up = 
        
        h2up_prob = 
        h2up = 
        
        h1down_prob, h1down, h2down_prob, h2down = self.gibbs_sampling(???, steps)
        
        return h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down

    
    def gibbs_sampling(self, h2, steps=None):
        h2down = h2
        
        steps_to_do = self.cd_k
        
        if steps is not None:
            steps_to_do = steps

        for step in range(0, steps_to_do):
            h1down_prob = 
            h1down = 
            
            h2down_prob = 
            h2down = 
            
        return h1down_prob, h1down, h2down_prob, h2down 
    
    def reconstruct(self, h2, steps=None):
        _, _, h2down_prob, h2down = self.gibbs_sampling(???, steps)
        
        h1down_prob = 
        h1down = 
        
        v_prob = 
        v_out = 
        
        return v_prob, v_out, h2down_prob, h2down
    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down = self.forward(batch)

        w2_positive_grad = 
        w2_negative_grad = 

        dw2 = (w2_positive_grad - w2_negative_grad) / h1up.shape[0]

        self.W2 = self.W2 + 
        self.h1_bias = self.h1_bias + 
        self.h2_bias = self.h2_bias + 
        
                
    
    def __call__(self, batch):
        return self.forward(batch)


dbnmodel = DBN(model, second_hidden_size=100, cd_k=2)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        dbnmodel.update_weights_for_batch(sample, learning_rate=0.1)

plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel.W2.data.cpu(), 10, 10, slice_shape=(10, 10))

sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20):
    h1up_prob, h1up, h2up_prob, h2up, h1down_prob, h1down, h2down_prob, h2down = dbnmodel(sample[idx, ...])
    v_prob, v, _, _ = dbnmodel.reconstruct(h2down)

    plt.figure(figsize=(4*3, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(sample[idx,...].view(28, 28))
    if idx == 0:
        plt.title("Test input")
    
    plt.subplot(1, 3, 2)
    plt.imshow(v_prob[0, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    
    plt.subplot(1, 3, 3)
    plt.imshow(h2down.view(10, 10))
    if idx == 0:
        plt.title("Hidden state")


r_input = np.random.rand(100, HIDDEN_SIZE)
r_input[r_input > 0.9] = 1 # postotak aktivnih - slobodno varirajte
r_input[r_input < 1] = 0
r_input = r_input * 20 # pojačanje za slučaj ako je mali postotak aktivnih

s = 10
i = 0
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s

v_out_prob, v_out, h2down_prob, h2down = dbnmodel.reconstruct(torch.from_numpy(r_input).float(), 100)

plt.figure(figsize=(16, 16))
for idx in range(0, 19):
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(r_input[idx, ...].reshape(10, 10))
    if idx == 0:
        plt.title("Set state")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(h2down[idx, ...].view(10, 10))
    if idx == 0:
        plt.title("Final state")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(v_out_prob[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    plt.axis('off')
```

Kako bi se dodatno poboljšala generativna svojstva DBN-a, može se provesti generativni fine-tuning parametara mreže. U 2. zadatku, prilikom rekonstruiranja korištene su iste težine i pomaci u prolascima prema dolje i prema gore. Kod fine-tuninga, parametri koji vežu sve slojeve osim dva najgornja, razdvajaju se u dva skupa. Matrice težina između nižih slojeva dijele se na: $$\mathbf R_n$$ za prolaz prema gore i $$\mathbf W'_n$$ za prolaz prema dolje. Inicijalno su obje matrice jednake originalnoj matrici $$\mathbf W_n$$. Kod prolaza prema gore (faza budnosti - wake phase) određuju se nova stanja viših skrivenih slojeva $$\mathbf s^{(n)}$$ iz nižih stanja $$\mathbf s^{(n-1)}$$ pomoću matrica $$\mathbf R$$ postupkom uzorkovanja ($$sample \left(\sigma \left(\mathbf R_n \mathbf s^{(n-1)} + \mathbf b^{up}_n\right)\right) \to \mathbf s^{(n)}$$) . Pri prolasku prema dolje (faza spavanja - sleep phase) određuju se "rekonstrukcije" nižih stanja $$\mathbf s^{(n-1)}$$ iz $$\mathbf s^{(n)}$$ i matrica $$\mathbf W'$$ ($$sample \left( \sigma \left(\mathbf W'_n \mathbf s^{(n)} + \mathbf b^{down}_{n-1} \right) \right) \to \mathbf s^{(n-1)}$$). Najgornja dva sloja su klasični RBM i dijele istu matricu težina za prolaske u oba smjera, a modificiranje tih težina provodi se na isti način kao u 1.zadatku.

Treniranje težina između nižih slojeva je drugačije. Matrice $$\mathbf W'_n$$ se korigiraju kada se određuju nova stanja pomoću matrica $$\mathbf R_n$$ u prolasku prema gore. U prolasku prema dolje korigiraju se matrice $$\mathbf R_n$$. Vektori pomaka pojedinih slojeva $$\mathbf b_n$$ se isto dijele na varijante za prolaz prema gore $$\mathbf b_n^{up}$$ i za ptrolaz prem dolje $$\mathbf b_n^{down}$$. Inicijalni pomaci jednaki su originalnim pomacima $$\mathbf b$$.

Za korekciju matrica $$\mathbf W'_n$$ prilikom prolaska prema gore ($$sample \left(\sigma \left(\mathbf R_n \mathbf s^{(n-1)} + \mathbf b^{up}_n\right)\right) \to \mathbf s^{(n)}$$) provodi se i $$sample \left( \sigma \left(\mathbf W'_n \mathbf s^{(n)} + \mathbf b^{down}_{n-1} \right) \right) \to \mathbf s^{(n-1)novo}$$. Korekcija elemenata radi se na sljedeći način
$$\Delta w'_{\mathit{ij}}=\eta
s_{j}^{(n)}(s_{i}^{(n-1)}-s_{i}^{(n-1)\mathit{novo}})$$
Korekcija pomaka za prolaz prema dolje provodi se na sljedeći način
$$\Delta b_{\mathit{i}}^{\mathit{down}}=\eta
(s_{i}^{(n-1)}-s_{i}^{(n-1)\mathit{novo}})$$

Za korekciju matrica $$\mathbf R_n$$ prilikom prolaska prema dolje ($$sample \left( \sigma \left(\mathbf W'_n \mathbf s^{(n)} + \mathbf b^{down}_{n-1} \right) \right) \to \mathbf s^{(n-1)}$$) provodi se i $$sample \left(\sigma \left(\mathbf R_n \mathbf s^{(n-1)} + \mathbf b^{up}_n\right)\right) \to \mathbf s^{(n)novo}$$. Korekcija elemenata radi se na sljedeći način
$$\Delta r_{\mathit{ij}}=\eta
s_{i}^{(n-1)}(s_{j}^{(n)}-s_{j}^{(n)\mathit{novo}})$$
Korekcija pomaka za prolaz prema dolje provodi se na sljedeći način
$$\Delta b_{\mathit{i}}^{\mathit{up}}=\eta
(s_{i}^{(n)}-s_{i}^{(n)\mathit{novo}})$$

Navedeni postupak provodi se za svaki uzorak za treniranje te se naziva up-down algoritam (ponegdje i wake-sleep algoritam).

HINT: pseudokod za treniranje četveroslojnog DBN-a nalazi se u dodacima ovog [članka](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)



<a name='3zad'></a>

### 3. Zadatak

Implementirajte postupak generativnog fine-tuninga na DBN iz 2. zadatka. Za treniranje gornjeg RBM-a koristite CD-2.

**Podzadaci:**

1. Vizualizirajte konačne varijante matrica $$\mathbf W'$$, $$\mathbf R$$ i njihovu apsolutnu razliku.
2. Vizualizirajte rezultate rekonstrukcije prvih 20 testnih uzoraka MNIST baze.
3. Slučajno inicijalizirajte krovni skriveni sloj, provedite nekoliko Gibbsovih uzorkovanje te vizualizirajte generirane uzorke vidljivog sloja - usporedite s prethodnim zadacima

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/DBN2.svg" width="40%">
</div>

Koristite sljedeći predložak, kao i predloške 1. i 2. zadatka.

**NAPOMENA**: Osim nadopunjavanja koda koji nedostaje, predložak se treba prilagođavati prema potrebi, a može i prema vlastitim preferencijama. Stoga **budite oprezni s tvrdnjama da vam neki dio koda ne radi!**

```python
class DBNWithFineTuning():

    def __init__(self, base_dbn: DBN, cd_k=1):
        self.v_size = base_dbn.v_size
        self.h1_size = base_dbn.h1_size
        self.h2_size = base_dbn.h2_size
        self.cd_k = cd_k
        
        normal_dist = tdist.Normal(0, 0.1)
        
        self.R1 = base_dbn.W1.clone()
        self.W1_down = base_dbn.W1.T.clone()
        self.v1_bias = base_dbn.v_bias.clone()
        self.h1_up_bias = base_dbn.h1_bias.clone()
        self.h1_down_bias = base_dbn.h1_bias.clone()
        
        self.W2 = base_dbn.W2.clone()
        self.h2_bias = base_dbn.h2_bias.clone()
    
    
    def forward(self, batch, steps=None):
        batch = batch.view(-1, 784)
        
        h1_up_prob = 
        h1_up = 
        
        v1_up_down_prob = 
        v1_up_down = 
        
        h2_up_prob = 
        h2_up = 
        
        h1_down_prob, h1_down, h2_down_prob, h2_down = self.gibbs_sampling(???, steps=steps)
        
        v1_down_prob = 
        v1_down = 
        
        h1_down_up_prob = 
        h1_down_up = 
        
        return h1_up_prob, h1_up, v1_up_down_prob, v1_up_down, h2_up_prob, h2_up, h1_down_prob, h1_down, h2_down_prob, h2_down, v1_down_prob, v1_down, h1_down_up_prob, h1_down_up
    
    def gibbs_sampling(self, h2, steps=None):
        h2_down = h2
        
        steps_to_do = self.cd_k
        
        if steps is not None:
            steps_to_do = steps
        
        
        for step in range(0, self.cd_k):
            h1_down_prob =
            h1_down = 

            h2_down_prob = 
            h2_down = 
            
        return h1_down_prob, h1_down, h2_down_prob, h2_down


    
    def reconstruct(self, h2, steps=None):
        h1_down_prob, h1_down, h2_down_prob, h2down = self.gibbs_sampling(???, steps)
        
        v_out_tmp_prob = 
        v_out_tmp =
        v_out_prob = 
        v_out = 
        
        return v_out_prob, v_out, h2_down_prob, h2down
    
    def update_weights_for_batch(self, batch, learning_rate=0.01):
        h1_up_prob, h1_up, v1_up_down_prob, v1_up_down, h2_up_prob, h2_up, h1_down_prob, h1_down, h2_down_prob, h2_down, v1_down_prob, v1_down, h1_down_up_prob, h1_down_up = self.forward(batch)
        
        self.W1_down = self.W1_down + 
        self.R1 = self.R1 + 
        
        self.v1_bias = self.v1_bias +
        
        self.h1_down_bias = self.h1_down_bias + 
        self.h1_up_bias = self.h1_up_bias + 
        
        
        w2_positive_grad = 
        w2_negative_grad = 
        dw2 = (w2_positive_grad - w2_negative_grad) / h1_up.shape[0]
        
        self.W2 = self.W2 + 
        self.h2_bias = self.h2_bias + 
        
    
    def __call__(self, batch):
        return self.forward(batch)


dbnmodel_ft = DBNWithFineTuning(dbnmodel, cd_k=2)
for curr_epoch in tqdm.tqdm(range(0, EPOCHS)):
    for sample, label in train_loader:
        sample = sample.view(-1, 784)
        dbnmodel_ft.update_weights_for_batch(sample, 0.01)

plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel_ft.R1.data, 10, 10)
plt.tight_layout()


plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(dbnmodel_ft.W1_down.T.data, 10, 10)
plt.tight_layout()

difference = torch.abs(dbnmodel_ft.R1.data - dbnmodel_ft.W1_down.T.data)
plt.figure(figsize=(12, 12), facecolor='w')
visualize_RBM_weights(difference, 10, 10)
plt.tight_layout()

sample, _ = next(iter(test_loader))
sample = sample.view(-1, 784)

for idx in range(0, 20): 
    # rbn reconstruct
    _, _, _, _, recon1, _ = model(sample[idx, ...])
    
    # dbn reconstruct
    _, _, _, _, _, _, _, h2down = dbnmodel.forward(sample[idx, ...])
    recon2, _, _, _ = dbnmodel.reconstruct(h2down)
    
    # dbn fine tune reconstruct
    _, _, _, _, _, _, _, _, _, h2_down, _, _, _, _ = dbnmodel_ft(sample[idx, ...])
    recon3, _, _, _ = dbnmodel_ft.reconstruct(h2_down, 2)
    
    plt.figure(figsize=(5*3, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(sample[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Original image")
    
    plt.subplot(1, 5, 2)
    plt.imshow(recon1.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 1")
    
    plt.subplot(1, 5, 3)
    plt.imshow(recon2.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 2")
    
    plt.subplot(1, 5, 4)
    plt.imshow(recon3.view(28, 28))
    if idx == 0:
        plt.title("Reconstruction 3")
    
    plt.subplot(1, 5, 5)
    plt.imshow(h2_down.view(10, 10))
    if idx == 0:
        plt.title("Top state 3")

r_input = np.random.rand(100, 100)
r_input[r_input > 0.9] = 1
r_input[r_input < 1] = 0

s = 10
i = 0
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s
i += 1
r_input[i,:] = 0
r_input[i,i]= s

    
v_out_prob, v_out, h2_down_prob, h2down = dbnmodel_ft.reconstruct(torch.from_numpy(r_input).float(), 100)

plt.figure(figsize=(16, 16))
for idx in range(0, 19):
    plt.figure(figsize=(14, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(r_input[idx, ...].reshape(10, 10))
    if idx == 0:
        plt.title("Set state")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(h2down[idx, ...].view(10, 10))
    if idx == 0:
        plt.title("Final state")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(v_out_prob[idx, ...].view(28, 28))
    if idx == 0:
        plt.title("Reconstruction")
    plt.axis('off')
```


<a name='vae'></a>

### Varijacijski autoenkoder (VAE)

Autoenkoder je mreža s prolazom u naprijed koja za treniranje koristi backpropagation te može imati duboku strukturu. Autoenkoderi su gnereativne mreže sa karakterističnom dvodjelonom strukturom. Prvi dio naziva se enkoder i preslikava (enkodira) ulazni sloj u skriveni sloj.  Drugi dio je dekoder i preslikava skriveni sloj u izlazni sloj. Primarni cilj takve mreže je postići što veću sličnost ulaza i izlaza  za svaki uzorak za treniranje, maksimizirajući neku mjeru sličnosti. Primarni cilj je jednostavan, što autoenkodere svrstava u mreže koje se treniraju bez nadzora. Maksimalna uspješnost može se jednostavno postići direktnim kopiranjem ulaz izlaz, no to nije u skladu sa skrivenim ciljem. Uvjetno rečeno skriveni cilj, koji je u stvari jedini interesantan, je naučiti bitne značajke uzoraka iz skupa za treniranje. Da bi se to postiglo, i izbjeglo direktno kopiranje, koriste se razne tehnike regularizacije. Alterativno, može se koristiti druga mjera uspješnosti, poput maksimiziranja vjerojatnsoti. U svakom slučaju, varijable skrivenog sloja $$\mathbf z$$ zadužene su za otkrivanje bitnih značajki ulaznih uzoraka.

[Varijacijski autoenkoderi (VAE)](http://arxiv.org/abs/1312.6114) su Autoenkoderi koji maksimiziraju vjerojatnost $$p(\mathbf x)$$ svih uzoraka za treniranje $$\mathbf x^{(i)}$$. Kod VAE se ne koriste dodatne regularizacijske tehnike, ali neke od njih se mogu uključiti u VAE (npr. kombinacija VAE i denoising autoenkodera daje bolje rezultate). Bitne značajke u skrivenom sloju $$\mathbf z$$ tada imaju ulogu u modeliranju $$p(\mathbf x)$$. 

$$
p(\mathbf{x})=\int
p(\mathbf{x}\vert \mathbf{z};\theta
)p(\mathbf{z})\mathbf{\mathit{dz}}
$$

Vjerojatnosti sa desne strane jednadžbe su jednako nepoznate kao i $$p(\mathbf x)$$, no njih ćemo aproksimirati Gaussovim distribucijama. $$\Theta$$ su parametri modela i njih određujemo kroz postupak treniranja. Dodatni cilj nam je minimizirati količinu uzorkovanja, koje je obično nužno porvoditi kod estimacija nepoznatih distribucija.
U ovom slučaju, pogodno je maksimizirati logaritam vjerojatnosti.

$$
\log _{\mathbf{\theta }}p(\mathbf{x}^{(1)},\ldots
,\mathbf{x}^{(N)})=\sum _{i=1}^{N}\log
_{\mathbf{\theta }}p(\mathbf{x}^{(i)})
$$

Za $$p(\mathbf z)$$ odabiremo normalnu distribuciju

$$
p(z)=N(0,1)
$$

Time ih ograničavamo i naizgled onemogučavamo u tome da poredstavljaju bitne značajke. 

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE1.svg" width="50%">
</div>

Dekoderskim dijelom modeliramo uvjetnu distribuciju $$p(\mathbf x \vert \mathbf z)$$ kao normalnu distribuciju, a parametre te distribucije određuju parametri mreže $$\Theta$$.

$$
p_{\mathbf{\theta }}(x\vert z)=N(\mu _{x}(z),\sigma _{x}(z))
$$

Parametri $$\Theta$$ uključuju težine i pomake svih neurona skrivenih slojeva dekodera te se određuju kroz treniranje. Složenost $$p(\mathbf x \vert \mathbf z)$$ ovisi o broju skrivenih slojeva i broju neurona u njima. Crtkane linije na dijagramu označavaju operacije uzorkovanja. U dijagramu je radi preglednosti prikazana samo jedna varijabla z, umjesto vektora skrivenih varijabli $$\mathbf z$$.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_dec.svg" width="50%">
  <div class="figcaption figcenter">Dekoderski dio</div>
</div>

Sada bi trebalo odrediti $$p(\mathbf z \vert \mathbf x)$$, takav da gornje pretpostavke dobro funkcioniraju. Kako nemamo načina odrediti odgovarajući $$p(\mathbf z \vert \mathbf x)$$, aproksimarati ćemo ga normalnom distribucijom $$q(\mathbf z \vert \mathbf x)$$, ali ćemo pažljivo odrediti parametre te zamjenske distribucije.

$$
q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x})=N(\mathbf{\mu
_{z}(x),\sigma _{z}(x)})
$$

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_enc.svg" width="50%">
  <div class="figcaption figcenter">Enkoderski dio</div>
</div>

Slično kao i kod dekodera, parametri $$\Phi$$ uključuju težine i pomake skrivnih slojeva enkodera te se oni određuju kroz postupak treniranja. Kompleksnost $$q(\mathbf z \vert \mathbf x)$$ ovisi o broju skrivenih slojeva i broju neurona u njima. 

Model je sada potpun, samo nam još treba funkcija cilja koju možemo optimizirati ispravnim odabirom parametara $$\Theta$$ i $$\Phi$$.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_enc_dec1.svg" width="100%">
</div>

Neuroni koji predstavljaju srednje vrijednosti i standardne devijacije obično nemaju nelinearne aktivacijske funkcije. 
Kako smo već naveli, želja nam je maksimizirati 

$$
\log _{\mathbf{\theta }}p(\mathbf{x}^{(1)},\ldots
,\mathbf{x}^{(N)})=\sum _{i=1}^{N}\log
_{\mathbf{\theta }}p(\mathbf{x}^{(i)})
$$

Prikladnom transformacijom $$\log(p(\mathbf x))$$, što je jedan pribrojnik u grnjoj jednadžbi, možemo prikazati kao

$$
\text{}\log (p(\mathbf x))=D_{\mathit{KL}}\left(q( \mathbf z\vert \mathbf x)\parallel
p(\mathbf z\vert \mathbf x)\right)-D_{\mathit{KL}}\left(q(\mathbf z\vert \mathbf x)\parallel p(\mathbf z)\right)+\text{E}_{q(\mathbf z\vert \mathbf x)}\left(\log
(p(\mathbf x\vert \mathbf z))\right)
$$

$$D_{\mathit{KL}}$$ je [Kullback–Leibler divergencija](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) i predstavlja mjeru sličnosti dviju distribucija. Kako $$p(\mathbf z\vert \mathbf x)$$ mijenjamo sa $$q(\mathbf z\vert \mathbf x)$$, logično je težiti tome da te dvije distribucije budu što sličnije. Tada bi KL divergencija u prvom pribrojniku težila maksimumu, no kako nam je $$p(\mathbf z\vert \mathbf x)$$ nepoznat, maksimiziramo preostala dva pribrojnika. Te dvije komponente zajedno čine donju varijacijsku granicu $$L$$ od $$log(p(\mathbf x))$$ te maksimizacijom donje granice podižemo ukupnu vjerojatnost ulaznog uzorka $$\mathbf x$$. 

$$
L(\mathbf{\theta ,\phi
,x^{(i)}})=-D_{\mathit{KL}}\left(q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})\parallel
p(\mathbf{z})\right)+\text{E}_{q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})}\left[\log
(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}))\right]
$$

Drugi pribrojnik u gornjoj jednadžbi možemo promatrati kao mjeru uspješnosti rekonstrukcije (mogući maksimum je log(1) kada skriveni sloj $$\mathbf z$$ omogućuje savršenu rekonstrukciju). Prvi član se smatra regularizacijskom komponentom te on potiče izjendačavaje distribucija $$q(\mathbf z\vert \mathbf x)$$ i $$p(\mathbf z)$$.

Uz odabrane aproksimacije
$$q_{\mathbf{\phi }}(z\vert x)=N(\mu _{z}(x),\sigma _{z}(x))
$$
$$
p(z)=N(0,1)
$$
$$
p_{\mathbf{\theta }}(x\vert z)=N(\mu _{x}(z),\sigma _{x}(z))
$$
dvije komponente donje varijacijske granice postaju

$$\begin{equation*}
-D_{\mathit{KL}}\left(q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})\parallel
p(\mathbf{z})\right)=\frac{1}{2}\sum _{j}\left(1+\log (\sigma
_{z_{j}}^{(i)2})-\mu _{z_{j}}^{(i)2}-\sigma _{z_{j}}^{(i)2}\right)
\end{equation*}
$$

$$
\text{E}_{q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})}\left[\log
(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}))\right]\approx
\frac{1}{K}\sum _{k=1}^{K}\log \left(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}^{(i,k)})\right)\approx -\sum
_{j}{\frac{1}{2}\log (\sigma _{x_{j}}^{(i,k)2})+\frac{(x_{j}^{(i)}-\mu
_{x_{j}}^{(i,k)})^{2}}{2\sigma _{x_{j}}^{(i,k)2}}}
$$

Obično se $$K$$ postavlja na 1 kako bi se smanjila količina uzorkovanja, uz uvjet da je veličina minibatcha  barem 100.
Konačni izrazi za dvije komponente sada nam daju konačnu funkciju cilja za jedan ulazni uzorak. Optimizira se prosječna vrijednost za sve uzorke $$\mathbf x^{(i)}$$! Prethodno je potrebno još malo modificirati strukturu mreže kako bi omogućili backproapagation i u enkoderski sloj. Nužno je stohastičke neurone $$\mathbf z$$ pretvoriti u determinističke neurone s stohastičkim dodatkom (generatorom šuma ε po normalnoj razdiobi $$N(0,1)$$).

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_enc_dec2.svg" width="100%">
</div>

Primijetite da konačna struktura mreže uključuje stohastičko uzorkovanje, no dijelovi mreže gdje se to događa, ne utječu na propagaciju gradijenta pogreške. To uključuje i same izlaze mreže koji, možda neočekivano, ne sudjeluju u funkciji cilja. U konačnom izrazu funkcije cilja, javljaju se srednje vrijednosti i standardne devijacije izlaza i skrivenih varijabli, a to su zapravo izlazi enkodera i dekodera. Za standardne devijacije je karakteristično da su one uvijek pozitivne, no izlaz iz mreže ne mora nužno biti. Kako bi se iskoristio puni raspon i smanjio broj potrebnih izračuna, izlazi mreža postavljaju se na $$log(\sigma^2)$$ umjesto $$σ$$.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/VAE_enc_dec3.svg" width="100%">
</div>

Konačni algoritam treniranja VAE sada je:
1. Inicijaliziraj parametre $$\Theta$$ i $$\Phi$$
2. Ponavljaj
3. &nbsp;&nbsp; Odaberi slučajni minibatch $$\mathbf X^M$$
4. &nbsp;&nbsp;  Uzorkuj ε
5. &nbsp;&nbsp;  Odredi gradijent od $$L$$ s obzirom na $$\Theta$$ i $$\Phi$$
6. &nbsp;&nbsp;  Izračunaj nove vrijednosti za $$\Theta$$ i $$\Phi$$ prema gradijentu
7. Dok $$\Theta$$ i $$\Phi$$ ne konvergiraju

Dakle, ovim postupkom maksimiziramo donju granicu log vjerojatnosti ulaznih uzoraka. To nam ulijeva sigurnost da će i sama log vjerojatnost rasti, ali nije garancija. Teoretski, može se desiti da donja granica raste, a sama vjerojatnost pada, ali u praksi to najčešće nije slučaj. Moguće objašnjenje ovog efekta leži u činjenici da uz dovoljno složen enkoder, $$q(\mathbf z\vert \mathbf x)$$ postaje dovljno kompleksna i omogućuje približavanje distribuciji $$p(\mathbf z\vert \mathbf x)$$, čime se maksimizira i prvi (zanemareni) član izraza za $$log(p(\mathbf x))$$.

Generiranje novih uzoraka provodi se samo dekoderskim dijelom uz slučajnu inicijalizaciju skrivenog sloja $$\mathbf z$$ prama zadanoj distribuciji 
$$
p(z)=N(0,1)
$$
ili nekom ciljanim vektorom $$\mathbf z$$.

I u ovom zadatku se koristi MNIST baza, čije slike i ovaj puta tretiramo kao niz zamišljenih binarnih piksela $$x_i$$ sa Bernoulijevom distribucijom i vjerojatnosti zadanom ulaznom vrijednosti piksela $$p(x_i = 1) = x_i^{in}$$. Tada je bolje da dekoder implementira Bernoulijvu razdiobu umjesto Gaussove. Izlaz dekodera tada može predstavljati vjerojatnost $$p(x_i = 1)$$, što je ujedno i očekivana vrijednost izlaza $$x_i$$. Sama vjerojatnost može biti definiarana kao 

$$
p(x_{i}^{\mathit{out}}=1)=\sigma \left(\sum
_{j=1}^{N}w_{\mathit{ji}}h_{j}+b_{i}\right)
$$

gdje su $$\mathbf W$$ i $$\mathbf b$$ težine i pomaci koji povezuju zadnji sloj dekodera ($$\mathbf h$$) sa vjerojatnosti izlazne varijable $$x_i$$.
U skladu s ovom izmjenom, nužno je izmijeniti funkciju cilja, preciznije, njezin dio koji se odnosi na točnost rekonstrukcije

$$\text{E}_{q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})}\left[\log
(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}))\right]$$

Kod binarnih varijabli s Bernoulijevom razdiobom, izraz postaje

$$
\text{E}_{q_{\mathbf{\phi
}}(\mathbf{z}\vert \mathbf{x}^{(i)})}\left[\log
(p_{\mathbf{\theta
}}(\mathbf{x}^{(i)}\vert \mathbf{z}))\right]=-\sum
_{j}\left[x_{j}^{\text{in}}\log
p(x_{j}^{\mathit{out}}=1)+(1-x_{j}^{\text{in}})\log
(1-p(x_{j}^{\mathit{out}}=1))\right]=\sum
_{j}H\left(p(x_{j}=1),p(x_{j}^{\mathit{out}}=1)\right)
$$


<a name='4zad'></a>

### 4. Zadatak

Implementirajte VAE sa 20 skrivenih varijabli $$z$$. Ulazni podaci neka su MNIST brojevi. Enkoder i dekoder neka imaju po dva skrivena sloja, svaki sa 200 neurona sa "softplus" aktivacijskim funkcijama. 

**Podzadaci:**

 1. Vizualizirajte rezultate rekonstrukcije za prvih 20 testnih uzoraka MNIST baze. 
 2. Vizualizirajte distribucije srednjih vrijednosti i standardnih devijacija skrivenih varijabli $$z$$ za primjereni broj ulaznih uzoraka
 3. Vizualizirajte raspored testnih uzoraka u 2D prostoru skrivenih varijabli.
 4. Ponovite eksperimente iz prethodnih podzadataka za samo 2 elementa u skrivenenom sloju $$\mathbf z$$. 

Koristite sljedeći predložak:

**NAPOMENA**: Osim nadopunjavanja koda koji nedostaje, predložak se treba prilagođavati prema potrebi, a može i prema vlastitim preferencijama. Stoga **budite oprezni s tvrdnjama da vam neki dio koda ne radi!**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torchvision import datasets, transforms
import tqdm
from torchvision.utils import make_grid

import torch.distributions as tdist

import numpy as np
import tqdm

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

class VAE(nn.Module):
    
    def __init__(self, latent_size):
        super(VAE, self).__init__()
        
        self.latent_size = latent_size
        
        ???
        
    def encode(self, x):
        # TODO!
        return ???, ???
        
    def reparametrize(self, mu, logvar):
        # TODO!
        return ???
    
    def decode(self, z):
        # TODO!
        return ???
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, z, mu, logvar
    
    @staticmethod
    def loss_fn(reconstruction, batch, mu, logvar):
        crossentropy = 
        kl_div = 
        return ???

LATENT_SIZE = 2
model = VAE(LATENT_SIZE)
model = train(model, batch_size=1024, device='cuda', n_epochs=100, log_epochs=10, learning_rate=3.24e-4)

plot_reconstructions('cuda', state_shape=(2, 1))

_, test_loader = prepare_data_loaders()
df_mu, df_logvar, df_dec1_weights = generate_latent_dataframes(test_loader)
plt.figure(figsize=(16, 16))
sns.scatterplot(x='mu_z0', y='mu_z1', hue='label', s=50, data=df_mu)

plot_data_boxplots(df_mu, df_logvar, df_dec1_weights)

walk_in_latent_space(latent_space_abs_limit=1.5, sqrt_sample_count=15, latent_size=LATENT_SIZE, dimensions_to_walk=(0,1))
```

Funkcije korištene u gornjem primjeru možete nakodirati sami, ili možete iskoristiti sljedeći predložak:

```python
def prepare_data_loaders(batch_size=32):
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                   ])), batch_size=batch_size)
    
    return train_loader, test_loader

def train(model, n_epochs=10, log_epochs=1, batch_size=32, learning_rate=1e-3, device='cpu'):
    train_loader, test_loader = prepare_data_loaders(batch_size)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for epoch_idx in range(0, n_epochs):
        
        train_loss = 0
        for batch_idx, (image_data, _) in enumerate(train_loader):
            image_data = image_data.to(device)
            
            optimizer.zero_grad()
            reconstructed_batch, batch_z, batch_mu, batch_logvar = model(image_data)
            loss = model.loss_fn(reconstructed_batch, image_data, batch_mu, batch_logvar)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        if epoch_idx % log_epochs == 0:
            print(f"Epoch {epoch_idx+1}/{n_epochs}: {train_loss / (len(train_loader) * train_loader.batch_size):.2f}")
            
    model.eval()
    
    return model

def plot_reconstructions(device='cpu', number_of_samples=10, state_shape=(4, 5)):
    train_loader, test_loader = prepare_data_loaders(batch_size=number_of_samples)
    batch, _ = next(iter(test_loader))
    recons, zs, mus, logvars = model(batch.to(device))
    
    for idx in range(0, number_of_samples):
        original_image = batch[idx, ...].view(28, 28).data.cpu()
        recon_image = recons[idx, ...].view(28, 28).data.cpu()
        state = zs[idx, ...].view(*state_shape).data.cpu()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)

        plt.subplot(1, 3, 2)
        plt.imshow(recon_image)
        
        plt.subplot(1, 3, 3)
        plt.imshow(state)
        plt.clim(-4, 4)
        plt.colorbar()

def generate_latent_dataframes(data_loader):
    mu_acc = []
    logvar_acc = []
    label_acc = []

    for image_data, label in tqdm.tqdm(data_loader):
        mu, logvar = model.encode(image_data.view(-1, 784).to('cuda'))

        mu_acc.extend(mu.data.cpu().numpy())
        logvar_acc.extend(logvar.data.cpu().numpy())
        label_acc.extend(label.data.cpu().numpy())

    mu_acc = np.array(mu_acc)
    logvar_acc = np.array(logvar_acc)


    tmp = {
        'label': label_acc
    }
    for idx in range(0, mu_acc.shape[1]):
        tmp[f'mu_z{idx}'] = mu_acc[..., idx]

    df_mu = pd.DataFrame(tmp)
    df_mu['label'] = df_mu['label'].astype('category')


    tmp = {
        'label': label_acc
    }
    for idx in range(0, mu_acc.shape[1]):
        tmp[f'logvar_z{idx}'] = np.square(np.exp(logvar_acc[..., idx]))

    df_logvar = pd.DataFrame(tmp)
    df_logvar['label'] = df_logvar['label'].astype('category')


    tmp = {}
    for idx in range(0, model.dec_fc1.weight.T.shape[0]):
        tmp[f'w{idx}'] = list(model.dec_fc1.weight.T[idx, ...].data.cpu().numpy())

    df_dec1_weights = pd.DataFrame(tmp)
    
    return df_mu, df_logvar, df_dec1_weights

def plot_data_boxplots(df_mu, df_logvar, df_dec1_weights, baseline_figsize=(1.2, 6)):
    figwidth, figheight = baseline_figsize
    df_mu2 = df_mu.melt(['label'])
    plt.figure(figsize=(int(figwidth * LATENT_SIZE), figheight))
    sns.boxplot(x='variable', y='value', data=df_mu2)
    plt.title("Distribution of $\mu$ in latent space")

    df_logvar2 = df_logvar.melt(['label'])
    plt.figure(figsize=(int(figwidth * LATENT_SIZE), figheight))
    sns.boxplot(x='variable', y='value', data=df_logvar2)
    plt.title("Distribution of $\sigma^2$ in latent space")

    df_dec1_weights2 = df_dec1_weights.melt()
    plt.figure(figsize=(int(figwidth * LATENT_SIZE), figheight))
    sns.boxplot(x='variable', y='value', data=df_dec1_weights2)
    plt.title("Weights going to decoder from latent space")

def walk_in_latent_space(latent_space_abs_limit=3, sqrt_sample_count=20, latent_size=2, dimensions_to_walk=(0, 1), figsize=(16, 16)):
    dim1, dim2 = dimensions_to_walk
    canvas = np.zeros((sqrt_sample_count * 28, sqrt_sample_count * 28))

    synthetic_representations = np.zeros((sqrt_sample_count * sqrt_sample_count, latent_size))

    synthetic_representations[..., dim1] = np.linspace(-latent_space_abs_limit, latent_space_abs_limit, num=sqrt_sample_count * sqrt_sample_count)
    synthetic_representations[..., dim2] = np.linspace(-latent_space_abs_limit, latent_space_abs_limit, num=sqrt_sample_count * sqrt_sample_count)

    recons = model.decode(torch.from_numpy(synthetic_representations).float().to('cuda'))

    for idx in range(0, sqrt_sample_count * sqrt_sample_count):
        x, y = np.unravel_index(idx, (sqrt_sample_count, sqrt_sample_count))
        canvas[y*28:((y+1) * 28), x*28:((x+1) * 28)] = recons[idx, ...].view(28, 28).data.cpu().numpy()

    plt.figure(figsize=figsize)
    plt.imshow(canvas)
```


<a name='gan'></a>

### Generative asdversarial networks (GAN)

Primarna namjena GAN-a je isto generiranje novih i uvjerljivih uzoraka, no princip rada je malo drugačiji od prethodna dva modela. GAN ne procjenjuje direktno parametre $$p(\mathbf x)$$ ili bilo koje druge distribucije, premda se njegovo treniranje može interpretirati kao estimacija $$p(\mathbf x)$$. Najvjerojatnije zahvaljujući tom drugačijem pristupu, GAN-ovi često generiraju vizualno najbolje uzorke u usporedbi sa VAE ili drugim generativnim mrežama.

GAN se sastoji od dvije zasebne mreže 

1. Generator (G) koji ima zadatak generirati uvjerljive uzorke
2. Diskriminator (D) koji ima zadatak prepoznati radi li se o pravom uzorku (iz skupa za trniranje) ili lažnom uzorku koji je generirao G

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/GAN.svg" width="100%">
</div>

Te dvije mreže su protivnici (Adversaries), imaju dijametralno suprotstavljene ciljeve te se pokušavaju nadmudriti. To nadmetanje ih tjera da budu sve bolji u postizanju svog cilja i da se fokusiraju na sve bitne detalje ulaznih podataka. Očekivano, njihovo nadmetanje trebalo bi dovesti do toga da generator generira savršene uzorke koje diskriminator ne može razlikovati od uzoraka iz skupa za treniranje. Da bi generator postigao takav uspjeh nužno je da i diskriminator bude maksimalno dobar u svom zadatku.

Generator na svojem izlazu generira uzorke za neki slučajni ulazni vektor koji prati neku distribuciju. Ta slučajnost na ulazu omogućuje generatoru da uvijek generira nove uzorke. Pri tome nema nekih posebnih ograničenja na arhitekturu generatora, no poželjno je da se može trenirati backpropagation algoritmom. 

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/G.svg" width="50%">
</div>

Diskriminator na svome izlazu treba estimirati pripadnost razredu stvarnih ili lažnih uzoraka za svaki ulazni vektor. Za razliku od generatora, ovdje je moguće koristiti učenje pod nadzorom jer se za svaki uzorak zna da li je došao iz skupa za treniranje ili od generatora. Radi jednostavnosti možemo izlaz diskriminatora ograničiti u rasponu $$[0,1]$$ i interpretirati kao vjerojatnost da je ulazni uzorak stvaran (iz skupa za treniranje).

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/D.svg" width="50%">
</div>

Gore opisani ciljevi diskriminatora i generatora mogu se formalno izraziti u sljedećoj funkciji cilja:

$$\min_G \max_D V(D,G) = E_{ \mathbf x \sim p_{data}(\mathbf x) } [\log D( \mathbf x)] + E_{ \mathbf z  \sim p_{\mathbf z}(\mathbf z) } [\log(1 - D(G( \mathbf z)))]$$

Prvi pribrojnik predstavlja očekivanje procjene log vjerojatnosti da su uzorci iz skupa za treniranje stvarni. Drugi pribrojnik predstavlja očekivanje procjene log vjerojatnosti da generirani uzorci nisu stvarni, tj. da su umjetni. Diskriminator ima za cilj maksimizirati oba pribrojnika, dok generator ima za cilj minimizirati drugi pribrojnik. Svaki pribrojnik funkcije cilja može se jednostavno procijeniti za jednu mini grupu te se može procijeniti gradijent s obzirom na prametre obiju mreža. 

Treniranje dviju mreža (G i D) može se provesti istovremeno ili se u jednoj iteraciji prvo može trenirati jedna mreža a zatim druga. Dodatno, neki autori preporučuju da se u nekoliko uzastopnih iteracija trenira jedna mreža, a nakon toga druga mreža samo jednu iteraciju.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/GAN2.svg" width="100%">
</div>


Kod generiranja slika uspješnim se pokazao Deep Convolutional GAN (DCGAN) koji u skrivenim slojevima obiju mreža koristi konvolucijske slojeve. Za razliku od klasičnih konvolucijskih mreža, ovdje se ne koriste pooling slojevi nego se uzorkovanje provodi pomoću konvolucijskih slojeva koji imaju posmak veći od 1. Autori mreže preporučuju korištenje Batch normalizacije u svim slojevima osim u izlaznom sloju generatora te ulaznom i izlaznom sloju diskriminatora. Korištenje Leaky ReLU aktivacijskih funkcija u svim slojevima osim u izlaznim je još jedna specifičnost DCGAN-a kao i eliminacija potpuno povezanih slojeva.

<div class="fig figcenter fighighlight">
  <img src="/assets/lab4/DCGAN.svg" width="100%">
</div>

<a name='5zad'></a>

### 5. Zadatak

Implementirajte DCGAN s generatorom (4 konvolucijska sloja) i diskriminatorom (3 konvolucijska sloja). Arhitekura treba biti:
    
* Generator
    * Sloj 1 - Broj izlaznih kanala = 512, veličina jezgre = 4, veličina koraka = 1
    * Sloj 2 - Broj izlaznih kanala = 256, veličina jezgre = 4, veličina koraka = 2, padding = 1
    * Sloj 3 - Broj izlaznih kanala = 128, veličina jezgre = 4, veličina koraka = 2, padding = 1
    * Sloj 4 - Broj izlaznih kanala = 64, veličina jezgre = 4, veličina koraka = 2, padding = 1
    * Sloj 5 - Broj izlaznih kanala = 1, veličina jezgre = 4, veličina koraka = 2, padding = 1

* Diskriminator
    * Sloj 1 - Broj izlaznih konvolucija = 64, veličina jezgre = 4, veličina koraka = 2, padding = 1
    * Sloj 2 - Broj izlaznih konvolucija = 128, veličina jezgre = 4, veličina koraka = 2, padding = 1
    * Sloj 3 - Broj izlaznih konvolucija = 256, veličina jezgre = 4, veličina koraka = 2, padding = 1
    * Sloj 4 - Broj izlaznih konvolucija = 512, veličina jezgre = 4, veličina koraka = 2, padding = 1
    * Sloj 5 - Broj izlaznih konvolucija = 1, veličina jezgre = 4, veličina koraka = 1, padding = 0

Ulaz u generator $$\mathbf z$$ neka ima 100 elemenata prema normalnoj distribuciji $$N(0,1)$$. Ulazni podaci neka su MNIST brojevi skalirani na veličinu 32x32 te treniranje provedite kroz barem 20 epoha. U jednoj iteraciji provedite jednu optimizaciju generatora i jednu optimizaciju diskriminatora s po jednom mini grupom. Koristite tanh aktivacijsku funkciju za izlaz generatora i sigmoid aktivaciju za izlaz diskriminator, a za ostaje slojeve "propustljivi" ReLU sa "negative_slope" parametrom od 0.2. Batch noramlizacija (jedan od podzadataka) ide iza svakog sloja.

**Podzadaci:**

 1. Vizualizirajte rezultate generiranja 100 novih uzoraka iz slučajnih vektora $$\mathbf z$$. Usporedite rezultate s uzorcima generiranim pomoću VAE.
 2. U jednoj iteraciji provedite treniranje diskriminatora sa dvaje minigrupe a generatora sa jednom minigrupom. Vizualizirajte generirane uzorke. Ponovite isti postupak samo zamijenite mjesta generatora i diskriminatora. Komentirajte retzultate.
 3. Isključite batch normalizaciju u obje mreže. Komentirajte rezultate.

Koristite sljedeći predložak:

**NAPOMENA**: Osim nadopunjavanja koda koji nedostaje, predložak se treba prilagođavati prema potrebi, a može i prema vlastitim preferencijama. Stoga **budite oprezni s tvrdnjama da vam neki dio koda ne radi!**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torchvision import datasets, transforms
import tqdm
from torchvision.utils import make_grid

import torch.distributions as tdist

import numpy as np
import tqdm

import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

def prepare_data_loaders(batch_size=32):
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files', train=True, download=True, 
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((64, 64)),
                                   torchvision.transforms.ToTensor()
                               ])), batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize((64, 64)),
                                       torchvision.transforms.ToTensor()
                                   ])), batch_size=batch_size)
    
    return train_loader, test_loader

class Generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.latent_size = latent_size
        
        ???


    def forward(self, x):
        ???

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        ???

    def forward(self, x):
        ???

        return x

def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

dmodel = Discriminator()
gmodel = Generator(100)

dmodel.apply(weights_init)
gmodel.apply(weights_init)

def train(gmodel: Generator, dmodel: Discriminator, n_epochs=10, log_epochs=1, batch_size=32, learning_rate=1e-3, device='cpu'):
    train_loader, test_loader = prepare_data_loaders(batch_size=batch_size)
    
    gmodel = gmodel.to(device)
    dmodel = dmodel.to(device)
    
    gmodel.train()
    dmodel.train()
    
    criterion = nn.BCELoss()
    
    g_optim = optim.Adam(gmodel.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optim = optim.Adam(dmodel.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    for epoch_idx in range(0, n_epochs):
        
        g_loss, d_loss = 0, 0
        
        for image_data, _ in tqdm.tqdm(train_loader):
            # discriminator update
            dmodel.zero_grad()
            
            # real data pass
            image_data = image_data.to(device)
            
            batch_size = image_data.shape[0]
            labels = torch.ones(batch_size, device=device).float()
            
            d_output = dmodel(image_data)
            d_err_real = criterion(d_output, labels)
            d_err_real.backward()
            d_loss += d_err_real.item() / batch_size
            
            # fake data pass
            noise = torch.randn(batch_size, gmodel.latent_size, 1, 1, device=device)
            fake_image_data = gmodel(noise)
            labels = torch.zeros(batch_size, device=device).float()
            
            d_output = dmodel(fake_image_data.detach())
            d_error_fake = criterion(d_output, labels)
            d_error_fake.backward()
            d_loss += d_error_fake.item() / batch_size
    
            d_optim.step()
            
            # generator update
            gmodel.zero_grad()
            
            labels = torch.ones(batch_size, device=device)
            d_output = dmodel(fake_image_data)
            g_error = criterion(d_output, labels)
            g_error.backward()
            g_loss += g_error.item() / batch_size 
            g_optim.step()
            
        if (epoch_idx + 1) % log_epochs == 0:
            print(f"[{epoch_idx+1}/{n_epochs}]: d_loss = {d_loss:.2f} g_loss {g_loss:.2f}")

    gmodel.eval()
    dmodel.eval()
    
    return gmodel, dmodel

gmodel, dmodel = train(gmodel, dmodel, n_epochs=15, batch_size=256, device='cuda')

random_sample = gmodel(torch.randn(100, 100, 1, 1).to('cuda')).view(100, 64, 64).data.cpu().numpy()

plt.figure(figsize=(16, 16))
for idx in range(0, 100):
    plt.subplot(10, 10, idx+1)
    plt.imshow(random_sample[idx, ...])
    plt.clim(0, 1)
    plt.axis('off')
```

