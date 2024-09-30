import vbll

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import transforms

def dict_to_data(dict):

    mnist_train_dataset = datasets.MNIST(root='data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    mnist_test_dataset = datasets.MNIST(root='data',
                                train=False,
                                transform=transforms.ToTensor())

    fashion_train_dataset = datasets.FashionMNIST(root='data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    fashion_test_dataset = datasets.FashionMNIST(root='data',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)


    if dict.TRAIN == "mnist":
        train =  mnist_train_dataset
    elif dict.TRAIN == "fashion":
        train = fashion_train_dataset
    if dict.TEST == "mnist":
        test =  mnist_test_dataset
    elif dict.TEST == "fashion":
        test = fashion_test_dataset
    if dict.OOD == "mnist":
        ood =  mnist_test_dataset
    elif dict.OOD == "fashion":
        ood = fashion_test_dataset
    return train, test, ood

class GenVBLLMLP(nn.Module):
  def __init__(self, cfg):
    super(GenVBLLMLP, self).__init__()

    self.params = nn.ModuleDict({
      'in_layer': nn.Linear(cfg.IN_FEATURES, cfg.HIDDEN_FEATURES),
      'core': nn.ModuleList([nn.Linear(cfg.HIDDEN_FEATURES, cfg.HIDDEN_FEATURES) for i in range(cfg.NUM_LAYERS)]),
      'out_layer': vbll.GenClassification(cfg.HIDDEN_FEATURES, cfg.OUT_FEATURES, cfg.REG_WEIGHT, parameterization = cfg.PARAM, return_ood=cfg.RETURN_OOD, prior_scale=cfg.PRIOR_SCALE),
    })
    self.activations = nn.ModuleList([nn.ELU() for i in range(cfg.NUM_LAYERS)])
    self.cfg = cfg

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.params['in_layer'](x)

    for layer, ac in zip(self.params['core'], self.activations):
      x = ac(layer(x))

    return self.params['out_layer'](x)

def train(model, train_cfg, train_dataset, test_dataset, ood_dataset, device):
  """Train a standard classification model with either standard or VBLL models.
  """

  if train_cfg.VBLL:
    # for VBLL models, set weight decay to zero on last layer
    param_list = [
        {'params': model.params.in_layer.parameters(), 'weight_decay': train_cfg.WD},
        {'params': model.params.core.parameters(), 'weight_decay': train_cfg.WD},
        {'params': model.params.out_layer.parameters(), 'weight_decay': 0.}
    ]
  else:
    param_list = model.parameters()
    loss_fn = nn.CrossEntropyLoss() # define loss function only for non-VBLL model

  optimizer = train_cfg.OPT(param_list,
                            lr=train_cfg.LR,
                            weight_decay=train_cfg.WD)

  train_dataloader = DataLoader(train_dataset, batch_size = train_cfg.BATCH_SIZE, shuffle=True)
  val_dataloader = DataLoader(test_dataset, batch_size = train_cfg.BATCH_SIZE, shuffle=True)
  ood_dataloader = DataLoader(ood_dataset, batch_size = train_cfg.BATCH_SIZE, shuffle=True)

  output_metrics = {
      'train_loss': [],
      'val_loss': [],
      'train_acc': [],
      'val_acc': [],
      'ood_auroc': []
  }

  for epoch in range(train_cfg.NUM_EPOCHS):
    model.train()
    running_loss = []
    if train_cfg.VBLL and model.params.out_layer.return_empirical:
      output_metrics['train_loss_empirical'] = []
      running_loss_empirical = []
    running_acc = []

    for train_step, data in enumerate(train_dataloader):
      optimizer.zero_grad()
      x = data[0].to(device)
      y = data[1].to(device)

      out = model(x)
      if train_cfg.VBLL:
        loss = out.train_loss_fn(y)
        if model.params.out_layer.return_empirical:
          loss_empirical = out.train_loss_fn_empirical(y, train_cfg.N_SAMPLES)
          running_loss_empirical.append(loss_empirical.item())
        probs = out.predictive.probs
        acc = eval_acc(probs, y).item()
      else:
        loss = loss_fn(out, y)
        acc = eval_acc(out, y).item()

      running_loss.append(loss.item())
      running_acc.append(acc)

      if train_cfg.VBLL and model.params.out_layer.return_empirical and train_cfg.VBLL_EMPIRICAL and train_cfg.N_SAMPLES:
        loss_empirical.backward()
      else:
        loss.backward()
      optimizer.step()

    output_metrics['train_loss'].append(np.mean(running_loss))
    if train_cfg.VBLL and model.params.out_layer.return_empirical:
      output_metrics['train_loss_empirical'].append(np.mean(running_loss_empirical))
    output_metrics['train_acc'].append(np.mean(running_acc))

    if epoch % train_cfg.VAL_FREQ == 0:
      running_val_loss = []
      running_val_acc = []

      with torch.no_grad():
        model.eval()
        for test_step, data in enumerate(val_dataloader):
          x = data[0].to(device)
          y = data[1].to(device)

          out = model(x)
          if train_cfg.VBLL:
            loss = out.val_loss_fn(y)
            probs = out.predictive.probs
            acc = eval_acc(probs, y).item()
          else:
            loss = loss_fn(out, y)
            acc = eval_acc(out, y).item()

          running_val_loss.append(loss.item())
          running_val_acc.append(acc)

        output_metrics['val_loss'].append(np.mean(running_val_loss))
        output_metrics['val_acc'].append(np.mean(running_val_acc))
      output_metrics['ood_auroc'].append(eval_ood(model, val_dataloader, ood_dataloader, device, VBLL=train_cfg.VBLL))
      print('Epoch: {:2d}, train loss: {:4.4f}, train acc: {:4.4f}'.format(epoch, np.mean(running_loss), np.mean(running_acc)))
      if train_cfg.VBLL and model.params.out_layer.return_empirical:
        print('Epoch: {:2d}, train loss empirical: {:4.4f}'.format(epoch, np.mean(running_loss_empirical)))
      print('Epoch: {:2d}, valid loss: {:4.4f}, valid acc: {:4.4f}'.format(epoch, np.mean(np.mean(running_val_loss)), np.mean(np.mean(running_val_acc))))
  return output_metrics

def eval_acc(preds, y):
    map_preds = torch.argmax(preds, dim=1)
    return (map_preds == y).float().mean()

def eval_ood(model, ind_dataloader, ood_dataloader, device, VBLL=True):
    ind_preds = []
    ood_preds = []

    def get_score(out):
        if VBLL:
            score = out.ood_scores.detach().cpu().numpy()
        else:
            score = torch.max(out, dim=-1)[0].detach().cpu().numpy()
        return score

    for i, (x, y) in enumerate(ind_dataloader):
        x = x.to(device)
        out = model(x)
        ind_preds = np.concatenate((ind_preds, get_score(out)))

    for i, (x, y) in enumerate(ood_dataloader):
        x = x.to(device)
        out = model(x)
        ood_preds = np.concatenate((ood_preds, get_score(out)))

    labels = np.concatenate((np.ones_like(ind_preds)+1, np.ones_like(ind_preds)))
    scores = np.concatenate((ind_preds, ood_preds))
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=2)
    return metrics.auc(fpr, tpr)

def viz_performance(logs):
    """
    A visualization function that plots losses, accuracies, and out of
    distribution AUROC.

    logs: a dictionary, with keys corresponding to different model evals and values
    corresponding to dicts of results.
    """

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(24, 16))

    # get list of colors
    color = cm.rainbow(np.linspace(0, 1, len(logs)))

    for i, (k,v) in enumerate(logs.items()):
        # train and val loss
        axs[0].plot(v['train_loss'], label=k + ' (train)', color=color[i])
        axs[0].plot(v['val_loss'], label=k + ' (val)', linestyle = '--', color=color[i])
        if 'train_loss_empirical' in v.keys():
            plt.plot(v['train_loss_empirical'], label=k + ' (train empirical)', linestyle = 'dotted', color=color[i])
    axs[0].legend()
    axs[0].set_ylabel('Loss')
    axs[0].set_xlabel('Epoch')

    for i, (k,v) in enumerate(logs.items()):
        axs[1].plot([1 - x for x in v['train_acc']], label=k + ' (train)', color=color[i])
        axs[1].plot([1 - x for x in v['val_acc']], label=k + ' (val)', linestyle='--', color=color[i])

    axs[1].set_ylabel('Error rate')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()
    axs[1].semilogy()

    for i, (k,v) in enumerate(logs.items()):
        axs[2].plot(v['ood_auroc'], label=k, color=color[i])
    axs[2].legend()
    axs[2].set_ylabel('OOD AUROC')
    axs[2].set_xlabel('Epoch')
    return fig

def main():

    device = "cuda"

    class data_cfg:
        TRAIN = "mnist"
        TEST = "mnist"
        OOD = "fashion"

    train_dataset, test_dataset, ood_dataset = dict_to_data(data_cfg)

    class cfg:
        IN_FEATURES = 784
        HIDDEN_FEATURES = 128
        OUT_FEATURES = 10
        NUM_LAYERS = 2
        REG_WEIGHT = 1./train_dataset.__len__()
        PARAM = 'diagonal'
        SOFTMAX_BOUND = "jensen"
        RETURN_EMPIRICAL = True
        SOFTMAX_BOUND_EMPIRICAL = "montecarlo"
        RETURN_OOD = True
        PRIOR_SCALE = 1.

    gen_vbll_model = GenVBLLMLP(cfg()).to(device)

    class train_cfg:
        DATA = data_cfg()
        NUM_EPOCHS = 2
        BATCH_SIZE = 512
        LR = 3e-3
        WD = 1e-4
        OPT = torch.optim.AdamW
        CLIP_VAL = 1
        VAL_FREQ = 1
        VBLL = True
        VBLL_EMPIRICAL = True
        N_SAMPLES = 10

    outputs = {}
    outputs['GenVBLL'] = train(gen_vbll_model, train_cfg(), train_dataset, test_dataset, ood_dataset, device)
    f = viz_performance(outputs)
    f.savefig(f"testgen_emp{train_cfg.N_SAMPLES}.png")

    def logit_predictive_likedisc(model, x):
        x = x.view(x.shape[0], -1)
        x = model.params['in_layer'](x)

        for layer, ac in zip(model.params['core'], model.activations):
          x = ac(layer(x))
        
        return model.params['out_layer'].logit_predictive_likedisc(x)
    
    uncerts = logit_predictive_likedisc(gen_vbll_model, test_dataset.data[:100].to(device).to(torch.float32))
    uncerts = F.softmax(uncerts, dim=-1)
    print(uncerts.shape)
    uncerts = uncerts[:, :, 0].std(dim=0)
    print(uncerts.shape)

if __name__ == "__main__":
    main()