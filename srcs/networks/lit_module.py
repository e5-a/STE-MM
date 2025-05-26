import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
import torch
import numpy as np


class Model(pl.LightningModule):
    def __init__(self,
                 Net,
                 learning_rate,
                 input_shape,
                 num_classes,
                 weight_decay,
                 momentum,
                 lr_scheduler,
                 optimizer,
                 train_batches=None,
                 num_epochs=None,
                 **kwargs):
        super().__init__()
        self.lr = learning_rate
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.model = Net(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                **kwargs)
        self.train_acc = Accuracy(
                task="multiclass",
                num_classes=num_classes)
        self.val_acc = Accuracy(
                task="multiclass",
                num_classes=num_classes)
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.momentum = momentum
        self.num_epochs = num_epochs
        self.train_batches = train_batches

        assert self.lr_scheduler in ["", "cosine_w_warmup", "linear_up_down"]
        assert self.optimizer in ["Adam", "SGD"]

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.cross_entropy(out, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.train_acc(out, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        #self.log('lr', self.sch.get_last_lr()[0], on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        #print(x.shape, y.shape, out.shape)
        val_loss = F.cross_entropy(out, y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.val_acc(out, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer == "Adam":
            opt = torch.optim.Adam(self.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        elif self.optimizer == "SGD":
            opt = torch.optim.SGD(self.parameters(),
                                  lr=self.lr,
                                  momentum=self.momentum,
                                  weight_decay=self.weight_decay)

        ret = {'optimizer': opt}
        if self.lr_scheduler == 'linear_up_down':
            lr_sch = np.interp(np.arange(1 + self.num_epochs),
                               [0, 5, self.num_epochs],
                               [0, 1, 0])
            sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_sch.__getitem__)
            self.sch = sch
            ret['lr_scheduler'] = {'scheduler': sch, 'interval': 'epoch'}

        return ret
