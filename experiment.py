import os
from torch import optim
from models import BaseVAE
from models.types_ import *
import pytorch_lightning as pl
import torchvision.utils as vutils


def get_optimizer(parameters, config):
    if config['type'] == "AdamW":
        return optim.AdamW(parameters, **config['kwargs'])
    elif config['type'] == "Adam":
        return optim.Adam(parameters, **config['kwargs'])
    elif config['type'] == "SGD":
        return optim.SGD(parameters, **config['kwargs'])
    else:
        raise NotImplementedError


def get_scheduler(optimizer, config):
    if config['type'] == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(optimizer, **config['kwargs'])
    elif config['type'] == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, **config['kwargs'])
    elif config['type'] == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, **config['kwargs'])
    else:
        raise NotImplementedError


class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.
            params['kld_weight'],  #al_img.shape[0]/ self.num_train_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx)

        self.log_dict({key: val.item()
                       for key, val in train_loss.items()},
                      sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,  #real_img.shape[0]/ self.num_val_imgs,
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx)

        self.log_dict(
            {f"val_{key}": val.item()
             for key, val in val_loss.items()},
            sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(
            iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(
            recons.data,
            os.path.join(
                self.logger.log_dir, "Reconstructions",
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
            normalize=True,
            nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(
                    self.logger.log_dir, "Samples",
                    f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                normalize=True,
                nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = get_optimizer(self.model.parameters(),
                                  self.params['optimizer'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['optimizer_2'] is not None:
                optimizer2 = get_optimizer(
                    getattr(self.model, self.params['submodel']).parameters(),
                    self.params['optimizer_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['lr_scheduler'] is not None:
                scheduler = get_scheduler(optims[0],
                                          self.params['lr_scheduler'])
                scheds.append(scheduler)
                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['lr_scheduler_2'] is not None:
                        scheduler2 = get_scheduler(
                            optims[1], self.params['lr_scheduler_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
