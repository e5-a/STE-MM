import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
from utils.seed import fix_randomseed
from networks.lit_module import Model
from networks import mlp, VGG, resnet20
from datasets import MNIST, CIFAR10, FMNIST
from utils.idgenerator import IDGenerator
import sys


@hydra.main(version_base=None, config_name='config_train', config_path='../configs')
def main(cfg: DictConfig):
    stg = cfg.train
    identity = None
    out_dir = str(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    fix_randomseed(seed=stg.seed)
    dm = eval(stg.dataset.name).DataModule(
            batch_size=stg.batch_size,
            num_workers=cfg.machine.num_workers,
            **OmegaConf.to_container(stg.dataset, resolve=True))
    dm.setup()

    fix_randomseed(seed=stg.seed)
    model = Model(
            Net=eval(stg.network.name).Net,
            learning_rate=stg.learning_rate,
            num_epochs=stg.epochs,
            train_batches=len(dm.train_dataloader()),
            weight_decay=stg.weight_decay,
            momentum=stg.momentum,
            optimizer=stg.optimizer,
            lr_scheduler=stg.lr_scheduler,
            **OmegaConf.to_container(stg.network, resolve=True))

    callbacks = [pl.callbacks.ModelCheckpoint(
        dirpath=f'{out_dir}/checkpoints',
        filename='{epoch}',
        mode='max',
        monitor='val_acc_epoch',
        save_weights_only=True,
        save_top_k=1)]
    loggers = []
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         max_epochs=stg.epochs,
                         callbacks=callbacks,
                         logger=loggers,
                         precision='bf16-mixed')
                         #strategy='ddp')

    trainer.fit(model, dm)


if __name__ == "__main__":
    id_generator = IDGenerator()
    OmegaConf.register_new_resolver("experiment_id", id_generator.generate)
    main()
