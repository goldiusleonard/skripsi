from functools import partial
import os
import pickle
import shutil
from argparse import Namespace
from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path
from typing import List

import mlflow
import numpy as np
import ray
import torch
import torch.backends.cudnn
import torch.optim
import torch.utils.data.distributed
import yaml
from addict import Dict as Adict
from fas_eval.evaluators import BinarySpoofEvaluator
from fas_simple_distill.data.webdatasets import general as general_webdataset
from fas_simple_distill.data.webdatasets.multidataset import MultiUnlimitedDataLoader
from fas_simple_distill.model.ssan.ssan_r import SSAN_R
from fas_simple_distill.utils.config_utils import (
    parse_tuner_args_with_config,
    setconfig_dict,
)

import fas_simple_distill.ops.finetune_loss as finetune_loss

from fas_simple_distill.ops.magface import MagLinear
from fas_simple_distill.utils.general import *  # pylint: disable=unused-wildcard-import,wildcard-import
from fas_simple_distill.utils.logging import *  # pylint: disable=unused-wildcard-import,wildcard-import
from fas_simple_distill.utils.metrics import *  # pylint: disable=unused-wildcard-import,wildcard-import
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from ray import tune
from torch.nn.utils import clip_grad_norm_
from torchmetrics.classification import accuracy
from torchvision.transforms import Compose
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.trial import Trial

import torch.nn as nn
from fas_simple_distill.ops.contrast_loss import ContrastLoss

import torch.nn.functional as F

SAGEMAKER_CHECKPOINT_PATH = "F:/skripsi/FAS-Skripsi-4"


def trial_name_creator(run_name, trial: Trial):
    env_name = 0
    if "env" in trial.config:
        env_name = trial.config["env"]
        if isinstance(env_name, type):
            env_name = env_name.__name__

    identifier = f"{run_name}__{env_name}__{trial.trial_id}"
    identifier.replace("/", "_")
    return identifier


class SSANBinaryTuneTrainer:
    def __init__(
        self,
        args: Namespace,
        config: Adict,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.args = args
        self.config = Adict(config)
        self.config.freeze(True)

        seed_all(self.config.hparams.seed)

        if self.config.use_cpu:
            self.device = "cpu"
        else:
            self.device = "cuda"

        self.start_epoch: int = 0
        self.current_epoch: int = self.start_epoch
        self.global_step: int = 0
        self.iter_per_epoch: int = self.config.hparams.iter_per_epoch
        self.max_iter: int = self.config.hparams.max_iter

        self.initialize_train_loader()
        self.initialize_val_loaders()
        self.build_model()
        self.configure_optimizers()

        self.load_checkpoint(checkpoint_dir)

    def build_model(self):
        self.model = SSAN_R(ada_num=2, max_iter=4000).to(self.device)

    def configure_optimizers(self):
        # --------------------------------- optimizer -------------------------------- #
        param_groups = [
            {"params": self.model.parameters()},
        ]
        self.binary_func = nn.CrossEntropyLoss()
        self.contrast_loss = ContrastLoss()
        self.optimizer = object_from_dict(
            self.config.hparams.optimizer, params=param_groups
        )
        self.scheduler = None
        self.sched_interval = None
        sched_conf = self.config.hparams.get("scheduler")
        if sched_conf:
            self.scheduler = object_from_dict(sched_conf, optimizer=self.optimizer)
            self.sched_interval = self.config.hparams.scheduler_interval
            if self.sched_interval not in ["step", "epoch"]:
                raise ValueError(
                    f"scheduler_interval must be either `step` or `epoch`, got {self.sched_interval}"
                )
    
    def initialize_train_loader(self):
        
        def get_label(annodict: dict):
            lbl = annodict.get("label")
            if lbl is None:
                lbl = annodict.get("labels")

            if lbl is None:
                raise RuntimeError(f"Cannot get label key, available keys are {annodict.keys()}")

            return int(lbl)
        
        train_transforms_configs = self.config.dataset.train_aug.transforms
        if isinstance(train_transforms_configs, (list, tuple)):
            transforms_list = [
                object_from_dict(t) for t in self.config.dataset.train_aug.transforms
            ]
        else:
            transforms_list = [object_from_dict(train_transforms_configs)]

        train_transform = Compose(transforms_list)
        tgt_transform = get_label
#         tgt_transform = lambda tgt_dict: int(tgt_dict["label"])

        urls_rec = self.config.dataset.train_dataset.urls
        assert isinstance(urls_rec, list)
        urls = [u["url"] for u in urls_rec]

        self.use_custom_domain_lbls = (
            self.config.dataset.train_dataset.use_custom_domain_label
        )
        assert self.use_custom_domain_lbls

        custom_domain_lbls = [u["dom_lbl"] for u in urls_rec]
        self.num_live_domains = sum([el <= 0 for el in custom_domain_lbls])

        self.train_loader = MultiUnlimitedDataLoader(
            urls=urls,
            num_workers=self.config.num_workers,
            batch_size=self.config.hparams.batch_size,
            transforms=[train_transform for _ in range(len(urls))],
            target_transforms=[tgt_transform for _ in range(len(urls))],
            custom_dom_lbl=custom_domain_lbls,
        )

    def initialize_val_loaders(self):
        
        def get_label(annodict: dict):
            lbl = annodict.get("label")
            if lbl is None:
                lbl = annodict.get("labels")

            if lbl is None:
                raise RuntimeError(f"Cannot get label key, available keys are {annodict.keys()}")

            return int(lbl)
        
        val_transforms = [
            object_from_dict(t) for t in self.config.dataset.val_aug.transforms
        ]
        val_transform = Compose(val_transforms)
        val_tgt_transform = get_label
#         val_tgt_transform = lambda tgt_dict: int(tgt_dict["label"])

        url_dict = self.config.dataset.val_dataset.urls
        assert isinstance(url_dict, list)

        VAL_LOADER = namedtuple("VAL_LOADER", ["loader", "name"])

        self.val_loaders: List[VAL_LOADER] = []
        for url in url_dict:
            loader = general_webdataset.get_multi_loader(
                urls=url["url"],
                dataset_size=url["size"],
                batch_size=self.config.hparams.batch_size,
                num_workers=self.config.num_workers,
                transform=val_transform,
                target_transform=val_tgt_transform,
                partial_batch=True,
                ddp_equalize=False,
                shuffle=False,
                split_by_node=False,
            )
            self.val_loaders.append(VAL_LOADER(loader, url["name"]))

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path is None:
            return

        ckpt_dict = torch.load(
            os.path.join(checkpoint_path, "ckpt.pth"), map_location="cpu"
        )
        self.model.load_state_dict(ckpt_dict["model"])

        if self.model_ema:
            self.model_ema.load_state_dict(ckpt_dict["model_ema"])

        self.classifier.load_state_dict(ckpt_dict["classifier"])
        self.discriminator.load_state_dict(ckpt_dict["discriminator"])
        self.metric_loss.load_state_dict(ckpt_dict["metric_loss"])

        if self.scheduler:
            self.scheduler.load_state_dict(ckpt_dict["scheduler"])

        self.optimizer.load_state_dict(ckpt_dict["optimizer"])

        self.current_epoch = ckpt_dict["current_epoch"]
        self.start_epoch = self.current_epoch
        self.global_step = ckpt_dict["global_step"]
        
        if(self.config.hparams.metric_loss.loss_type == "MagFace"):
            ckpt_dict["maglinear"] = self.maglinear.state_dict()

    def save_checkpoint(self):
        ckpt_dict = {}
        if self.config.data_parallel:
            ckpt_dict["model"] = self.model.module.state_dict()
        else:
            ckpt_dict["model"] = self.model.state_dict()

        if self.model_ema:
            ckpt_dict["model_ema"] = self.model_ema.state_dict()

        ckpt_dict["classifier"] = self.classifier.state_dict()
        ckpt_dict["discriminator"] = self.discriminator.state_dict()
        ckpt_dict["metric_loss"] = self.metric_loss.state_dict()
        ckpt_dict["optimizer"] = self.optimizer.state_dict()
        ckpt_dict["scheduler"] = self.scheduler.state_dict() if self.scheduler else None
        ckpt_dict["current_epoch"] = self.current_epoch
        ckpt_dict["global_step"] = self.global_step
        
        if(self.config.hparams.metric_loss.loss_type == "MagFace"):
            ckpt_dict["maglinear"] = self.maglinear.state_dict()

        with tune.checkpoint_dir(step=self.global_step) as checkpoint_dir:
            save_path = os.path.join(checkpoint_dir, "ckpt.pth")
            torch.save(ckpt_dict, save_path)

            folder_name = Path(save_path).parent.name
            mlflow.log_artifact(save_path, folder_name)

    def set_modules_train(self):
        self.model.train()
        self.classifier.train()
        self.discriminator.train()
        self.metric_loss.train()
        if(self.config.hparams.metric_loss.loss_type == "MagFace"):
            self.maglinear.train()

    def set_modules_eval(self):
        self.model.eval()
        self.classifier.eval()
        self.discriminator.eval()
        self.metric_loss.eval()
        if(self.config.hparams.metric_loss.loss_type == "MagFace"):
            self.maglinear.eval()

    def train(self):
        print("Training started")
        self.set_modules_train()

        train_loss_meter = AverageMeter("train_loss")
        binary_loss_meter = AverageMeter("binary_loss")
        constra_loss_meter = AverageMeter("constra_loss")
        adv_loss_meter = AverageMeter("adv_loss")
        
        train_acc_meter = AverageMeter("train_acc")
        live_dist_meter = AverageMeter("live_dist")
        acc_calculator = accuracy.Accuracy(threshold=0.2, compute_on_step=True).to(
            self.device
        )
        brd_acc_calculator = accuracy.Accuracy(threshold=0.5, compute_on_step=True).to(
            self.device
        )

        self.train_loader = iter(  # pylint: disable=attribute-defined-outside-init
            self.train_loader
        )

        while self.global_step <= self.max_iter:
            batch_data = next(self.train_loader)
            if self.use_custom_domain_lbls:
                img, target, data_idx, lbl_metrics = batch_data
            else:
                img, target, data_idx = batch_data
            img: torch.Tensor = img.to(self.device, non_blocking=True)
            target: torch.Tensor = target.to(self.device, non_blocking=True)
            data_idx: torch.Tensor = data_idx.to(self.device, non_blocking=True)
            lbl_metrics: torch.Tensor = lbl_metrics.to(self.device, non_blocking=True)

            live_dist = torch.count_nonzero(target) / target.size(0)

            self.optimizer.zero_grad()
            with torch.autocast(
                device_type=self.device, enabled=self.config.amp.enabled
            ):
                rand_idx = torch.randperm(img.shape[0])
                cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = self.model(img, img[rand_idx, :, :, :])
                binary_loss = nn.CrossEntropyLoss(cls_x1_x1, target.type_as(cls_x1_x1).view_as(cls_x1_x1))
                
                contrast_label = target[:,0].long() == target[rand_idx, 0].long()
                contrast_label = torch.where(contrast_label==True,1,-1)
                constra_loss = ContrastLoss(fea_x1_x1, fea_x1_x2, contrast_label)
                
                adv_loss = nn.CrossEntropyLoss(domain_invariant, lbl_metrics)
                
                train_loss = binary_loss + constra_loss + adv_loss
                
            train_acc = acc_calculator(
                F.softmax(cls_x1_x1, dim=1)[..., 1].view(-1).float(), target.view(-1)
            )
            
            # ------------------------- running metric statistics ------------------------ #
            minibatch_size = img.shape[0]
            binary_loss_meter.update(binary_loss.item(), n=minibatch_size)
            constra_loss_meter.update(constra_loss.item(), n=minibatch_size)
            adv_loss_meter.update(adv_loss.item(), n=minibatch_size)
            train_loss_meter.update(train_loss.item(), n=minibatch_size)
            train_acc_meter.update(train_acc.item(), n=minibatch_size)
            live_dist_meter.update(live_dist.item(), n=minibatch_size)
            # --------------------------------- backprop --------------------------------- #
<<<<<<< HEAD:fas_simple_distill/trainer/ssan_multi_tuner_finetune.py
            self.model.zero_grad()
            train_loss.backward()
            self.optimizer.step()
=======
            self.grad_scaler.scale(train_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm_to_clip = self.config.hparams.clip_grad_norm
            if grad_norm_to_clip > 0.0:
                clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=grad_norm_to_clip,
                )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            if self.model_ema:
                self.model_ema.update(self.model)

>>>>>>> dev-agi-research:fas_simple_distill/trainer/drdfl_liveinfo_trainer.py
            # -------------------------- validation and logging -------------------------- #
            self.global_step += 1
            if self.global_step % self.config.log_every_n_step == 0:
                print("currently on step: ", self.global_step)
            
            if self.global_step % self.iter_per_epoch == 0:
                log_dict = {
                    "train_loss_avg": train_loss_meter.avg,
                    "binary_loss_avg": binary_loss_meter.avg,
                    "constra_loss_avg": constra_loss_meter.avg,
                    "adv_loss_avg": adv_loss_meter.avg,
                    "train_acc_avg": train_acc_meter.avg,
                    "live_dist_avg": live_dist_meter.avg,
                    "epoch": self.current_epoch
                }
                
                self.set_modules_eval()
                val_log_dict = self.run_validation()
                log_dict.update(val_log_dict)
                
                self.save_checkpoint
                
                self.set_modules_train()
                self.current_epoch += 1
                
                if self.sched_interval == "epoch" and self.scheduler:
                    self.scheduler.step()
            if self.sched_interval == "step" and self.scheduler:
                self.scheduler.step()

    @torch.no_grad()
    def run_validation(self):
        all_log_dicts = {}
        for val_loader in self.val_loaders:
            print("Evaluating {}".format(val_loader.name))
            val_classif_losses: List[torch.Tensor] = []
            predictions = []
            brd_predictions = []
            targets = []

            for img, target in val_loader.loader:
                img: torch.Tensor = img.to(self.device, non_blocking=True)
                target: torch.Tensor = target.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device, enabled=self.config.amp.enabled
                ):
                    rand_idx = torch.randperm(img.shape[0])
                    cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = self.model(img, img[rand_idx, :, :, :])

                    # because this trainer uses binary, target should be the same shape as cls_out

                    classif_loss = self.criterion_cls(cls_x1_x1, target)
                    val_classif_losses.append(classif_loss.item())

                preds = F.softmax(cls_x1_x1, dim=1)[..., 1]
                predictions.append(preds.cpu())
                
                targets.append(target.cpu())

            val_classif_loss = np.mean(val_classif_losses)
            val_loss = val_classif_loss

            predictions = torch.cat(predictions).cpu().numpy()
            targets = torch.cat(targets).cpu().numpy()
            
            brd_predictions = torch.cat(brd_predictions).cpu().numpy()

            evaluator = BinarySpoofEvaluator(predictions, targets, interp="nearest")
            evaluator.calculate_metrics(threshold_from="eer")
            metrics = evaluator.metrics
            eer_curve = metrics["EER_CURVE"]

            acc = metrics["ACC"]

            npcer_at_apcer0, thresh_at_apcer0 = get_npcer_at_apcer(
                0.0,
                npcer=eer_curve["FRR"],
                apcer=eer_curve["FAR"],
                thresholds=eer_curve["THRESHOLDS"],
            )
            

            val_log_dict = {
                f"{val_loader.name}_loss": val_loss,
                f"{val_loader.name}_threshold": metrics["THRESHOLD"],
                f"{val_loader.name}_AUC": metrics["ROC_AUC"] * 100,
                f"{val_loader.name}_EER": metrics["EER"] * 100,
                f"{val_loader.name}_ACER": metrics["ACER"] * 100,
                f"{val_loader.name}_APCER": metrics["APCER"] * 100,
                f"{val_loader.name}_NPCER": metrics["NPCER"] * 100,
                f"{val_loader.name}_NPCER_at_APCER5e-2": metrics["NPCER@APCER5%"] * 100,
                f"{val_loader.name}_NPCER_at_APCER1e-2": metrics["NPCER@APCER1%"] * 100,
                f"{val_loader.name}_NPCER_at_APCER5e-3": metrics["NPCER@APCER0.5%"]
                * 100,
                f"{val_loader.name}_NPCER_at_APCER0": npcer_at_apcer0 * 100,
                f"{val_loader.name}_THRESH_at_APCER0": thresh_at_apcer0,
                f"{val_loader.name}_ACC": acc,
            }
            all_log_dicts.update(val_log_dict)

        all_log_dicts["epoch"] = self.current_epoch
        return all_log_dicts

    def finish(self):
        print("DONE TRAINING!")


# @mlflow_mixin
<<<<<<< HEAD:fas_simple_distill/trainer/ssan_multi_tuner_finetune.py
def ssan_binary_tuning(
    configs: dict, checkpoint_dir: Optional[str] = None, tune_config_path=None,
=======
def drdfl_liveinfo_train(
    configs: dict,
    checkpoint_dir: Optional[str] = None,
    tune_config_path=None,
>>>>>>> dev-agi-research:fas_simple_distill/trainer/drdfl_liveinfo_trainer.py
):
    args = {}
    for k, v in configs.items():
        if "_args." in k:
            args[k] = v

    args = Namespace(**args)

    config_path = configs.pop("_config_path")
    with open(config_path) as f:
        config = Adict(yaml.load(f, Loader=yaml.SafeLoader))

    for k, v in configs.items():
        setconfig_dict(config, k, v, delim=".", verbose=False)

    mlflow.log_params(flatten_dict(config.hparams))
    mlflow.set_tags(flatten_dict(config.mlflow_tags))
    mlflow.log_artifact(config_path, "config")
    mlflow.log_artifact(tune_config_path, "config")

    trainer = SSANBinaryTuneTrainer(args, config, checkpoint_dir)
    trainer.train()
    trainer.finish()


def _main():
    args, config, tuning_config = parse_tuner_args_with_config()

    config.mlflow_tags.repo_commit = get_head_commit()
    config.mlflow_tags.task = "face antispoofing"
    config.mlflow_tags.dl_framework = "pytorch"
    config.mlflow_tags.source = Path(__file__).name

    torch.backends.cudnn.benchmark = config.benchmark
    torch.backends.cudnn.deterministic = config.deterministic

    os.environ["MLFLOW_TRACKING_USERNAME"] = config.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config.mlflow_tracking_password
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config.mlflow_experiment_name

    local_dir = args.checkpoint_path
    if not local_dir:
        local_dir = str(Path(SAGEMAKER_CHECKPOINT_PATH) / "ray_results")
    Path(local_dir).mkdir(exist_ok=True, parents=True)
    Path(local_dir).joinpath(args.run_name).mkdir(exist_ok=True, parents=True)

    config_path = os.path.join(local_dir, args.run_name, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, Dumper=yaml.SafeDumper)

    tune_config_path = os.path.join(local_dir, args.run_name, "tune_config.yml")
    shutil.copy(
        args.tune_config_path, tune_config_path
    )

    run_config = tuning_config["search_space"]
    run_config.update({f"_args.{k}": v for k, v in args.__dict__.items()})
    run_config["_config_path"] = config_path
    run_config["mlflow"] = {
        "tracking_uri": config.mlflow_tracking_uri,
        "experiment_name": config.mlflow_experiment_name,
    }

    ray.init(**tuning_config["tuning_config"].init_kwargs)

    analysis = tune.run(
        mlflow_mixin(partial(ssan_binary_tuning, tune_config_path=tune_config_path)),
        name=args.run_name,
        trial_name_creator=partial(trial_name_creator, args.run_name),
        search_alg=tuning_config["algo"],
        scheduler=tuning_config["scheduler"],
        config=run_config,
        local_dir=local_dir,
        fail_fast=True,
        **tuning_config["tuning_config"].run_kwargs,
        resume="AUTO",
    )

    with open(os.path.join(local_dir, args.run_name, "analysis.pkl"), "wb") as analyf:
        pickle.dump(analysis, analyf)

    print("finished")


if __name__ == "__main__":
    _main()
