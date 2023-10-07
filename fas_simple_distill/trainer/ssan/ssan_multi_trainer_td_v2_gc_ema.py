from functools import partial
import os
import pickle
import shutil
from argparse import Namespace
from collections import namedtuple
from copy import deepcopy
from collections.abc import Iterable
from pathlib import Path
from tabnanny import check
from typing import List, Sequence, Tuple

import mlflow
import numpy as np
import ray
import torch
import torch.nn as nn
import torchmetrics
import torch.backends.cudnn
import torch.optim
import torch.utils.data.distributed
import torch.nn.functional as F
import yaml
from addict import Dict as Adict
from fas_eval.evaluators import BinarySpoofEvaluator
from fas_simple_distill.data.webdatasets import general as general_webdataset
from fas_simple_distill.data.webdatasets.multidataset import MultiUnlimitedDataLoader
from fas_simple_distill.utils.config_utils import (
    parse_tuner_args_with_config,
    setconfig_dict,
)

from fas_simple_distill.model.ssan.ssan_r import SSAN_R

from fas_simple_distill.utils.parameters import check_module_has_params

from fas_simple_distill.data.datapipe.face_crop import (
    FaceAlignFromDetsIterDataPipe,
    FaceCropFromDetsIterDataPipe,
)
from fas_simple_distill.data.datapipe.mitigation import DictBlankImageSkipperDP
from fas_simple_distill.data.transform.labels import webdataset_label_transform

# from fas_simple_distill.utils.scheduler import (
#     cosine_anneal_schedule,
# )

from fas_simple_distill.ops.contrast_loss import ContrastLoss


from datatools.torch_data import webdataset as tdswds
from datatools.torch_data import datapipes as tdsdps

from datatools.torch_data.datapipes import (
    DictTransformIterDataPipe,
    IterDataLoader,
    MultiViewDictTransformIterDataPipe,
    ImageTransformIterDataPipe,
    WDSFilterCSVDataPipe
)
from datatools.torch_data.webdataset import (
    get_multidata_webdataset_datapipe_v2,
    get_webdataset_datapipe,
)

from fas_simple_distill.trainer.base_trainer import BaseTrainer
from fas_simple_distill.ops.magface import MagLinear
from fas_simple_distill.utils.model_ema import ModelEmaV3
from fas_simple_distill.utils.general import *  # pylint: disable=unused-wildcard-import,wildcard-import
from fas_simple_distill.utils.logging import *  # pylint: disable=unused-wildcard-import,wildcard-import
from fas_simple_distill.utils.metrics import *  # pylint: disable=unused-wildcard-import,wildcard-import
from fas_simple_distill.utils.raytune import trial_name_creator_v2
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from ray import tune
from torch.nn.utils import clip_grad_norm_
from torchmetrics.classification import accuracy
from torchvision.transforms import Compose
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.trial import Trial

SAGEMAKER_CHECKPOINT_PATH = "/opt/ml/checkpoints"
IMAGE = ".png"
TGT_LABEL = ".pickle"
DATA_LABEL = "data_lbl"

class SSANMultiTuneTrainer(BaseTrainer, nn.Module):
    def __init__(
        self, args: Namespace, config: Adict, checkpoint_dir: Optional[str] = None,
    ) -> None:
        super().__init__(args, config, checkpoint_dir)

        self.max_iter = self.config.hparams.max_iter



        self.load_checkpoint(checkpoint_dir)

        self._use_checkpointing = os.environ.get("USE_MODEL_CHECKPOINT")
        self._metric_means: Dict[str, torchmetrics.MeanMetric] = {}

        self.max_epoch: int = self.config.hparams.get("max_epoch", 0)
        if self.max_epoch and self.max_iter != -1:
            raise RuntimeError(
                "max_iter should be set to -1 if max_epoch is defined in config."
            )

    def _check_valid_live_class_labels(self, live_class_labels):
        if not isinstance(live_class_labels, Sequence):
            raise TypeError("Expected live_class_labels to be a sequence.")

        if any((type(_) != int for _ in live_class_labels)):
            raise TypeError(
                "All elements in live_class_labels must be a int denoting class label."
            )

        set_class_labels = set(live_class_labels)
        if len(set_class_labels) != len(live_class_labels):
            raise ValueError("No duplicate is allowed in live_class_labels.")

    def to(self, device, *args, **kwargs):
        self.device = device
        super().to(device, *args, **kwargs)

    def autoset_num_iter_per_epoch(self):
        if self.iter_per_epoch != 0:
            raise RuntimeError("iter_per_epoch is already set.")

        n_iters = 0
        for _ in deepcopy(self.train_loader):
            n_iters += 1

        print(f"iter_per_epoch is set to {n_iters}")
        self.iter_per_epoch = n_iters

    def build_model(self):
        self.define_criterions()
        self.model_ema = None
        
        self.model = SSAN_R(ada_num=2, max_iter=self.max_iter).to(self.device)
        model_ema = self.config.hparams.get("model_ema", None)
        if ema_kwargs := self.config.hparams.get("model_ema", None):
            print("Using ModelEmaV2")
            self.model_ema = object_from_dict(model_ema, model=self.model)

    def define_criterions(self):
        self.binary_func = nn.CrossEntropyLoss()
        self.contrast_loss = ContrastLoss()
        
        self.grad_accum = self.config.hparams.get("grad_accum", 1)
        if self.grad_accum <= 0 or not isinstance(self.grad_accum, int):
            raise RuntimeError("grad_accum must be an integer and more than 0.")

        if self.grad_accum > 1:
            print(f"Using gradient accumulation: {self.grad_accum} steps")


    def get_non_ema_parameters(self):
        return (p for n, p in self.named_parameters() if "model_ema" not in n)

    def _forward_impl(self, x, return_pred: bool = False):
        rand_idx = torch.randperm(x.shape[0])
        cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = self.model(x, x[rand_idx, :, :, :])

        logits = F.softmax(cls_x1_x1, dim=1)[..., 1].view(-1).float()

        retval = (logits, fea_x1_x1, fea_x1_x2, domain_invariant)

        return retval

    def forward(self, x, return_pred=False):
        return self._forward_impl(x, return_pred)

    def configure_optimizers(self):
        self.optimizer = object_from_dict(
            self.config.hparams.optimizer,
            params=self.get_non_ema_parameters(),
        )
        

        self.configure_schedulers()
        self.configure_gradscaler()

    def configure_schedulers(self):
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

    def configure_gradscaler(self):
        self.grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(**self.config.amp)

    def get_train_transform(self):
        train_transforms_configs = self.config.dataset.train_aug.transforms
        if not isinstance(train_transforms_configs, Sequence):
            raise TypeError("Transform config must be a sequence of dict!")
        transforms_list = [object_from_dict(t) for t in train_transforms_configs]
        train_transform = Compose(transforms_list)
        return train_transform

    def get_train_data_lbl(self, urls_rec):
        if self.config.dataset.train_dataset.use_custom_domain_label:
            return [u["dom_lbl"] for u in urls_rec]
        return None

    def initialize_train_loader(self):
        train_transform = self.get_train_transform()

        urls_rec = self.config.dataset.train_dataset.urls
        assert isinstance(urls_rec, list)
        urls = [u["url"] for u in urls_rec]
        data_lbl = self.get_train_data_lbl(urls_rec)
        
        train_datapipe = get_multidata_webdataset_datapipe_v2(
            dataset_urls=urls,
            data_labels=data_lbl,
            enforce_balance="largest",
            seed=self.config.hparams.seed,
            shuffle_per_epoch=self.config.hparams.get("shuffle_per_epoch", True),
        )
        
        train_datapipe = self._postprocess_train_datapipe(
            train_transform,
            train_datapipe,
        )

        train_loader = IterDataLoader(
            train_datapipe,
            batch_size=self.config.hparams.batch_size,
            num_workers=self.config.num_workers,
            drop_last=True,
        )

        if self.iter_per_epoch > 0:
            train_loader = train_loader.unlimited(periodic_limit=self.iter_per_epoch)

        self.train_loader = train_loader

    def _postprocess_train_datapipe(self, train_transform, train_datapipe):
        if crop_config := self.config.dataset.train_dataset.get("crop_config"):
            train_datapipe = FaceCropFromDetsIterDataPipe(train_datapipe, **crop_config)
        if align_config := self.config.dataset.train_dataset.get("align_config"):
            train_datapipe = FaceAlignFromDetsIterDataPipe(
                train_datapipe, **align_config
            )

        if not self.config.dataset.train_dataset.get("disable_blank_mitigation", False):
            print("Blank image mitigation active")
            train_datapipe = DictBlankImageSkipperDP(train_datapipe, img_key=".png")

        train_datapipe = DictTransformIterDataPipe(
            train_datapipe,
            {
                ".png": train_transform,
                ".pickle": webdataset_label_transform,
            },
#             exclude_keys=[".pickle"],
        )

        return train_datapipe

    def initialize_val_loaders(self):
        val_transform = self.get_val_transform()
        
        url_dict = self.config.dataset.val_dataset.urls
        assert isinstance(url_dict, list)

        VAL_LOADER = namedtuple("VAL_LOADER", ["loader", "name", "size"])

        self.val_loaders: List[Tuple[IterDataLoader, str, int]] = []
        for url in url_dict:
            val_loader = self.create_val_loader(val_transform, url)
            self.val_loaders.append(VAL_LOADER(val_loader, url["name"], url["size"]))

    def create_val_loader(self, val_transform, url):
        val_datapipe = get_webdataset_datapipe(
            urls=url["url"],
            shuffle_per_epoch=False,
            split_by_rank=False,
            split_by_worker=True,
            rank_splitter_enforce_equal=False,
        )

        if crop_config := self.config.dataset.val_dataset.get("crop_config"):
            val_datapipe = FaceCropFromDetsIterDataPipe(val_datapipe, **crop_config)
        if align_config := self.config.dataset.val_dataset.get("align_config"):
            val_datapipe = FaceAlignFromDetsIterDataPipe(
                val_datapipe, **align_config
            )

        if not self.config.dataset.val_dataset.get("disable_blank_mitigation", False):
            print("Blank image mitigation active")
            val_datapipe = DictBlankImageSkipperDP(val_datapipe, img_key=".png")

        val_datapipe = DictTransformIterDataPipe(
            val_datapipe,
            {
                ".png": val_transform,
                ".pickle": webdataset_label_transform,
            },
        )

        val_loader = IterDataLoader(
            val_datapipe,
            batch_size=self.config.hparams.batch_size,
            num_workers=self.config.num_workers,
            drop_last=False,
        )

        return val_loader

    def get_val_transform(self):
        val_transforms = [
            object_from_dict(t) for t in self.config.dataset.val_aug.transforms
        ]
        val_transform = Compose(val_transforms)
        return val_transform

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        if checkpoint_path is None:
            return

        ckpt_dict = torch.load(
            os.path.join(checkpoint_path, "ckpt.pth"), map_location="cpu"
        )
        self.model.load_state_dict(ckpt_dict["model"])

        if self.model_ema:
            self.model_ema.load_state_dict(ckpt_dict["model_ema"])

        if self.scheduler:
            self.scheduler.load_state_dict(ckpt_dict["scheduler"])

        self.optimizer.load_state_dict(ckpt_dict["optimizer"])

        self.current_epoch = ckpt_dict["current_epoch"]
        self.start_epoch = self.current_epoch
        self.global_step = ckpt_dict["global_step"]

    def save_checkpoint(self):
        ckpt_dict = {}
        if self.config.data_parallel:
            ckpt_dict["model"] = self.model.module.state_dict()
        else:
            ckpt_dict["model"] = self.model.state_dict()

        if self.model_ema:
            ckpt_dict["model_ema"] = self.model_ema.state_dict()
        ckpt_dict["optimizer"] = self.optimizer.state_dict()
#         ckpt_dict["scheduler"] = self.scheduler.state_dict() if self.scheduler else None
        ckpt_dict["current_epoch"] = self.current_epoch
        ckpt_dict["global_step"] = self.global_step

        with tune.checkpoint_dir(step=self.global_step) as checkpoint_dir:
            save_path = os.path.join(checkpoint_dir, "ckpt.pth")
            torch.save(ckpt_dict, save_path)

            folder_name = Path(save_path).parent.name
            mlflow.log_artifact(save_path, folder_name)

    def reset_metric_means(self):
        for v in self._metric_means.values():
            v.reset()

    def is_done_grad_accum(self):
        return (self.global_step + 1) % self.grad_accum== 0

    def on_epoch_step(self, batch_dict: Dict[str, torch.Tensor]):
#         print("BATCH DICT", batch_dict.keys())
        img: torch.Tensor = batch_dict[IMAGE].to(self.device, non_blocking=True)
        target: torch.Tensor = batch_dict[TGT_LABEL].to(self.device, non_blocking=True)
        lbl_metrics: torch.Tensor = batch_dict[DATA_LABEL].to(self.device, non_blocking=True)
        # lbl_metrics -= 1 # turn it into index

        if self.is_done_grad_accum():
            self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=self.device, enabled=self.config.amp.enabled):
            rand_idx = torch.randperm(img.shape[0])
            cls_x1_x1, fea_x1_x1, fea_x1_x2, domain_invariant = self.model(img, img[rand_idx, :, :, :])
            binary_loss = self.binary_func(cls_x1_x1, target)
            
            contrast_label = target.long() == target[rand_idx].long()
            contrast_label = torch.where(contrast_label==True,1,-1)
            contrast_loss = self.contrast_loss(fea_x1_x1, fea_x1_x2, contrast_label)
            
            adv_loss = self.binary_func(domain_invariant, lbl_metrics)
            
            train_loss = binary_loss + contrast_loss + adv_loss

            with torch.no_grad():
                logits = F.softmax(cls_x1_x1, dim=1)
                train_acc = torchmetrics.functional.accuracy(
                    logits[..., 1].view(-1).float(), target.view(-1) 
                )

        live_dist = torch.count_nonzero(target) / target.size(0)

        train_metrics = {
            "train_loss": train_loss,
            "contrast_loss": contrast_loss,
            "adv_loss": adv_loss,
            "train_live_dist": live_dist,
            "train_acc": train_acc,
        }

        self.run_backward(train_loss)
        if self.model_ema is not None:
#             self.model_ema.update(self)
            self.model_ema.update(self.model)

        return train_metrics

    def on_epoch_end(self):
        self.eval()
        val_log_dict = self.run_validation()

        self.save_checkpoint()

        tune.report(**val_log_dict)
        mlflow.log_metrics(val_log_dict, step=self.global_step)

        if self.sched_interval == "epoch" and self.scheduler:
            self.scheduler.step()

    def run_backward(self, train_loss):
        # self.model.zero_grad()
        # train_loss.backward()
        # self.optimizer.step()
        self.grad_scaler.scale(train_loss).backward()

        if self.is_done_grad_accum():
            self.grad_scaler.unscale_(self.optimizer)
            grad_norm_to_clip = self.config.hparams.clip_grad_norm
            if grad_norm_to_clip > 0.0:
                clip_grad_norm_(
                    self.get_non_ema_parameters(),
                    max_norm=grad_norm_to_clip,
                )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

    def run_training(self):
        if self.config.get("autoset_iter_per_epoch"):
            self.autoset_num_iter_per_epoch()

        print("Training started")
        self.training_loop()
        self.finish()

    def training_loop(self):
        self.train()

        if self.max_epoch:
            max_epoch = self.max_epoch
            self.max_iter = sys.maxsize
        else:
            max_epoch = self.max_iter // self.iter_per_epoch

        for epoch in range(self.current_epoch, max_epoch):
            self.current_epoch = epoch
            self.train()
            batch_dict: Dict[str, torch.Tensor]

            for batch_dict in self.train_loader:
                train_metrics = self.on_epoch_step(batch_dict)
                self.global_step += 1
                self.on_epoch_step_end(train_metrics)

                if self.global_step >= self.max_iter:
                    break

            self.on_epoch_end()

    def on_epoch_step_end(self, train_metrics):
        for k, v in train_metrics.items():
            if k not in self._metric_means.keys():
                self._metric_means[k] = torchmetrics.MeanMetric().to(self.device)

            self._metric_means[k].update(v)


        if self.global_step % self.config.log_every_n_step == 0:
            train_log_dict = {
                f"{k}_avg": v.compute().item() for k, v in self._metric_means.items()
            }
            train_log_dict["epoch"] = self.current_epoch
            train_log_dict["optim_lr"] = self.optimizer.param_groups[0]["lr"]
            mlflow.log_metrics(train_log_dict, step=self.global_step)
            print("currently on step: ", self.global_step, train_log_dict)

            self.reset_metric_means()

        if self.sched_interval == "step" and self.scheduler:
            self.scheduler.step()

    @torch.no_grad()
    def run_validation(self, *, _sanity_check=False):
        self.eval()
        all_log_dicts = {}
        for val_loader, val_name, val_size in self.val_loaders:
            print("Evaluating {}".format(val_name))
            targets = []
            val_preds = []

            batch_dict: Dict[str, torch.Tensor]
            for batch_dict in val_loader:
                with torch.autocast(
                    device_type=self.device, enabled=self.config.amp.enabled
                ):
                    val_step_ret = self.on_val_step(batch_dict)

                targets.append(val_step_ret["target"].cpu())
                val_preds.append(val_step_ret["preds"].cpu())

            targets = torch.cat(targets).numpy()
            val_preds = torch.cat(val_preds).numpy()

            if _sanity_check:
                if val_size != targets.shape[0]:
                    raise RuntimeError(
                        "Sanity check failed: mismatch number of samples "
                        f"(expected {val_size}, got {targets.shape[0]}) on {val_name}"
                    )

            metrics_val = self.calculate_val_metrics(val_preds, targets)

            val_log_dict = self.create_val_log_dict(
                val_name, metrics_val
            )

            all_log_dicts.update(val_log_dict)

        all_log_dicts["epoch"] = self.current_epoch
        return all_log_dicts

    def create_val_log_dict(self, val_name, metrics_val):
        val_log_dict = {}
        val_log_dict.update(
            {
                f"{val_name}_threshold": metrics_val["THRESHOLD"],
                f"{val_name}_AUC": metrics_val["ROC_AUC"] * 100,
                f"{val_name}_EER": metrics_val["EER"] * 100,
                f"{val_name}_ACER": metrics_val["ACER"] * 100,
                f"{val_name}_APCER": metrics_val["APCER"] * 100,
                f"{val_name}_NPCER": metrics_val["NPCER"] * 100,
                f"{val_name}_NPCER_at_APCER5e-2": metrics_val["NPCER@APCER5%"]
                * 100,
                f"{val_name}_NPCER_at_APCER1e-2": metrics_val["NPCER@APCER1%"]
                * 100,
                f"{val_name}_NPCER_at_APCER5e-3": metrics_val["NPCER@APCER0.5%"]
                * 100,
                f"{val_name}_NPCER_at_APCER0": metrics_val["NPCER_at_APCER0"]
                * 100,
                f"{val_name}_THRESH_at_APCER0": metrics_val["THRESH_at_APCER0"],
                f"{val_name}_ACC": metrics_val["ACC"],
            }
        )

        return val_log_dict

    def calculate_val_metrics(self, val_pred, targets):
        evaluator = BinarySpoofEvaluator(val_pred, targets, interp="nearest")
        evaluator.calculate_metrics(threshold_from="eer")
        metrics = evaluator.metrics

        eer_curve = metrics["EER_CURVE"]
        npcer_at_apcer0, thresh_at_apcer0 = get_npcer_at_apcer(
            0.0,
            npcer=eer_curve["FRR"],
            apcer=eer_curve["FAR"],
            thresholds=eer_curve["THRESHOLDS"],
        )

        metrics["NPCER_at_APCER0"] = npcer_at_apcer0
        metrics["THRESH_at_APCER0"] = thresh_at_apcer0
        return metrics

    def on_val_step(self, batch_dict: Dict[str, torch.Tensor]):
        img: torch.Tensor = batch_dict[IMAGE].to(self.device, non_blocking=True)
        target: torch.Tensor = batch_dict[TGT_LABEL].to(self.device, non_blocking=True)
        
        model = self
        if self.model_ema is not None:
            print("model ema used in val")
            model = self.model_ema
        cls_x1_x1, _, _, _ = model.module(img, img)
        print("cls_x1_x1 shape:", cls_x1_x1.shape)

        preds = F.softmax(cls_x1_x1, dim=1)[:, 1]


        val_step_ret = {
            "target": target,
            "preds": preds,
        }

        return val_step_ret

    def run_sanity_check(self):
        print("Running sanity check: pass 1")
        first_pass_result = self.run_validation(_sanity_check=True)
        print("Running sanity check: pass 2")
        second_pass_result = self.run_validation(_sanity_check=True)

        for k, v in first_pass_result.items():
            if v != second_pass_result[k]:
                raise RuntimeError(
                    f"The value of {k} on the first pass ({v}) doesn't match with the value of second pass ({second_pass_result[k]})"
                )


    def finish(self):
        print("DONE TRAINING!")


def ssan_multi_tuning(
    configs: dict, checkpoint_dir: Optional[str] = None, tune_config_path=None,
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

    trainer = SSANMultiTuneTrainer(args, config, checkpoint_dir)
#     if config.get("run_sanity_check", False):
#         trainer.run_sanity_check()
    trainer.run_training()


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

    _ = tune.run(
        mlflow_mixin(partial(ssan_multi_tuning, tune_config_path=tune_config_path)),
        name=args.run_name,
        trial_name_creator=partial(trial_name_creator_v2, args.run_name),
        search_alg=tuning_config["algo"],
        scheduler=tuning_config["scheduler"],
        config=run_config,
        local_dir=local_dir,
        fail_fast=True,
        **tuning_config["tuning_config"].run_kwargs,
        resume="AUTO",
        reuse_actors=True,
        sync_config=tune.SyncConfig(syncer=None, sync_on_checkpoint=False),
    )

    print("finished")


if __name__ == "__main__":
    _main()
