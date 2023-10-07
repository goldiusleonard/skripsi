from functools import partial
import os
import shutil
from argparse import Namespace
from collections import namedtuple
from pathlib import Path
from typing import List, Sequence, Tuple

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
from datatools.torch_data.datapipes import DictTransformIterDataPipe, IterDataLoader
from datatools.torch_data.webdataset import get_multidata_webdataset_datapipe_v2, get_webdataset_datapipe_v1
from fas_simple_distill.data.datapipe.face_crop import FaceAlignFromDetsIterDataPipe, FaceCropFromDetsIterDataPipe
from fas_simple_distill.data.datapipe.mitigation import DictBlankImageSkipperDP, DictToPILIterDataPipe
from fas_simple_distill.data.transform.labels import webdataset_label_transform
from fas_simple_distill.utils.config_utils import (
    parse_tuner_args_with_config,
    setconfig_dict,
)

from fas_simple_distill.utils.general import *  # pylint: disable=unused-wildcard-import,wildcard-import
from fas_simple_distill.utils.logging import *  # pylint: disable=unused-wildcard-import,wildcard-import
from fas_simple_distill.utils.metrics import *  # pylint: disable=unused-wildcard-import,wildcard-import
from iglovikov_helper_functions.config_parsing.utils import object_from_dict
from ray import tune
from torchmetrics.classification import accuracy
from torchvision.transforms import Compose
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.trial import Trial

SAGEMAKER_CHECKPOINT_PATH = "F:/skripsi/FAS-Skripsi-4"
IMG_KEY = ".png"
TGT_KEY = ".pickle"
DOM_LBL_KEY = "data_lbl"

def trial_name_creator(run_name, trial: Trial):
    env_name = 0
    if "env" in trial.config:
        env_name = trial.config["env"]
        if isinstance(env_name, type):
            env_name = env_name.__name__

    identifier = f"{run_name}__{env_name}__{trial.trial_id}"
    identifier.replace("/", "_")
    return identifier

class SSDGBinaryTuneTrainer:
    def __init__(
        self, args: Namespace, config: Adict, checkpoint_dir: Optional[str] = None,
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
        self.max_epoch: int = self.config.hparams.get("max_epoch", 0)

        self.initialize_train_loader()
        self.initialize_val_loaders()
        self.build_model()
        self.configure_optimizers()

        self.load_checkpoint(checkpoint_dir)

    def build_model(self):
        # --------------------------- model initialization --------------------------- #
        self.model = object_from_dict(self.config.hparams.model).to(self.device)
        if self.config.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

        self.discriminator = object_from_dict(
            self.config.hparams.discriminator, num_classes=self.num_live_domains
        ).to(self.device)

        self.metric_loss = object_from_dict(self.config.hparams.HardTripletLoss).to(
            self.device
        )

        # -------------------------------- criterions -------------------------------- #
        self.criterion_cls = object_from_dict(self.config.hparams.criterion_cls).to(self.device)
        self.criterion_dom = object_from_dict(self.config.hparams.criterion_dom).to(self.device)

    def configure_optimizers(self):
        # --------------------------------- optimizer -------------------------------- #
        param_groups = [
            {"params": filter(lambda p: p.requires_grad, self.model.parameters()), "lr": self.config.hparams.init_lr},
            {"params": filter(lambda p: p.requires_grad, self.discriminator.parameters()), "lr": self.config.hparams.init_lr},
        ]

        self.optimizer = object_from_dict(
            self.config.hparams.optimizer, params=param_groups
        )
        self.init_param_lr = []
        for param_group in self.optimizer.param_groups:
            self.init_param_lr.append(param_group["lr"])

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

        self.use_custom_domain_lbls = self.config.dataset.train_dataset.get(
            "use_custom_domain_label", False
        )

        urls_rec = self.config.dataset.train_dataset.urls
        assert isinstance(urls_rec, list)
        urls = [u["url"] for u in urls_rec]

        data_lbl = self.get_train_data_lbl(urls_rec)
        self.num_live_domains = self.config.hparams.num_live_domains

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
            drop_last=False,
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

        train_datapipe = DictToPILIterDataPipe(
            train_datapipe,
            img_key=IMG_KEY,
        )

        train_datapipe = DictTransformIterDataPipe(
            train_datapipe,
            {
                IMG_KEY: train_transform,
                TGT_KEY: webdataset_label_transform,
            },
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
        val_datapipe = get_webdataset_datapipe_v1(
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
            val_datapipe = DictBlankImageSkipperDP(val_datapipe, img_key=IMG_KEY)

        val_datapipe = DictToPILIterDataPipe(
            val_datapipe,
            img_key=IMG_KEY,
        )

        val_datapipe = DictTransformIterDataPipe(
            val_datapipe,
            {
                IMG_KEY: val_transform,
                TGT_KEY: webdataset_label_transform,
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

        self.discriminator.load_state_dict(ckpt_dict["discriminator"])

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

        ckpt_dict["discriminator"] = self.discriminator.state_dict()
        ckpt_dict["optimizer"] = self.optimizer.state_dict()
        ckpt_dict["scheduler"] = self.scheduler.state_dict() if self.scheduler else None
        ckpt_dict["current_epoch"] = self.current_epoch
        ckpt_dict["global_step"] = self.global_step

        with tune.checkpoint_dir(step=self.global_step) as checkpoint_dir:
            save_path = os.path.join(checkpoint_dir, "ckpt.pth")
            torch.save(ckpt_dict, save_path)

            # folder_name = Path(save_path).parent.name
            # mlflow.log_artifact(save_path, folder_name)

    def set_modules_train(self):
        self.model.train()
        self.discriminator.train()

    def set_modules_eval(self):
        self.model.eval()
        self.discriminator.eval()

    def adjust_learning_rate(self, epoch, init_param_lr, lr_epoch_1, lr_epoch_2):
        i = 0
        for param_group in self.optimizer.param_groups:
            init_lr = init_param_lr[i]
            i += 1
            if(epoch <= lr_epoch_1):
                param_group['lr'] = init_lr * 0.1 ** 0
            elif(epoch <= lr_epoch_2):
                param_group['lr'] = init_lr * 0.1 ** 1
            else:
                param_group['lr'] = init_lr * 0.1 ** 2

    def train(self):
        print("Training started")
        self.set_modules_train()

        train_loss_meter = AverageMeter("train_loss")
        classif_loss_meter = AverageMeter("classif_loss")
        metric_loss_meter = AverageMeter("metric_loss")
        ad_loss_meter = AverageMeter("ad_loss")
        train_acc_meter = AverageMeter("train_acc")
        train_domain_acc_meter = AverageMeter("train_domain_acc")
        acc_calculator = accuracy.Accuracy(threshold=0.2, compute_on_step=True).to(
            self.device
        )

        if self.max_epoch:
            max_epoch = self.max_epoch
            self.max_iter = sys.maxsize
        else:
            max_epoch = self.max_iter // self.iter_per_epoch

        for epoch in range(self.current_epoch, max_epoch):
            self.current_epoch = epoch
            batch_dict: Dict[str, torch.Tensor]

            for batch_dict in self.train_loader:
                img = batch_dict[IMG_KEY].to(self.device, non_blocking=True)
                target = batch_dict[TGT_KEY].to(self.device, non_blocking=True)
                lbl_metrics = batch_dict[DOM_LBL_KEY].to(self.device, non_blocking=True)

                param_lr_tmp = []
                for param_group in self.optimizer.param_groups:
                    param_lr_tmp.append(param_group["lr"])

                self.set_modules_train()
                self.optimizer.zero_grad()
                self.adjust_learning_rate(self.current_epoch+1, self.init_param_lr, self.config.hparams.lr_epoch_1, self.config.hparams.lr_epoch_2)
                # with torch.autocast(device_type=self.device, enabled=self.config.amp.enabled):
                cls_out, feat = self.model(img)

                classif_loss = self.criterion_cls(
                    cls_out, target.view(-1)
                )

                real_idxs = torch.where(target == 1)
                real_feats = feat[real_idxs]
                real_doms = lbl_metrics[real_idxs].abs()

                discriminator_out = self.discriminator(real_feats, self.global_step)
                ad_loss = self.criterion_dom(discriminator_out, real_doms.view(-1))
                train_domain_acc = topk_accuracy(discriminator_out, real_doms)[0]

                if self.use_custom_domain_lbls:
                    lbl_metrics = lbl_metrics.clamp_min(0)
                else:
                    lbl_metrics = target

                metric_loss = self.metric_loss(feat, lbl_metrics.view(-1))

                train_loss = (
                    (classif_loss * self.config.hparams.lambda_cls)
                    + (ad_loss * self.config.hparams.lambda_adloss)
                    + (metric_loss * self.config.hparams.lambda_metric)
                )

                train_acc = acc_calculator(
                    cls_out.softmax(dim=1)[..., 1], target.view(-1)
                )

                # ------------------------- running metric statistics ------------------------ #
                minibatch_size = img.shape[0]
                classif_loss_meter.update(classif_loss.item(), n=minibatch_size)
                ad_loss_meter.update(ad_loss.item(), n=minibatch_size)
                metric_loss_meter.update(metric_loss.item(), n=minibatch_size)
                train_loss_meter.update(train_loss.item(), n=minibatch_size)
                train_acc_meter.update(train_acc.item(), n=minibatch_size)
                train_domain_acc_meter.update(train_domain_acc.item(), n=minibatch_size)

                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # -------------------------- validation and logging -------------------------- #
                self.global_step += 1
                if self.global_step % self.config.log_every_n_step == 0:
                    print("currently on step: ", self.global_step)

                if self.sched_interval == "step" and self.scheduler:
                    self.scheduler.step()

            log_dict = {
                "train_loss_avg": train_loss_meter.avg,
                "classif_loss_avg": classif_loss_meter.avg,
                "metric_loss_avg": metric_loss_meter.avg,
                "ad_loss_avg": ad_loss_meter.avg,
                "train_acc_avg": train_acc_meter.avg,
                "train_acc_cal": acc_calculator.compute().item(),
                "train_dom_acc_avg": train_domain_acc_meter.avg,
                "epoch": self.current_epoch,
                "optim_lr": param_lr_tmp[0],
            }

            self.set_modules_eval()
            val_log_dict = self.run_validation()
            log_dict.update(val_log_dict)

            self.save_checkpoint()

            tune.report(**log_dict)
            mlflow.log_metrics(log_dict, step=self.global_step)

            self.set_modules_train()
            self.current_epoch += 1

            if self.sched_interval == "epoch" and self.scheduler:
                self.scheduler.step()

    @torch.no_grad()
    def run_validation(self):
        all_log_dicts = {}
        for val_loader, val_name, _ in self.val_loaders:
            print("Evaluating {}".format(val_name))
            val_classif_losses: List[torch.Tensor] = []
            predictions = []
            targets = []

            batch_dict: Dict[str, torch.Tensor]
            for batch_dict in val_loader:
                img: torch.Tensor = batch_dict[IMG_KEY].to(self.device, non_blocking=True)
                target: torch.Tensor = batch_dict[TGT_KEY].to(self.device, non_blocking=True)

                # with torch.autocast(
                #     device_type=self.device, enabled=self.config.amp.enabled
                # ):
                cls_out, _ = self.model(img)

                classif_loss = self.criterion_cls(
                    cls_out, target.view(-1)
                )
                val_classif_losses.append(classif_loss.item())

                preds = cls_out.softmax(dim=1)[..., 1]
                predictions.append(preds.cpu())
                targets.append(target.cpu())

            val_classif_loss = np.mean(val_classif_losses)
            val_loss = val_classif_loss

            predictions = torch.cat(predictions).cpu().numpy()
            targets = torch.cat(targets).cpu().numpy()

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
                f"{val_name}_loss": val_loss,
                f"{val_name}_threshold": metrics["THRESHOLD"],
                f"{val_name}_AUC": metrics["ROC_AUC"] * 100,
                f"{val_name}_EER": metrics["EER"] * 100,
                f"{val_name}_ACER": metrics["ACER"] * 100,
                f"{val_name}_APCER": metrics["APCER"] * 100,
                f"{val_name}_NPCER": metrics["NPCER"] * 100,
                f"{val_name}_NPCER_at_APCER5e-2": metrics["NPCER@APCER5%"] * 100,
                f"{val_name}_NPCER_at_APCER1e-2": metrics["NPCER@APCER1%"] * 100,
                f"{val_name}_NPCER_at_APCER5e-3": metrics["NPCER@APCER0.5%"]
                * 100,
                f"{val_name}_NPCER_at_APCER0": npcer_at_apcer0 * 100,
                f"{val_name}_THRESH_at_APCER0": thresh_at_apcer0,
                f"{val_name}_ACC": acc,
            }
            all_log_dicts.update(val_log_dict)

        all_log_dicts["epoch"] = self.current_epoch
        return all_log_dicts

    def finish(self):
        print("DONE TRAINING!")

def ssdg_binary_tuning(
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

    trainer = SSDGBinaryTuneTrainer(args, config, checkpoint_dir)
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

    # os.environ["MLFLOW_TRACKING_USERNAME"] = config.mlflow_tracking_username
    # os.environ["MLFLOW_TRACKING_PASSWORD"] = config.mlflow_tracking_password
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
        mlflow_mixin(partial(ssdg_binary_tuning, tune_config_path=tune_config_path)),
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

    print("finished")


if __name__ == "__main__":
    _main()
