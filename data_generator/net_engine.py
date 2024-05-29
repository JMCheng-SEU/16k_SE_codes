import copy
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import warnings
from collections import Counter
from itertools import repeat
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from thop import profile
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset, RandomSampler, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

sys.path.append(str(Path(__file__).parent.parent))
from utils.common.define import AECScenario
from utils.logger import get_logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# from torch import autograd
from tqdm import tqdm

from utils.dsp import specgram
from utils.net_loss_fn import TYPE_LOSS_FN, LossMeter
from utils.net_valid_fn import (
    TYPE_VALID_FN,
    Metrics,
    compute_erle,
    compute_pesq,
    compute_si_snr,
    compute_sigmos,
    compute_stoi,
    mos_fn,
)

torch.autograd.set_detect_anomaly(True)

# plt.switch_backend("agg")


def setup_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


setup_seed()


def collate_fn_clip(batch):
    """
    Args:
        batch: [batch, 2] where 2 for noisy, clean
               format like [(noisy, clean), (), ... ()]
    """
    print(len(batch[0][0]), type(batch), type(batch[0]))
    N = min(batch, key=lambda x: len(x[0]))
    print("###", len(N[0]))


def check_string(inp: str, pattern: str):
    """Return `True` if `inp` contains `pattern`"""
    return re.search(pattern, inp) is not None


# TODO modify validate_ns and validate_aec func
def compute_valid_item(
    name: str,
    lbl: Union[np.ndarray, torch.Tensor],
    est: Union[np.ndarray, torch.Tensor],
    fs: Optional[int] = None,
    njobs: int = 0,
    pool: Optional[mp.Pool] = None,
) -> Union[Tuple[np.ndarray, int], Tuple[Dict, int]]:
    if name == "PESQ" or name == "STOI":
        # pesq only support 8K 'nb', 16K 'wb', range [-0.5, 4.5]
        # pesq_score = pesq(self.fs, sph, est, "wb")

        fn = compute_pesq if name == "PESQ" else compute_stoi
        scores = np.array(
            Parallel(n_jobs=njobs)(delayed(fn)(s, e, fs) for s, e in zip(lbl, est))
        )

    elif name == "SIGMOS":
        assert pool is not None and fs is not None
        if isinstance(est, torch.Tensor):
            est = est.detach().cpu().numpy()

        mos_value = pool.starmap(
            compute_sigmos,
            zip(est, repeat(fs)),
        )  # [{k:v}, {k:v}]

        score = Counter(mos_value[0])
        for value in mos_value[1:]:
            score += Counter(value)

        return dict(score), len(mos_value)

    elif name == "ERLE":
        scores = compute_erle(lbl, est)

    elif name == "SI-SNR":
        scores = compute_si_snr(lbl, est)
        scores = scores.cpu().numpy()
    else:
        raise RuntimeError(f"valid item {name} not spuported.")

    score, num = np.sum(scores), len(scores)
    return score, num


def compute_parallel(
    fn: TYPE_VALID_FN,
    lbl: Union[np.ndarray, torch.Tensor],
    est: Union[np.ndarray, torch.Tensor],
    fs: Optional[int] = None,
    njobs: int = 0,
):
    """__doc__
    Args:
        njobs: using batch calculation methods if equals 0, only valuded in numpy-base methods
    """
    if isinstance(lbl, np.ndarray):
        if njobs == 0:
            if fs is not None:
                scores = fn(lbl, est, fs)
            else:
                scores = fn(lbl, est)
        else:
            if fs is not None:
                scores = np.array(
                    Parallel(n_jobs=njobs)(
                        delayed(fn)(s, e, fs) for s, e in zip(lbl, est)
                    )
                )
            else:
                scores = np.array(
                    Parallel(n_jobs=njobs)(delayed(fn)(s, e) for s, e in zip(lbl, est))
                )

    elif isinstance(lbl, torch.Tensor):
        scores = fn(lbl, est).cpu().numpy()
    else:
        raise RuntimeError("input type error!")

    score, num = np.sum(scores), len(scores)
    return score, num


###############################################################
# Sections for core class implementing the training framework #
###############################################################


class NNCore:
    """Training engines
    Args:
        train_dataset: a torch.utils.data.Dataset class returning
            1. ref, mic, sph, scenario while `application` is AEC;
            2. mic, sph  while `application` is NS;
        application: 'AEC', 'NS', 'HS;
        net: input with mic, ref, or mic;
        net_D_metric: input with sph, est, training with metric GAN if configured, default with PESQ;
        net_D_mos: single input, training with GAN if configured;

    Examples Training:
        1. train by given train, valid dataset respectively.
            >>> init_dict = {
                "net": net,
                "train_dataset": Trunk(
                    f"/home/{os.getlogin()}/datasets/sig/dns_p09",
                ),
                "valid_dataset": NSTrunk(f"/home/{os.getlogin()}/datasets/sig/test"),
                "sample_rate": 48000,
                "batch_sz": batch_sz,
                "ckpt_path": f"{args.name}.pth",
                "retrain": False,
                "distributed": True,
                "application": "NS",
            }, where `NSTrunk` need to support `__len__` and `__getitem__(self, index)` api;
            >>> engine = NNCore(**init_dict)
            >>> engine.fit(n_epoch, loss_fn, valid_step, valid_name)

        2. train by given only one dataset and will split valid dataset with the given proportion.
            >>> init_dict = {
                    "net": net,
                    "train_dataset": Trunk(
                        f"/home/{os.getlogin()}/datasets/sig/dns_p09",
                    ),
                    "train_proportion": 0.9,
                    "sample_rate": 48000,
                    "batch_sz": batch_sz,
                    "ckpt_path": f"{args.name}.pth",
                    "retrain": args.retrain,
                    "distributed": True,
                    "application": "NS",
            }, where `NSTrunk` need to support `__len__` and `__getitem__(self, index)` api;
            >>> engine = NNCore(**init_dict)
            >>> engine.fit(n_epoch, loss_fn, valid_step, valid_name)

        3. training with GAN
            from utils.nn.GAN.CMGAN import Discriminator
            >>> init_dict = {
                    "net": net,
                    "train_dataset": NSTrunk(
                        # f"/home/{os.getlogin()}/datasets/howling/train_src",
                        f"/home/{os.getlogin()}/datasets/howling/train",
                        patten="**/[!^.]*_mic.wav",
                        keymap=("_mic.wav", "_sph.wav"),
                    ),
                    "valid_dataset": NSTrunk(
                        f"/home/{os.getlogin()}/datasets/howling/valid",
                        patten="**/[!^.]*_mic.wav",
                        keymap=("mic.wav", "sph.wav"),
                    ),
                    "batch_sz": batch_sz,
                    "ckpt_path": f"{args.name}.pth",
                    "retrain": args.retrain,
                    "distributed": True,
                    "application": "HS",
                    "sample_rate": 16000,
                    "net_D": Discriminator(512, 256, ndf=16),
                    "tfb_dir": f"{os.environ['HOME']}/tfb",
                }
            >>> engine = NNCore(**init_dict)
            >>> engine.fit(n_epoch, loss_fn, valid_step, valid_name)

    Examples Predicting:
        1. predict by given dataset.
            pnet = NNPredict(
                net=net,
                save_p=f"{args.name}-best.pth",
                out_path=f"/home/{os.getlogin()}/datasets/howling/est",
                app="NS",
            )
            pnet.predict(dset=Trunk(f"/home/{os.getlogin()}/datasets/howling/test"))
            where `Trunk` need to support `__iter__(self)` and `__next__(self)` api.

    """

    def __init__(
        self,
        net: Union[nn.Module, DDP],
        train_dataset: Dataset,
        batch_sz,
        sample_rate,
        valid_batch_sz: Optional[int] = None,
        train_proportion: Optional[float] = None,
        valid_dataset: Optional[Dataset] = None,
        net_D_metric: Optional[nn.Module] = None,
        net_D_mos: Optional[nn.Module] = None,
        retrain=False,
        application: str = "AEC",
        optimizer: str = "adam",
        scheduler: str = "LR",
        ckpt_path: str = "ckpt.pth",
        distributed: bool = False,
        is_draw_valid: bool = True,
        draw_num: int = 6,
        draw_column: int = 3,
        tfb_dir: str = f"{os.environ['HOME']}/tfb",
        valid_first: bool = False,
    ):
        """
        training with dynamic dataset given the `valid_dataset` and `train_dataset`;
        otherwise training with the traditional stragety when user only configure `train_dataset`.
        """
        if distributed is True:
            # NOTE load data must after dist.init_process_group under distributed system
            dist.init_process_group("nccl")
            self.rank = dist.get_rank()  # process id
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.local_rank)
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1

        self.fs = sample_rate
        self.valid_first = valid_first

        if not os.path.exists("ckpts"):
            os.makedirs("ckpts")

        self.ckpt = os.path.join("ckpts", ckpt_path)
        self.ckpt_best = os.path.join(
            "ckpts", os.path.splitext(ckpt_path)[0] + "_best.pth"
        )
        self.distributed = distributed
        self.app = application
        self.ckpt_dirname = Path(tfb_dir) / self.app
        self.batch_sz = batch_sz
        self.log = get_logger(name=f"{self.rank}_{ckpt_path.split('.')[0]}")

        writer_logdir = str(self.ckpt_dirname / Path(ckpt_path).stem)

        # rename tensorboard and delete the checkpoint pth file
        # print(ckpt_path, os.path.exists(ckpt_path))
        if retrain is True and self.rank == 0:
            self.backup_history(writer_logdir)

        if self.distributed is True:
            dist.barrier()

        self.writer = SummaryWriter(log_dir=writer_logdir)

        if valid_dataset is None:
            # split dataset to train, val sub dataset
            assert (
                train_proportion is not None
            ), "Must configure `train_proportion` when `valid_dataset` is None"
            n_train = int(train_proportion * len(train_dataset))
            n_valid = len(train_dataset) - n_train

            # print(f"## {n_train} for training, {n_valid} for validation.")

            train_dset, valid_dset = random_split(
                train_dataset,
                [n_train, n_valid],
                generator=torch.Generator().manual_seed(0),
            )
        else:
            train_dset = train_dataset
            valid_dset = valid_dataset
            # print(
            #     f"## {len(train_dset)} for training, {len(valid_dset)} for validation."
            # )
            n_train = len(train_dset)
            n_valid = len(valid_dset)

        self.n_train = n_train
        self.n_valid = n_valid

        if self.distributed is True:
            # warp dataset by distributed sampler
            # dataloader.shuffle must be False, if configured sampler
            sampler_train = DistributedSampler(train_dset, rank=self.rank)
            sampler_valid = DistributedSampler(valid_dset, rank=self.rank)

        else:
            sampler_train = RandomSampler(train_dset)
            sampler_valid = RandomSampler(valid_dset)

        self.sampler_train = sampler_train
        self.sampler_valid = sampler_valid

        loader_train = DataLoader(
            train_dset,
            batch_size=batch_sz,
            sampler=sampler_train,
            num_workers=10,
            pin_memory=True,
            shuffle=False,
        )
        loader_valid = DataLoader(
            valid_dset,
            batch_size=batch_sz if valid_batch_sz is None else valid_batch_sz,
            sampler=sampler_valid,
            num_workers=10,
            pin_memory=True,
            shuffle=False,
            # collate_fn=collate_fn_clip,
        )

        self.is_draw_valid = is_draw_valid
        self.draw_n = draw_num
        self.draw_column = draw_column

        self.loader_train = loader_train
        self.loader_valid = loader_valid
        self.device = torch.device("cuda", self.local_rank)

        # used to statistic the validation dataset raw metrics
        self.metric_unprocess = Metrics()

        # calculate flops
        self._check_flops(net)
        # self._net_info()

        self.net = net.to(self.device)
        assert (
            net_D_metric is None or net_D_mos is None
        ), "Metric and MOS GAN can't enable at the same time."
        if net_D_metric is not None:
            self.net_D_metric = net_D_metric.to(self.device)
            self.gan_metric = True
        else:
            self.gan_metric = False

        if net_D_mos is not None:
            self.net_D_mos = net_D_mos.to(self.device)
            self.gan_mos = True
        else:
            self.gan_mos = False

        # self.log.info(f"load ckpt: {self.ckpt}")
        if ckpt_path != "" and os.path.exists(self.ckpt):
            # load model
            ckpt = self.load_ckpt(self.ckpt)
        else:
            self.start_epoch = 1
            self.best_score = torch.finfo(torch.float32).min
            ckpt = None

        # * create DDP model
        if self.distributed is True:
            # will sync the model params to cuda:0
            self.net = DDP(
                self.net,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                # find_unused_parameters=True,
            )
            if self.gan_metric:
                self.net_D_metric = DDP(
                    self.net_D_metric,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )
            if self.gan_mos:
                self.net_D_mos = DDP(
                    self.net_D_mos,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                )

        # * load optimizer and schedule status, must after DDP
        self.optimizer = self.config_optimizer(optimizer, self.net.parameters())
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.8)
        self.scheduler = self.config_scheduler("StepLR", self.optimizer)
        if ckpt is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])

        if self.gan_metric:
            self.optimizer_D = self.config_optimizer(
                optimizer, self.net_D_metric.parameters(), alpha=2.0
            )
            self.scheduler_D = self.config_scheduler("StepLR", self.optimizer_D)
            if ckpt is not None:
                self.optimizer_D.load_state_dict(ckpt["optimizer_D"])
                self.scheduler_D.load_state_dict(ckpt["scheduler_D"])

        if self.gan_mos:
            mp.freeze_support()
            self.pool = mp.Pool(processes=10)
            self.optimizer_D_mos = self.config_optimizer(
                optimizer, self.net_D_mos.parameters(), alpha=2.0
            )
            # self.scheduler_D_mos = lr_scheduler.StepLR(
            #     self.optimizer_D_mos, step_size=20, gamma=0.8
            # )
            self.scheduler_D_mos = self.config_scheduler("StepLR", self.optimizer_D_mos)

            if ckpt is not None:
                self.optimizer_D_mos.load_state_dict(ckpt["optimizer_D_mos"])
                self.scheduler_D_mos.load_state_dict(ckpt["scheduler_D_mos"])

    def __del__(self):
        if self.distributed is True:
            dist.destroy_process_group()
        if self.gan_mos:
            self.pool.close()

        self.valid_pool.close() if self.valid_pool is not None else None

    def __str__(self):
        content = ""
        ncol = 6
        total, trainable, total_sz = self._net_info(show=False)
        content += "=" * 60 + "\n"
        content += f"{'ckpt':<{ncol}}: {self.ckpt}\n"
        content += f"{'app':<{ncol}}: {self.app}\n"
        content += f"{'Total':<{ncol}}: {total_sz/1024**2:.3f}MB\n"
        content += f"{'nTotal':<{ncol}}: {total:<{ncol},d}, "
        content += f"nTrainable: {trainable: <{ncol},d}, "
        content += f"FLOPS: {self.flops / 1024**3:.3f}G\n"
        content += f"{'Batch':<{ncol}}: {self.batch_sz:,d}\n"
        content += f"{'fs':<{ncol}}: {self.fs:,d}\n"
        content += f"{'nTrain':<{ncol}}: {self.n_train:,d}, "
        content += f"nValid: {self.n_valid:,d}\n"
        content += "=" * 60
        return content

    def config_optimizer(
        self, opt_type: str = "adam", params=None, alpha: float = 1.0
    ) -> Optimizer:
        supported = {
            "adam": lambda p: torch.optim.Adam(p, lr=alpha * 5e-4, amsgrad=False),
            "adamw": lambda p: torch.optim.AdamW(p, lr=alpha * 5e-4, amsgrad=False),
            "rmsprop": lambda p: torch.optim.RMSprop(p, lr=alpha * 5e-4),
        }
        assert opt_type in supported, "optimizer type is not supported"
        # if mask_only:
        #     params = []
        #     for n, p in self.net.named_parameters():
        #         if "mask" in n:
        #             params.append(p)
        # elif df_only:
        #     params = (p for n, p in self.net.named_parameters() if "df" in n.lower())
        # else:
        #     params = self.net.parameters()
        return supported[opt_type](
            params if params is not None else self.net.parameters()
        )

    def config_scheduler(self, name: str, optimizer: Optimizer):
        supported = {
            "StepLR": lambda p: lr_scheduler.StepLR(p, step_size=20, gamma=0.8),
            "ReduceLROnPlateau": lambda p: lr_scheduler.ReduceLROnPlateau(
                p, mode="min", factor=0.5, patience=1
            ),
        }
        assert name in supported, "scheduler type is not supported"
        return supported[name](optimizer)

    def _check_flops(self, net: nn.Module):
        x = torch.randn(1, self.fs)
        with warnings.catch_warnings():
            # ignore api deprecated warning.
            warnings.simplefilter("ignore")
            # profile will add some nodes in Model, therefore, need deep copy
            if self.app != "AEC":
                self.flops, _ = profile(copy.deepcopy(net), inputs=(x,), verbose=False)
            else:
                self.flops, _ = profile(
                    copy.deepcopy(net), inputs=(x, x), verbose=False
                )

    def _net_info(self, show=True):
        # total, trainable, non_trainable = 0, 0, 0

        # for params in self.net.parameters():
        #     num = np.prod(params.size())
        #     total += num
        #     if params.requires_grad:
        #         trainable += num
        #     else:
        #         non_trainable += num
        total = sum(p.numel() for p in self.net.parameters())
        total_sz = sum(p.numel() * p.element_size() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        if show:
            print(f"Total params: {total:,d}")
            print(f"Trainable params: {trainable:,d}")
            print(f"Non-Trainable params: {total-trainable:,d}")

        return total, trainable, total_sz

    def backup_history(self, writer_logdir: str):
        # backup_dirname = writer_logdir + time.strftime("-%Y%m%d%H%M")
        backup_dirname = writer_logdir + "_hist_1"
        if os.path.exists(backup_dirname):
            hist = list(
                self.ckpt_dirname.rglob(Path(writer_logdir).stem + "_hist_[0-9]*")
            )
            hist = sorted(hist, key=lambda l: int(str(l).split("_")[-1]), reverse=True)
            for f in hist:
                fname = f.name
                num = fname.split("_")[-1]
                idx = int(num) + 1
                fname = fname[: -len(num)] + str(idx)
                fp = f.parent / fname
                f.rename(fp)

        if os.path.exists(writer_logdir):
            os.rename(writer_logdir, backup_dirname)

            if os.path.exists(self.ckpt):
                shutil.copy(self.ckpt, backup_dirname)
                os.remove(self.ckpt)

            if os.path.exists(self.ckpt_best):
                shutil.copy(self.ckpt_best, backup_dirname)
                os.remove(self.ckpt_best)

    def save_ckpt(self, epoch, is_best: bool = False):
        if self.rank == 0:
            if not is_best:
                state = {
                    "epoch": epoch,
                    # save net params that can be load by non-DDP module
                    # which load params from module.xxx
                    "net": self.net.module.state_dict()
                    if self.distributed
                    else self.net.state_dict(),
                    "best_score": self.best_score,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "metrics_unprocessed": self.metric_unprocess.state_dict
                    if self.metric_unprocess.done is True
                    else None,
                }
                if self.gan_metric:
                    state.update(
                        {
                            "discriminator": self.net_D_metric.module.state_dict()
                            if self.distributed
                            else self.net_D_metric.state_dict(),
                            "optimizer_D": self.optimizer_D.state_dict(),
                            "scheduler_D": self.scheduler_D.state_dict(),
                        }
                    )
                if self.gan_mos:
                    state.update(
                        {
                            "discriminator_mos": self.net_D_mos.module.state_dict()
                            if self.distributed
                            else self.net_D_mos.state_dict(),
                            "optimizer_D_mos": self.optimizer_D_mos.state_dict(),
                            "scheduler_D_mos": self.scheduler_D_mos.state_dict(),
                        }
                    )
                torch.save(state, self.ckpt)

                torch.save(
                    self.net.module.state_dict()
                    if self.distributed
                    else self.net.state_dict(),
                    self.ckpt.split(".")[0] + f"_{epoch}_epoch.pth",
                )
            else:  # is_best
                torch.save(
                    self.net.module.state_dict()
                    if self.distributed
                    else self.net.state_dict(),
                    self.ckpt_best,
                )

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.start_epoch = (
            ckpt["epoch"] + 1 if self.valid_first is False else ckpt["epoch"]
        )
        self.best_score = ckpt["best_score"]
        # print("##", dist.get_rank(), self.rank, self.start_epoch)
        self.net.load_state_dict(ckpt["net"])
        if self.gan_metric:
            self.net_D_metric.load_state_dict(ckpt["discriminator"])
        if self.gan_mos:
            self.net_D_mos.load_state_dict(ckpt["discriminator_mos"])
        # self.optimizer.load_state_dict(ckpt["optimizer"])
        # self.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt["metrics_unprocessed"] is not None:
            self.metric_unprocess.state_dict = ckpt["metrics_unprocessed"]
            self.metric_unprocess.done = True

        return ckpt

    def set_valid_fn(self, valid_name_l: List[str]):
        if "SIGMOS" in valid_name_l:
            mp.freeze_support()
            self.valid_pool = mp.Pool(processes=10)
        else:
            self.valid_pool = None

        valid_type = {
            "PESQ": "numpy",
            "STOI": "numpy",
            "ERLE": "numpy",
            "SI-SNR": "torch",
            "SIGMOS": "numpy",
        }
        self.metric_coeff = {
            "PESQ": 0.2,
            "STOI": 1.0,
            "ERLE": 0.0,
            "SI-SNR": 0.0,
            "MOS_COL": 0.0,
            "MOS_DISC": 0.0,
            "MOS_LOUD": 0.0,
            "MOS_NOISE": 0.0,
            "MOS_REVERB": 0.0,
            "MOS_SIG": 1.0,
            "MOS_OVRL": 1.0,
        }

        valid_dict = {}
        for name in valid_name_l:
            assert not (
                self.app != "AEC" and name == "ERLE"
            ), "ERLE is only used in AEC application"

            valid_dict[name] = valid_type[name]
        self.valid_name = valid_dict

    def _fit_each_step(
        self,
        batch,
        loss_fn,
        track,
        loss_fn_GAN=F.mse_loss,
        track_D: Optional[LossMeter] = None,
        debug: bool = False,
    ):
        # TODO need modified according to the dataset implementation
        self.optimizer.zero_grad()
        if self.app == "AEC":
            ref, mic, sph, _ = batch
            ref = ref.to(self.device)
            mic = mic.to(self.device)
            sph = sph.to(self.device)
            # scenario = scenario.to(self.device)
            est = self.net(mic, ref)

        elif self.app == "NS" or self.app == "HS":
            mic, sph = batch
            mic = mic.to(self.device)
            sph = sph.to(self.device)
            est = self.net(mic)

        else:
            raise RuntimeError(f"not supported application type {self.app}")

        assert loss_fn is not None if not self.gan_metric else True
        if loss_fn is not None:
            loss = loss_fn(sph, est)
        else:
            loss = torch.tensor(0.0, device=self.device)

        if self.gan_metric:
            score_G = self.net_D_metric(est, sph)
            loss += loss_fn_GAN(score_G, torch.ones_like(score_G))

        if self.gan_mos:
            mos_G_est = self.net_D_mos(est).flatten()
            # mos_G_lbl = torch.tensor(
            #     self.pool.starmap(
            #         mos_fn,
            #         zip(
            #             est.detach().cpu().numpy(), repeat(self.fs), repeat("MOS_OVRL")
            #         ),
            #     ),
            #     device = mos_G_est.device,
            # ).float()
            # loss += loss_fn_GAN(mos_G_est, mos_G_lbl)
            loss += loss_fn_GAN(mos_G_est, torch.ones_like(mos_G_est))

        if debug is True:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()

        # for n, p in self.net.named_parameters():
        #     assert (
        #         torch.isnan(p.grad).sum() == 0
        #     ), f"{torch.isnan(p.grad).sum()}"
        # nn.utils.clip_grad_norm_(
        #    self.net.parameters(), max_norm=1, norm_type=2
        # )

        self.optimizer.step()
        track.update(loss)

        # << training discriminator
        if self.gan_metric:
            self.optimizer_D.zero_grad()

            score_enh = self.net_D_metric(est.detach(), sph).flatten()
            score_max = self.net_D_metric(sph, sph).flatten()

            # TODO use different metric methods
            label_enh = torch.tensor(
                Parallel(n_jobs=-1)(
                    # delayed(compute_pesq)(l, e, self.fs, norm=True)
                    delayed(compute_pesq)(l, e, 16000, norm=True)
                    for l, e in zip(sph.cpu().numpy(), est.detach().cpu().numpy())
                ),
                device=self.device,
            ).float()

            loss_D = (
                loss_fn_GAN(score_enh, label_enh)
                + loss_fn_GAN(score_max, torch.ones_like(score_max))
            ) * 0.5
            loss_D.backward()
            self.optimizer_D.step()
            track_D.update(loss_D) if track_D is not None else None

        if self.gan_mos:
            self.optimizer_D_mos.zero_grad()

            score_enh = self.net_D_mos(est.detach()).flatten()
            score_sph = self.net_D_mos(sph).flatten()

            label_enh = torch.tensor(
                self.pool.starmap(
                    mos_fn,
                    zip(
                        est.detach().cpu().numpy(), repeat(self.fs), repeat("MOS_OVRL")
                    ),
                ),
                device=score_enh.device,
            ).float()
            # label_enh = torch.tensor(label_enh, device=score_enh.device).float()
            label_sph = torch.tensor(
                self.pool.starmap(
                    mos_fn,
                    zip(sph.cpu().numpy(), repeat(self.fs), repeat("MOS_OVRL")),
                ),
                device=score_sph.device,
            ).float()
            # label_sph = torch.tensor(label_sph, device=score_sph.device).float()

            loss_D = (
                loss_fn_GAN(score_enh, label_enh) + loss_fn_GAN(score_sph, label_sph)
            ) * 0.5
            loss_D.backward()
            self.optimizer_D_mos.step()
            track_D.update(loss_D) if track_D is not None else None
        # >> training discriminator

    def fit(
        self,
        n_epoch,
        loss_fn: Optional[TYPE_LOSS_FN] = None,
        loss_fn_GAN=F.mse_loss,
        valid_name: List[str] = ["PESQ", "STOI", "ERLE", "SI-SNR"],
        valid_step: int = 1,
        debug: bool = False,
        skip_train: bool = False,
    ):
        """
        loss_fn: (sph, est)
        """

        self.set_valid_fn(valid_name)

        # self.metric_unprocess.enable = valid_raw

        for i in range(self.start_epoch, n_epoch + 1):
            # Training Part
            track = LossMeter(dist=self.distributed)
            self.net.train()

            if self.gan_metric:
                track_D = LossMeter(dist=self.distributed)
                self.net_D_metric.train()
            else:
                track_D = None

            if self.gan_mos:
                track_D = LossMeter(dist=self.distributed)
                self.net_D_mos.train()
            else:
                track_D = None

            if self.distributed is True:
                assert isinstance(self.sampler_train, DistributedSampler)
                self.sampler_train.set_epoch(i)

            ###################################################
            ################## training part ##################
            ###################################################
            if self.valid_first is False:
                if self.rank == 0:
                    pbar = tqdm(
                        self.loader_train,
                        ncols=90 if self.gan_metric or self.gan_mos else 80,
                        # dynamic_ncols=True,
                        leave=True,
                        desc=f"Epoch-{i}/{n_epoch}",
                    )
                else:
                    pbar = self.loader_train

                for batch in pbar:
                    # train by each batch
                    self._fit_each_step(
                        batch=batch,
                        loss_fn=loss_fn,
                        loss_fn_GAN=loss_fn_GAN,
                        track=track,
                        track_D=track_D,
                        debug=debug,
                    )

                    if self.rank == 0 and isinstance(pbar, tqdm):
                        # only show the main process loss
                        if self.gan_metric or self.gan_mos:
                            pbar.set_postfix(L=track.value, LD=track_D.value)
                        else:
                            pbar.set_postfix(Loss=f"{track.value:.3f}")

                    if skip_train:
                        break

                self.scheduler.step()

                if self.gan_metric:
                    self.scheduler_D.step()
                if self.gan_mos:
                    self.scheduler_D_mos.step()

                if self.rank == 0:
                    if isinstance(pbar, tqdm):
                        pbar.close()

                    self.writer.add_scalars("train/loss", track.state_dict, i)
                    if self.gan_metric or self.gan_mos:
                        self.writer.add_scalars("train/loss_D", track_D.state_dict, i)

                self.save_ckpt(i, is_best=False)
                if i % valid_step != 0:
                    continue

            #########################################################
            #################### validation part ####################
            #########################################################
            mtric = Metrics(self.metric_coeff)

            self.net.eval()

            if self.rank == 0:
                pbar = tqdm(
                    self.loader_valid,
                    ncols=90 if self.gan_metric or self.gan_mos else 80,
                    # dynamic_ncols=true,
                    # leave=False,
                    leave=True,
                    desc=f"valid-{i}/{n_epoch}",
                )
            else:
                pbar = self.loader_valid

            for idx, batch in enumerate(pbar):
                self._valid_each_step(batch, mtric)
                # show valid score at the end of validation epoch

                if idx == len(pbar) - 1 and self.rank == 0 and isinstance(pbar, tqdm):
                    mtric.dist_all_reduce()
                    if "PESQ" in valid_name:
                        pbar.set_postfix(pesq=f"{mtric.get_score('PESQ'):.3f}")

            if self.metric_unprocess.done is False:
                self.metric_unprocess.dist_all_reduce()
                self.metric_unprocess.done = True

            if self.rank == 0:
                # todo update when not using default metrics
                # score = 0.2 * mtric.get_score("pesq") + mtric.get_score("stoi")
                score = mtric.score

                #####################
                # NOTE save ckpts #
                #####################
                if self.best_score < score:
                    self.best_score = score
                    self.save_ckpt(i, is_best=True)

                for k, v in mtric.state_dict.items():
                    if self.metric_unprocess.has_key(k):
                        raw = self.metric_unprocess.get_score(k)

                        if check_string(k, "^MOS*"):
                            self.writer.add_scalars(
                                "sigmos/" + k, {"est": v, "raw": raw}, i
                            )
                        else:
                            self.writer.add_scalars(
                                "eval/" + k, {"est": v, "raw": raw}, i
                            )
                    else:
                        if check_string(k, "^MOS*"):
                            self.writer.add_scalars("sigmos/" + k, {"est": v}, i)
                        else:
                            self.writer.add_scalars("eval/" + k, {"est": v}, i)

                if self.is_draw_valid is True:
                    self.draw_spectrum(
                        (n_epoch - i + 1), self.loader_train, self.writer
                    )

                if isinstance(pbar, tqdm):
                    pbar.close()

            if self.distributed is True:
                dist.barrier()

            self.valid_first = False
            # >> end one epoch

        # exit training
        self.writer.close()

    @torch.no_grad()
    def _valid_each_step(self, batch, mtric: Metrics):
        if self.app == "AEC":
            ref, mic, sph, scenario = batch
            ref = ref.to(self.device)
            mic = mic.to(self.device)
            sph = sph.to(self.device)

            with torch.no_grad():
                est = self.net(mic, ref)

            self.validate_aec(mtric, mic, sph, est, scenario)

        elif self.app == "NS" or self.app == "HS":
            mic, sph = batch
            mic = mic.to(self.device)
            sph = sph.to(self.device)

            with torch.no_grad():
                est = self.net(mic)
            self.validate_ns(mtric, mic, sph, est)

    def validate_ns(
        self,
        mtric: Metrics,
        mic: torch.Tensor,
        sph: torch.Tensor,
        est: torch.Tensor,
        njobs: int = 10,
    ):
        np_mic = mic.cpu().numpy()
        np_sph = sph.cpu().numpy()
        np_est = est.cpu().numpy()

        for name, dtype in self.valid_name.items():
            score, num = compute_valid_item(
                name,
                np_sph if dtype == "numpy" else sph,
                np_est if dtype == "numpy" else est,
                fs=self.fs,
                njobs=njobs,
                pool=self.valid_pool,
            )
            if name == "SIGMOS":
                for k, v in score.items():
                    mtric.update(k, v, num)
            else:
                mtric.update(name, score, num)

            if self.metric_unprocess.done is True:
                continue

            score, num = compute_valid_item(
                name,
                np_sph if dtype == "numpy" else sph,
                np_mic if dtype == "numpy" else mic,
                fs=self.fs,
                njobs=njobs,
                pool=self.valid_pool,
            )
            if name == "SIGMOS":
                for k, v in score.items():
                    self.metric_unprocess.update(k, v, num)
            else:
                self.metric_unprocess.update(name, score, num)

    def validate_aec(
        self,
        mtric: Metrics,
        mic: torch.Tensor,
        sph: torch.Tensor,
        est: torch.Tensor,
        scenario: np.ndarray,
        njobs: int = 10,
    ):
        np_mic = mic.cpu().numpy()
        np_sph = sph.cpu().numpy()
        np_est = est.cpu().numpy()
        scenario = scenario.squeeze()

        np_dt_sph = np_sph[scenario == AECScenario.DT.value]
        np_dt_est = np_est[scenario == AECScenario.DT.value]

        np_fe_est = np_est[scenario == AECScenario.FE.value]
        np_fe_mic = np_mic[scenario == AECScenario.FE.value]

        # np_ne_sph = np_sph[scenario == AECScenario.NE.value]
        # np_ne_est = np_est[scenario == AECScenario.NE.value]

        np_dt_mic = np_mic[scenario == AECScenario.DT.value]
        # np_ne_mic = np_mic[scenario == AECScenario.NE.value]

        for name, dtype in self.valid_name.items():
            if name == "PESQ" or name == "STOI":
                if len(np_dt_sph):
                    score, num = compute_valid_item(
                        name,
                        np_dt_sph,
                        np_dt_est,
                        fs=self.fs,
                        njobs=njobs,
                        pool=self.valid_pool,
                    )
                    mtric.update(name, score, num)

                    if self.metric_unprocess.done is True:
                        continue

                    score, num = compute_valid_item(
                        name,
                        np_dt_sph,
                        np_dt_mic,
                        fs=self.fs,
                        njobs=njobs,
                        pool=self.valid_pool,
                    )
                    self.metric_unprocess.update(name, score, num)

            elif name == "ERLE":
                if len(np_fe_mic):
                    score, num = compute_valid_item(
                        name,
                        np_fe_mic,
                        np_fe_est,
                        fs=self.fs,
                        njobs=njobs,
                        pool=self.valid_pool,
                    )
                    mtric.update(name, score, num)
            else:
                # TODO
                score, num = compute_valid_item(
                    name,
                    np_sph if dtype == "numpy" else sph,
                    np_est if dtype == "numpy" else est,
                    fs=self.fs,
                    njobs=njobs,
                    pool=self.valid_pool,
                )
                if name == "SIGMOS":
                    for k, v in score.items():
                        mtric.update(k, v, num)
                else:
                    mtric.update(name, score, num)

                if self.metric_unprocess.done is True:
                    continue

                score, num = compute_valid_item(
                    name,
                    np_sph if dtype == "numpy" else sph,
                    np_mic if dtype == "numpy" else mic,
                    fs=self.fs,
                    njobs=njobs,
                    pool=self.valid_pool,
                )
                if name == "SIGMOS":
                    for k, v in score.items():
                        self.metric_unprocess.update(k, v, num)
                else:
                    self.metric_unprocess.update(name, score, num)

        # for name, fn in self.valid_fn.items():
        #     if name == "PESQ" or name == "STOI":
        #         # pesq only support 8K 'nb', 16K 'wb', range [-0.5, 4.5]
        #         if len(np_dt_sph):
        #             score, num = compute_parallel(
        #                 fn, np_dt_sph, np_dt_est, njobs=njobs, fs=self.fs
        #             )
        #             mtric.update(name, score, num)

        #             if self.metric_unprocess.done is True:
        #                 continue
        #             score, num = compute_parallel(
        #                 fn, np_dt_sph, np_dt_mic, njobs=njobs, fs=self.fs
        #             )
        #             self.metric_unprocess.update(name, score, num)

        #     elif name == "ERLE":
        #         if len(np_fe_mic):
        #             score, num = compute_parallel(fn, np_fe_mic, np_fe_est, njobs=0)
        #             mtric.update(name, score, num)
        #     else:
        #         # NOTE check, using torch methods
        #         score, num = compute_parallel(fn, sph, est)
        #         mtric.update(name, score, num)

        #         if self.metric_unprocess.done is True:
        #             continue
        #         score, num = compute_parallel(fn, sph, mic)
        #         self.metric_unprocess.update(name, score, num)

    def draw_spectrum(self, epoch_i, loader: DataLoader, writer):
        self.net.eval()
        if self.app == "AEC":
            ref, mic, sph, scenario = next(iter(loader))
            scenario_np = scenario.cpu().numpy()
            if len(ref) > self.draw_n:
                ref = ref[: self.draw_n]
                mic = mic[: self.draw_n]
                sph = sph[: self.draw_n]
                scenario_np = scenario_np[: self.draw_n]

            ref = ref.to(self.device)
            mic = mic.to(self.device)
            sph = sph.to(self.device)

            with torch.no_grad():
                est = self.net(mic, ref)

        elif self.app == "NS" or self.app == "HS":
            mic, sph = next(iter(loader))
            if len(mic) > self.draw_n:
                mic = mic[: self.draw_n]
                sph = sph[: self.draw_n]

            mic = mic.to(self.device)
            sph = sph.to(self.device)

            with torch.no_grad():
                est = self.net(mic)

            scenario_np = np.full(self.draw_n, -1)

        else:
            raise RuntimeError(f"The application scenario is not defined: {self.app}")

        if len(est) >= self.draw_column:
            nrow = 3 * (len(est) // self.draw_column)
            ncol = self.draw_column
        else:  # len(est) < self.draw_column
            nrow = 3
            ncol = len(est)

        fig, ax = plt.subplots(nrow, ncol, constrained_layout=True, figsize=(16.0, 9.0))

        N = len(ax.flat)
        est_np = est.cpu().numpy()[:N]
        sph_np = sph.cpu().numpy()[:N]
        mic_np = mic.cpu().numpy()[:N]
        sce_np = scenario_np[:N]

        nframe = 512
        nhop = 256

        # for row in range(0, 3 * (len(est_np) // self.draw_column), 3):
        for row in range(0, nrow, 3):
            ax_n = ax[row, :]
            for axi, mic_data in zip(ax_n.flat, mic_np[:ncol]):
                spec, (xticks, xlabel), (yticks, ylabel) = specgram(
                    mic_data, nframe, nhop, self.fs
                )
                axi.set_yticks(yticks)
                axi.set_yticklabels(ylabel)
                axi.set_xticks(xticks)
                axi.set_xticklabels(xlabel)
                axi.imshow(spec, origin="lower", aspect="auto", cmap="jet")

            ax_n = ax[row + 1, :]
            for axi, est_data, sc in zip(ax_n.flat, est_np[:ncol], sce_np[:ncol]):
                # axi.specgram(
                #     est_data,
                #     NFFT=512,
                #     Fs=self.fs,
                #     scale_by_freq=True,
                #     window=np.sqrt(np.hanning(M=512)),
                #     cmap="jet",
                # )
                spec, (xticks, xlabel), (yticks, ylabel) = specgram(
                    est_data, nframe, nhop, self.fs
                )
                if sc == AECScenario.NE.value:
                    sc_type = "NE"
                elif sc == AECScenario.FE.value:
                    sc_type = "FE"
                elif sc == AECScenario.DT.value:
                    sc_type = "DT"
                else:
                    sc_type = self.app

                axi.set_title(f"Estimation_{sc_type}")
                axi.set_yticks(yticks)
                axi.set_yticklabels(ylabel)
                axi.set_xticks(xticks)
                axi.set_xticklabels(xlabel)
                axi.imshow(spec, origin="lower", aspect="auto", cmap="jet")

            ax_n = ax[row + 2, :]
            for axi, sph_data in zip(ax_n.flat, sph_np[:ncol]):
                spec, (xticks, xlabel), (yticks, ylabel) = specgram(
                    sph_data, nframe, nhop, self.fs
                )
                # axi.set_title("Clean")
                axi.set_yticks(yticks)
                axi.set_yticklabels(ylabel)
                axi.set_xticks(xticks)
                axi.set_xticklabels(xlabel)
                axi.imshow(spec, origin="lower", aspect="auto", cmap="jet")

            mic_np = mic_np[ncol:]
            est_np = est_np[ncol:]
            sph_np = sph_np[ncol:]
            sce_np = sce_np[ncol:]

        # plt.close(fig)
        # for axi, est_data in zip(ax)
        writer.add_figure(
            f"images/Epoch-{epoch_i}",
            fig,
            global_step=None,
            close=True,
            walltime=None,
        )

    def _check_params(self, net: nn.Module):
        for n, l in net.named_parameters():
            print(n, l)


class NNPredict:
    def __init__(
        self,
        net: nn.Module,
        save_p: str,
        out_path: str = "",
        fs=16000,
        app="AEC",
    ):
        """
        save_p: the model path
        out_path: the output dirname of est files
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # NOTE can't remove the output path since it could be the source directory.
        # if out_path != "":
        #     if os.path.exists(out_path) is True:
        #         shutil.rmtree(out_path)
        #     os.makedirs(out_path)
        self.out_path = out_path
        self.fs = fs
        self.app = app

        print("Loading ckpt: ", os.path.join("ckpts", save_p))
        ckpt = torch.load(os.path.join("ckpts", save_p))
        net.load_state_dict(ckpt)

        self._check_flops(net)

        net.to(device=self.device)
        net.eval()
        self.net = net

    def _check_flops(self, net: nn.Module):
        x = torch.randn(1, self.fs)
        with warnings.catch_warnings():
            # ignore api deprecated warning.
            warnings.simplefilter("ignore")
            # profile will add some nodes in Model, therefore, need deep copy
            if self.app != "AEC":
                self.flops, _ = profile(copy.deepcopy(net), inputs=(x,), verbose=False)
            else:
                self.flops, _ = profile(
                    copy.deepcopy(net), inputs=(x, x), verbose=False
                )

    def _net_info(self, show=True):
        # total, trainable, non_trainable = 0, 0, 0

        # for params in self.net.parameters():
        #     num = np.prod(params.size())
        #     total += num
        #     if params.requires_grad:
        #         trainable += num
        #     else:
        #         non_trainable += num
        total = sum(p.numel() for p in self.net.parameters())
        total_sz = sum(p.numel() * p.element_size() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        if show:
            print(f"Total params: {total:,d}")
            print(f"Trainable params: {trainable:,d}")
            print(f"Non-Trainable params: {total-trainable:,d}")

        return total, trainable, total_sz

    def __str__(self):
        content = ""
        ncol = 6
        total, trainable, total_sz = self._net_info(show=False)
        content += "=" * 60 + "\n"
        content += f"{'app':<{ncol}}: {self.app}\n"
        content += f"{'Total':<{ncol}}: {total_sz/1024**2:.3f}MB\n"
        content += f"{'nTotal':<{ncol}}: {total:<{ncol},d}, "
        content += f"nTrainable: {trainable: <{ncol},d}, "
        content += f"FLOPS: {self.flops / 1024**3:.3f}G\n"
        content += f"{'fs':<{ncol}}: {self.fs:,d}\n"
        content += "=" * 60
        return content

    @torch.no_grad()
    def predict(self, dset: Union[Generator, Dataset]):
        """
        dset: yield ref, mic, fout
        """
        pbar = tqdm(dset, total=len(dset), ncols=80)

        for batch in pbar:
            if self.app == "AEC":
                ref, mic, fout = batch
                ref = ref.to(self.device)  # (1, T)
                mic = mic.to(self.device)
                try:
                    est = self.net(mic, ref)
                except Exception as e:
                    print("##", fout)
                    continue
            elif self.app == "NS" or self.app == "HS":
                mic, fout = batch
                mic = mic.to(self.device)
                est = self.net(mic)
            else:
                raise RuntimeError(f"app not supported {self.app}")

            est = est.cpu().numpy()
            est = est.squeeze()

            if self.out_path != "":
                # _, fname = os.path.split(fout)
                # fout = os.path.join(self.out_path, fname)
                fout = os.path.join(self.out_path, fout)
            try:
                sf.write(fout, est, self.fs)
            except Exception as e:
                dir_p, _ = os.path.split(fout)
                if not os.path.exists(dir_p):
                    os.makedirs(dir_p)
                    sf.write(fout, est, self.fs)
                else:
                    print("##", e)

        pbar.close()


if __name__ == "__main__":
    dist.init_process_group("nccl")

    m = LossMeter()
    loss = torch.tensor([1])
    v1 = torch.tensor([2])

    m.update(loss, {"v1": v1})
    loss = torch.tensor([3])
    m.update(loss, {"v1": v1})
    print(m)

    dist.destroy_process_group()
