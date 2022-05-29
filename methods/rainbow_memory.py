"""
rainbow-memory
Copyright 2021-present NAVER Corp.
GPLv3
"""
import logging
import os
import random
import PIL
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from methods.finetune import Finetune
from utils.data_loader import cutmix_data, ImageDataset
from randaugment.randaugment import RandAugment
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.augment import Cutout, Invert, Solarize, select_autoaugment

logger = logging.getLogger()

writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class RM(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.exp_env = kwargs["stream_env"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "uncertainty"

    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=0):
        if len(self.memory_list) > 0:
            mem_dataset = ImageDataset(
                pd.DataFrame(self.memory_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            memory_loader = DataLoader(
                mem_dataset,
                shuffle=True,
                batch_size=(batch_size // 2),
                num_workers=n_worker,
            )
            stream_batch_size = batch_size - batch_size // 2
        else:
            memory_loader = None
            stream_batch_size = batch_size

        # train_list == streamed_list in RM
        train_list = self.streamed_list
        test_list = self.test_list
        random.shuffle(train_list)
        # Configuring a batch with streamed and memory data equally.
        train_loader, test_loader = self.get_dataloader(
            stream_batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)+len(self.memory_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        self.model = self.model.to(self.device)
        for epoch in range(n_epoch):
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()

            train_loss, train_acc = self._train(train_loader=train_loader, memory_loader=memory_loader,
                                                optimizer=self.optimizer, criterion=self.criterion)
            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )
            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            logger.info(
                f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            best_acc = max(best_acc, eval_dict["avg_acc"])

        return best_acc, eval_dict

    def update_model(self, x, y, criterion, optimizer):
        optimizer.zero_grad()

        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            logit = self.model(x)
            loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                logit, labels_b
            )
        else:
            logit = self.model(x)
            loss = criterion(logit, y)

        _, preds = logit.topk(self.topk, 1, True, True)

        loss.backward()
        optimizer.step()
        return loss.item(), torch.sum(preds == y.unsqueeze(1)).item(), y.size(0)

    def _train(
        self, train_loader, memory_loader, optimizer, criterion
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        self.model.train()
        if memory_loader is not None and train_loader is not None:
            data_iterator = zip(train_loader, cycle(memory_loader))
        elif memory_loader is not None:
            data_iterator = memory_loader
        elif train_loader is not None:
            data_iterator = train_loader
        else:
            raise NotImplementedError("None of dataloder is valid")

        for data in data_iterator:
            if len(data) == 2:
                stream_data, mem_data = data
                x = torch.cat([stream_data["image"], mem_data["image"]])
                y = torch.cat([stream_data["label"], mem_data["label"]])
            else:
                x = data["image"]
                y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)

            l, c, d = self.update_model(x, y, criterion, optimizer)
            total_loss += l
            correct += c
            num_data += d

        if train_loader is not None:
            n_batches = len(train_loader)
        else:
            n_batches = len(memory_loader)

        return total_loss / n_batches, correct / num_data

    def allocate_batch_size(self, n_old_class, n_new_class):
        new_batch_size = int(
            self.batch_size * n_new_class / (n_old_class + n_new_class)
        )
        old_batch_size = self.batch_size - new_batch_size
        return new_batch_size, old_batch_size

    def update_memory(self, cur_iter, num_class=None):
        if num_class is None:
            num_class = self.num_learning_class

        if not self.already_mem_update:
            logger.info(f"Update memory over {num_class} classes by {self.mem_manage}")
            candidates = self.streamed_list + self.memory_list
            if len(candidates) <= self.memory_size:
                self.memory_list = candidates
                self.seen = len(candidates)
                logger.warning("Candidates < Memory size")
            else:
                if self.mem_manage == "random":
                    self.memory_list = self.rnd_sampling(candidates)
                elif self.mem_manage == "reservoir":
                    self.reservoir_sampling(self.streamed_list)
                elif self.mem_manage == "prototype":
                    self.memory_list = self.mean_feature_sampling(
                        exemplars=self.memory_list,
                        samples=self.streamed_list,
                        num_class=num_class,
                    )
                elif self.mem_manage == "uncertainty":
                    if cur_iter == 0:
                        self.memory_list = self.equal_class_sampling(
                            candidates, num_class
                        )
                    else:
                        self.memory_list = self.uncertainty_sampling(
                            candidates,
                            num_class=num_class,
                        )
                else:
                    logger.error("Not implemented memory management")
                    raise NotImplementedError

            assert len(self.memory_list) <= self.memory_size
            logger.info("Memory statistic")
            memory_df = pd.DataFrame(self.memory_list)
            logger.info(f"\n{memory_df.klass.value_counts(sort=True)}")
            # memory update happens only once per task iteratin.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")


    def evaluation_ext(self, test_list):
        # evaluation from out of class
        test_dataset = ImageDataset(
            pd.DataFrame(test_list),
            dataset=self.dataset,
            transform=self.test_transform,
        )
        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=32, num_workers=2
        )
        eval_dict = self.evaluation(test_loader, self.criterion)

        return eval_dict

    def rnd_sampling(self, samples):
        random.shuffle(samples)
        return samples[: self.memory_size]

    def reservoir_sampling(self, samples):
        for sample in samples:
            if len(self.memory_list) < self.memory_size:
                self.memory_list += [sample]
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:
                    self.memory_list[j] = sample
            self.seen += 1

    def mean_feature_sampling(self, exemplars, samples, num_class):
        """Prototype sampling

        Args:
            features ([Tensor]): [features corresponding to the samples]
            samples ([Datalist]): [datalist for a class]

        Returns:
            [type]: [Sampled datalist]
        """

        def _reduce_exemplar_sets(exemplars, mem_per_cls):
            if len(exemplars) == 0:
                return exemplars

            exemplar_df = pd.DataFrame(exemplars)
            ret = []
            for y in range(self.num_learned_class):
                cls_df = exemplar_df[exemplar_df["label"] == y]
                ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                    orient="records"
                )

            num_dups = pd.DataFrame(ret).duplicated().sum()
            if num_dups > 0:
                logger.warning(f"Duplicated samples in memory: {num_dups}")

            return ret

        mem_per_cls = self.memory_size // num_class
        exemplars = _reduce_exemplar_sets(exemplars, mem_per_cls)
        old_exemplar_df = pd.DataFrame(exemplars)

        new_exemplar_set = []
        sample_df = pd.DataFrame(samples)
        for y in range(self.num_learning_class):
            cls_samples = []
            cls_exemplars = []
            if len(sample_df) != 0:
                cls_samples = sample_df[sample_df["label"] == y].to_dict(
                    orient="records"
                )
            if len(old_exemplar_df) != 0:
                cls_exemplars = old_exemplar_df[old_exemplar_df["label"] == y].to_dict(
                    orient="records"
                )

            if len(cls_exemplars) >= mem_per_cls:
                new_exemplar_set += cls_exemplars
                continue

            # Assign old exemplars to the samples
            cls_samples += cls_exemplars
            if len(cls_samples) <= mem_per_cls:
                new_exemplar_set += cls_samples
                continue

            features = []
            self.feature_extractor.eval()
            with torch.no_grad():
                for data in cls_samples:
                    image = PIL.Image.open(
                        os.path.join("dataset", self.dataset, data["file_name"])
                    ).convert("RGB")
                    x = self.test_transform(image).to(self.device)
                    feature = (
                        self.feature_extractor(x.unsqueeze(0)).detach().cpu().numpy()
                    )
                    feature = feature / np.linalg.norm(feature, axis=1)  # Normalize
                    features.append(feature.squeeze())

            features = np.array(features)
            logger.debug(f"[Prototype] features: {features.shape}")

            # do not replace the existing class mean
            if self.class_mean[y] is None:
                cls_mean = np.mean(features, axis=0)
                cls_mean /= np.linalg.norm(cls_mean)
                self.class_mean[y] = cls_mean
            else:
                cls_mean = self.class_mean[y]
            assert cls_mean.ndim == 1

            phi = features
            mu = cls_mean
            # select exemplars from the scratch
            exemplar_features = []
            num_exemplars = min(mem_per_cls, len(cls_samples))
            for j in range(num_exemplars):
                S = np.sum(exemplar_features, axis=0)
                mu_p = 1.0 / (j + 1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p, axis=1, keepdims=True)

                dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
                i = np.argmin(dist)

                new_exemplar_set.append(cls_samples[i])
                exemplar_features.append(phi[i])

                # Avoid to sample the duplicated one.
                del cls_samples[i]
                phi = np.delete(phi, i, 0)

        return new_exemplar_set

    def uncertainty_sampling(self, samples, num_class):
        """uncertainty based sampling

        Args:
            samples ([list]): [training_list + memory_list]
        """
        self.montecarlo(samples, uncert_metric=self.uncert_metric)

        sample_df = pd.DataFrame(samples)
        mem_per_cls = self.memory_size // num_class

        ret = []
        for i in range(num_class):
            cls_df = sample_df[sample_df["label"] == i]
            if len(cls_df) <= mem_per_cls:
                ret += cls_df.to_dict(orient="records")
            else:
                jump_idx = len(cls_df) // mem_per_cls
                uncertain_samples = cls_df.sort_values(by="uncertainty")[::jump_idx]
                ret += uncertain_samples[:mem_per_cls].to_dict(orient="records")

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret

    def _compute_uncert(self, infer_list, infer_transform, uncert_name):
        batch_size = 32
        infer_df = pd.DataFrame(infer_list)
        infer_dataset = ImageDataset(
            infer_df, dataset=self.dataset, transform=infer_transform
        )
        infer_loader = DataLoader(
            infer_dataset, shuffle=False, batch_size=batch_size, num_workers=2
        )

        self.model.eval()
        with torch.no_grad():
            for n_batch, data in enumerate(infer_loader):
                x = data["image"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = logit.detach().cpu()

                for i, cert_value in enumerate(logit):
                    sample = infer_list[batch_size * n_batch + i]
                    sample[uncert_name] = 1 - cert_value

    def montecarlo(self, candidates, uncert_metric="vr"):
        transform_cands = []
        logger.info(f"Compute uncertainty by {uncert_metric}!")
        if uncert_metric == "vr":
            transform_cands = [
                Cutout(size=8),
                Cutout(size=16),
                Cutout(size=24),
                Cutout(size=32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(45),
                transforms.RandomRotation(90),
                Invert(),
                Solarize(v=128),
                Solarize(v=64),
                Solarize(v=32),
            ]
        elif uncert_metric == "vr_randaug":
            for _ in range(12):
                transform_cands.append(RandAugment())
        elif uncert_metric == "vr_cutout":
            transform_cands = [Cutout(size=16)] * 12
        elif uncert_metric == "vr_autoaug":
            transform_cands = [select_autoaugment(self.dataset)] * 12

        n_transforms = len(transform_cands)

        for idx, tr in enumerate(transform_cands):
            _tr = transforms.Compose([tr] + self.test_transform.transforms)
            self._compute_uncert(candidates, _tr, uncert_name=f"uncert_{str(idx)}")

        for sample in candidates:
            self.variance_ratio(sample, n_transforms)

    def variance_ratio(self, sample, cand_length):
        vote_counter = torch.zeros(sample["uncert_0"].size(0))
        for i in range(cand_length):
            top_class = int(torch.argmin(sample[f"uncert_{i}"]))  # uncert argmin.
            vote_counter[top_class] += 1
        assert vote_counter.sum() == cand_length
        sample["uncertainty"] = (1 - vote_counter.max() / cand_length).item()

    def equal_class_sampling(self, samples, num_class):
        mem_per_cls = self.memory_size // num_class
        sample_df = pd.DataFrame(samples)
        # Warning: assuming the classes were ordered following task number.
        ret = []
        for y in range(self.num_learning_class):
            cls_df = sample_df[sample_df["label"] == y]
            ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                orient="records"
            )

        num_rest_slots = self.memory_size - len(ret)
        if num_rest_slots > 0:
            logger.warning("Fill the unused slots by breaking the equilibrium.")
            ret += (
                sample_df[~sample_df.file_name.isin(pd.DataFrame(ret).file_name)]
                    .sample(n=num_rest_slots)
                    .to_dict(orient="records")
            )

        num_dups = pd.DataFrame(ret).file_name.duplicated().sum()
        if num_dups > 0:
            logger.warning(f"Duplicated samples in memory: {num_dups}")

        return ret