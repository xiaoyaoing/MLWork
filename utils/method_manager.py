"""
ML大作业，method_managet
Copyright 2021-present NAVER Corp.
GPLv3
"""
import logging


# 论文的RM
# from methods.ori_rainbow_memory import RM

# 复现的RM
from methods.rainbow_memory import RM

logger = logging.getLogger()


def select_method(args, criterion, device, train_transform, test_transform, n_classes):
    kwargs = vars(args)

    method = RM(
        criterion=criterion,
        device=device,
        train_transform=train_transform,
        test_transform=test_transform,
        n_classes=n_classes,
        **kwargs,
    )

    logger.info("CIL Scenario: ")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_init_cls: {args.n_init_cls}")
    print(f"n_cls_a_task: {args.n_cls_a_task}")
    print(f"total cls: {args.n_tasks * args.n_cls_a_task}")


    return method
