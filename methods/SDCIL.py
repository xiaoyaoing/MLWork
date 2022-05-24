# 这个文件中继承finetune并实现四个方法
# before_task:  负责在每个任务训练前，负责更换分类头，重置优化器，以及一些变量的属性。
# train:  负责利用train loader进行训练。
# evaluation:  负责利用test loader测试结果
# afer_task: 负责在任务结束后更新保存的样本
import os
import finetune
class SDCIL(finetune.Finetune):
    # 先继承再重构
    def __init__(self, criterion, device, train_transform, test_transform, init_class, n_classes, train_loader, **kwargs): # 这里要放一些参数进去？ args, train_loader, feat_loader, current_task, fisher={}, prototype={}
        finetune.Finetune.__init__(self, criterion, device, train_transform, test_transform, init_class, n_classes, **kwargs)
        self.train_loader = train_loader

    def before_task(self, datalist):
        pass

    def train(self, cur_iter):
        pass

    def evaluation(self, test_loader, criterion):
        pass

    def after_task(self, cur_iter):
        pass