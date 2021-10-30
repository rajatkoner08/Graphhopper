from __future__ import division
from __future__ import absolute_import
import numpy as np
import torch

class baseline(object):
    def get_baseline_value(self):
        pass
    def update(self, target):
        pass

class ReactiveBaselineTF(baseline):
    def __init__(self, l):
        self.l = l
        self.b = tf.Variable( 0.0, trainable=False)
    def get_baseline_value(self):
        return self.b
    def update(self, target):
        self.b = tf.add((1-self.l)*self.b, self.l*target)

class ReactiveBaselineNumpy(baseline):
    def __init__(self, l):
        self.l = l
        self.b = np.array([0.], dtype=np.float32)
    def get_baseline_value(self):
        return self.b
    def update(self, target):
        self.b = np.add((1-self.l)*self.b, self.l*target)

class ReactiveBaselineTorch(baseline):
    def __init__(self, l):
        self.l = l
        self.b = torch.tensor([0.], dtype=torch.float32)
        if torch.cuda.is_available():
            self.b = self.b.cuda()
    def get_baseline_value(self):
        return self.b
    def update(self, target):
        self.b = torch.add((1-self.l)*self.b, self.l*target)
