

class ExponentialDecay(object):
    def __init__(self, base_learning_rate, decay_rate, decay_steps):
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate
        assert decay_steps!=0
        self.decay_steps = decay_steps
    def step(self, global_step):
        return self.base_learning_rate*self.decay_rate**(global_step/self.decay_steps)
