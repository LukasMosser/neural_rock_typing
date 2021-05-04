import torch.nn as nn

class BaseInterpreter(object):
    def __init__(self, model: nn.Module):
        self.model = model

    def make_map(self):
        pass


class CAMInterpreter(BaseInterpreter):
    def __init__(self, *args, **kwargs):
        self