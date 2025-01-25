from .make_env import make_env
from .config import games
from .task_gym import *


load_task = lambda hyp: GymTask(games[hyp['task']],paramOnly=True)


# Special classification task used for neat-backprop
def load_cls_task(p):
    """ 
    Customized task for classification 
    task object with supports .nInput, .nOutput, .absWCap, .activations, .layers
    """
    class Task:
        def __init__(self):
            self.nInput = 2
            self.nOutput = 2
            self.absWCap = 2.0
            self.activations = [9,9,9]
            self.layers = [2, 5, 5, 2]  # Input -> Hidden -> Hidden -> Output
            self.actRange = [1, 3, 4, 5, 6, 9, 10, 11]
            
    return Task()