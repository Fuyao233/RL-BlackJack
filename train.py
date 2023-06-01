from BlackJackBattleEnv import *
from DDQN import *
import random
import torch
import numpy as np

model = DDQN(3, 2)
# model.cuda()
trainer = Trainer(BlackJack(dealer_policy=SimplePolicy(17)), model, buffer_capacity=200)
trainer.train()