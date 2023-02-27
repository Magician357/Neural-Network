import numpy as np
from ai import *
from smalldataset import data

test=trainer(data,20,[2,3,2])
cur=test.best.forward(data[0])
print(cur)
test.train()
cur=test.best.forward(data[0])
print(cur)