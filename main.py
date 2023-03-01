import numpy as np
from ai import *
from smalldataset import data

test=trainer(data,20,[2,3,2])
cur=test.best.forward(data[0][0])
print(cur)
best=test.best
for n in range(100):
    test.train()
    cur=tuple(test.best.forward(data[0][0]))
    print(disttuple(cur,data[0][1]))
    if disttuple(cur,data[0][1]) < disttuple(best.forward(data[0][0]),data[0][1]):
        best=test.best

print("best:")
print(disttuple(best.forward(data[0][0]),data[0][1]))