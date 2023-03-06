from ai import *
import random
from smalldataset import data
from time import perf_counter

test=trainer(data,100,[2,3,2],lambda x: x,softmax)
prev=test.score
n=0
last=test.score
start=perf_counter()
prevspeed=start
while True:
    test.train()
    n+=1
    if n%50 == 0:
        new=test.score
        time=perf_counter()
        print("\n",n)
        print(prev,">" if prev > new else "<",new)
        print(last,">" if last > new else "<",new)
        print(time-start)
        print(time-prevspeed)
        prevspeed=time
        last=new