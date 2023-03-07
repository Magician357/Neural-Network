from ai import *
#https://replit.com/@Magic-Man/cython
import random
from smalldataset import data
from smallerdataset import full
from time import perf_counter
import multiprocessing


def main(i,max):
    test=trainer(data,100,[2,3,2],smooth,softmax)
    filename="results/data"+str(i)+".txt"
    f=open(filename,"w")
    f.write("")
    f.close()
    prev=test.score
    n=0
    last=test.score
    start=perf_counter()
    prevspeed=start
    while True:
        test.train()
        n+=1
        if n%50 == 0:
            time=perf_counter()
            new=test.score
            a=">" if prev > new else "<"
            b=">" if last > new else "<"
            c=time-start
            d=time-prevspeed
            string=f"\n {n}\n{prev} {a} {new}\n{last} {b} {new}\n{c}\n{d}"
            print("\n",i,string)
            prevspeed=time
            last=new
            f=open(filename,"a")
            f.write(string)
            f.close()

nonef=lambda x: x

def add(i,max):
    test=trainer(full,100,[2,3,1],nonef,nonef)
    filename="results/data"+str(i)+".txt"
    f=open(filename,"w")
    f.write("")
    f.close()
    prev=test.score
    n=0
    last=test.score
    start=perf_counter()
    prevspeed=start
    while True:
        test.train()
        n+=1
        if n%10 == 0:
            time=perf_counter()
            new=test.score
            a=">" if prev > new else "<"
            b=">" if last > new else "<"
            c=time-start
            d=time-prevspeed
            string=f"\n {n}\n{prev} {a} {new}\n{last} {b} {new}\n{c}\n{d}"
            print("\n",i,string)
            prevspeed=time
            last=new
            f=open(filename,"a")
            f.write(string)
            f.close()

cc=multiprocessing.cpu_count()
print(cc)
threads=[]
for n in range(cc):
    threads.append(multiprocessing.Process(target=add,args=tuple([n,10000])))

for process in threads:
    process.start()

for process in threads:
    process.join()