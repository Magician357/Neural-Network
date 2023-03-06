import random
import math
import copy

def softmax(inputs):
    temp = [math.exp(v) for v in inputs]
    total = sum(temp)
    return [t / total for t in temp]

class layer:
    def __init__(self,numnodesIn,numnodesOut):
        self.nni,self.nno=numnodesIn,numnodesOut
        self.b=[0 for _ in range(numnodesOut)]
        self.w=[[random.random() for _ in range(numnodesIn)] for _ in range(numnodesOut)]
        self.outputs=[0 for _ in range(numnodesOut)]
    def forward(self,inputs):
        self.outputs=[0 for _ in range(self.nno)]
        for n in range(self.nno):
            curArray=self.w[n]
            for i in range(self.nni):
                self.outputs[n]+=curArray[i]*inputs[i]
            self.outputs[n]+=self.b[n]
        return self.outputs
    def mutate(self):
        mc=0.5
        for nA in range(self.nno):
            for nB in range(self.nni):
                if random.random() < mc:
                    mc*=0.75
                    self.w[nA][nB]+=random.random()-0.5
        for n in range(self.nno):
            if random.random() < 0.25:
                self.b[n]+=random.random()-0.5

class network:
    def __init__(self,layersizes,activation,final):
        self._copys=(layersizes,activation,final)
        self.a,self.final=activation,final
        cur=[]
        for n in range(len(layersizes)-1):
            cur.append(layer(layersizes[n],layersizes[n+1]))
        self.n=cur
    def forward(self,inputs):
        for curlayer in self.n:
            inputs=self.a(curlayer.forward(inputs))
        return self.final(inputs)
    def mutate(self):
        for n in range(len(self.n)):
            self.n[n].mutate()
    def copied(self):
        return network(*self._copys)

def getbestremove(lists):
    smallest=1000000000000000000000000
    smallestindex=0
    for n, value in enumerate(lists):
        if value[1] <= smallest:
            smallest=value[1]
            smallestindex=n
    small=lists[smallestindex]
    del lists[smallestindex]
    return small, lists

def average(lst):
    return sum(lst) / len(lst)

def score(res,expect):
    cur=[]
    for n in range(len(expect)):
        cur.append(abs(res[n]-expect[n])**2)
    return average(cur)

class trainer:
    def __init__(self,dataset,amount,layersizes,activation,final):
        self.amount=amount
        cur=[]
        for _ in range(amount):
            cur.append(network(layersizes,activation,final))
        self.full=cur
        self.dataset=dataset
        self.results=[[n,0] for n in range(amount)]
    def train(self):
        self.results=[[n,0] for n in range(self.amount)]
        for data in self.dataset:
            for n, network in enumerate(self.full):
                result=network.forward(data[0])
                self.results[n][1]+=score(result,data[1])
        best=[]
        for n in range(math.ceil(len(self.results)*0.1)):
            b, self.results=getbestremove(self.results)
            best.append(self.full[b[0]])
        i=0
        cur=copy.deepcopy(best)
        while len(cur) < self.amount:
            new=best[i].copied()
            new.mutate()
            cur.append(new)
            i=(i+1)%len(best)
        self.full=cur
    @property
    def score(self):
        curscore=0
        total=0
        for data in self.dataset:
            for network in self.full:
                result=network.forward(data[0])
                curscore+=score(result,data[1])
                total+=1
        return curscore/total