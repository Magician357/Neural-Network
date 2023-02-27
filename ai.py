import numpy as np
import random

def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def nf(x):
    return x

class layer:
    def __init__(self,numnodesin: int,numnodesout: int):
        self.nni, self.nno = numnodesin, numnodesout
        self.w=np.random.rand(numnodesin,numnodesout)
        self.b=np.zeros(numnodesout)
        self.forward(np.zeros(numnodesin))
    def forward(self,inp):
        output=np.zeros(self.nno)
        for n, weight in enumerate(self.w):
            for j, curinp in enumerate(inp):
                for i in range(self.nno):
                    output[i]+=curinp*weight[i]
                    output[i]+=self.b[i]
        self.output=output
        return output
    def mutate(self):
        for pos, _ in np.ndenumerate(self.w):
            x,y=pos
            mc=random.random() < 0.3
            if mc:
                self.w[x,y]+=random.random()-0.5
        for n in range(self.nno):
            mc=random.random() < 0.2
            if mc:
                self.b[n]+=random.random()-0.5

class network:
    def __init__(self,layersizes: list,activation=nf,final=nf):
        cur=[]
        for n in range(len(layersizes)-1):
            cur.append(layer(layersizes[n],layersizes[n+1]))
        self.network=cur
        self.activation=activation
        self.final=final
    def forward(self,inp):
        current=inp
        for n in range(len(self.network)):
            current=self.activation(self.network[n].forward(current))
        return self.final(current)
    def mutate(self):
        for n in range(len(self.network)):
            self.network[n].mutate()

class builder:
    def __init__(self,amount,layersizes: list,activation=nf,final=nf):
        cur=[]
        for _ in range(amount):
            cur.append(network(layersizes,activation,final))
        self.full=cur
        self.amount=amount
    def rebuild(self,best: list):
        current=best
        n=0
        while len(current) > self.amount:
            curnet=best[n]
            n=(n+1)%len(best)
            curnet.mutate()
            current.append(curnet)
        self.full=current

def disttuple(a,b):
    dists=[np.absolute(a[n]-b[n]) for n in range(len(a))]
    return sum(dists)/len(dists)

def sortdict(x):
    sort={k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
    cur=[]
    for key in sort:
        cur.append(key)
    return cur
    

class teacher:
    def __init__(self,dataset: tuple):
        self.data=dataset
    def test(self,networks):
        cd={}
        for n, network in enumerate(networks):
            score=[]
            for data in self.data:
                inputs,expected=list(data[0]),data[1]
                got=tuple(network.forward(inputs))
                score.append(np.power(disttuple(got,expected),2))
            cd[n]=sum(score)/len(score)
        sort=sortdict(cd)
        ninty=int(np.round(0.1*len(sort)))
        best=sort[:ninty]
        return [networks[i] for i in best]
    def sort(self, networks):
        cd={}
        for n, network in enumerate(networks):
            score=[]
            for data in self.data:
                inputs,expected=data[0],data[1]
                got=tuple(network.forward(inputs))
                score.append(np.power(disttuple(got,expected),2))
            cd[n]=sum(score)/len(score)
        return sortdict(cd)


class trainer:
    def __init__(self,dataset: tuple, amount: int, layersizes: list,activation=nf,final=nf):
        self.builder=builder(amount,layersizes,activation,final)
        self.teacher=teacher(dataset)
    def train(self):
        new=self.teacher.test(self.builder.full)
        self.builder.rebuild(new)
    
    @property
    def best(self):
        networks=self.builder.full
        return networks[self.teacher.sort(networks)[0]]