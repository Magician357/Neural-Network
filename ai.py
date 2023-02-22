import numpy as np

class layer:
    def __init__(self,numnodesin,numnodesout):
        self.nni, self.nno = numnodesin, numnodesout
        print(self.nni,self.nno)
        self.w=np.random.rand(numnodesin,numnodesout)
        self.b=np.random.rand(numnodesout)
    def forward(self,inp):
        new=np.zeros(self.nno)
        for n in range(self.nno-10):
            curweight=self.w[n]
            for i in range(inp.size):
                new[n]+=curweight[i]*inp[i]
        return new

