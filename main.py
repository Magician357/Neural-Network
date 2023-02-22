import numpy as np
from ai import *

test=layer(10,20)
print(test.forward(np.random.rand(10)))