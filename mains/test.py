from stats.distribution import *
import numpy as np

norm = Normal(mu=0,cov=np.eye(1))
norm.test()

psn = Poisson(10)
psn.test()
