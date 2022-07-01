from stats.distribution import *
import numpy as np

norm = Normal(mu=0,cov=np.eye(1))
norm.test()

psn = Poisson(10)
psn.test()

mu_norm = 0
mu_psn = 50
sigma_norm = 0.1
psnorm = PoissonNormal(mu_norm,mu_psn,sigma_norm)
psnorm.test()

