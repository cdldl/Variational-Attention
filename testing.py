#TESTING
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from statsmodels.distributions.empirical_distribution import ECDF
val = pd.read_csv('../data/hypotheses_val.csv') 
print(val.quantile(np.linspace(0,1,11)))
ecdf = ECDF(val.hyp)
hypotheses_val = ecdf(val.hyp)#np.where(ecdf(hypotheses_val)>= 0.5,1,0)
ecdf2 = ECDF(val.real)
real_val = ecdf2(val.real)
#preds = np.where((hypotheses_val > 0.05)&(hypotheses_val < 0.2),-1,np.where((hypotheses_val > 0.8)&(hypotheses_val < 0.95),1,0))
#preds = np.where(hypotheses_val < 0.01,-1,np.where(hypotheses_val > 0.99,1,0))
preds = np.where((val.hyp < (np.mean(val.hyp) - 2 * val.hyp.std()))&(val.hyp > (np.mean(val.hyp) - 3 * val.hyp.std())),-1,
	np.where((val.hyp > (np.mean(val.hyp) + 1.5 * val.hyp.std()))&(val.hyp < (np.mean(val.hyp) + 3 * val.hyp.std())),1,0))

preds = np.where((val.hyp < (np.mean(val.hyp) - 2 * val.hyp.std())),-1,
	np.where((val.hyp > (np.mean(val.hyp) + 2 * val.hyp.std())),1,0))

print(Counter(preds))
#preds = np.where(hypotheses_val>0.5,1,-1)
print(np.corrcoef(val.hyp,val.real))
print(np.corrcoef(hypotheses_val,real_val))
plt.plot(np.cumsum(( preds[preds != 0] *val.real[preds != 0]).reset_index(drop=True))) 
#plt.plot(np.cumsum(( val.real[preds != 0]).reset_index(drop=True))) 
preds2 = preds[preds != 0]
real2 = val.real[preds!=0]
np.corrcoef(preds2,np.sign(real2))

accuracy_score(preds2,np.sign(real2))
plt.show()
