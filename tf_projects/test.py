import numpy as np
import pandas as pd

a = np.array([1,2,3,4,5,6,7,8,9,10])
aa = pd.DataFrame(a,columns=['Y'])
print(aa.keys().contains('Y'))