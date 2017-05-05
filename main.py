import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multi_regression as mr

sr = mr.multi_regression()
d = {'one': pd.Series([1., 2., 3., 4., 5., 6., 7., 8.], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']),
     'one2': pd.Series([1., 4., 9., 16., 25., 36., 49., 64.], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']),
     'two': pd.Series([1., 4., 9., 16., 25., 36., 49., 64.], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])}
df = pd.DataFrame(d)

sr.train(df, ['one', 'one2'], 'two')
print sr.weights
pred = np.array([[1., 1., 1.],
                 [1., 2., 4.],
                 [1., 3., 9.],
                 [1., 4., 16.],
                 [1., 5., 25.],
                 [1., 6., 36.],
                 [1., 7., 49.],
                 [1., 9., 81.],
                 [1., 10., 100.]
                 ])
predictions = sr.predict(pred)
print predictions
plt.plot(df['one'], df['two'], '.', pred[:, 1], predictions, '-')
plt.show()
