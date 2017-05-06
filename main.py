import matplotlib.pyplot as plt
import pandas as pd

import multi_regression as mr

sr = mr.multi_regression()
d = {'one': pd.Series([1., 2., 3., 4., 5., 6., 7., 8.], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']),
     'two': pd.Series([1., 4., 9., 16., 25., 36., 49., 64.], index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])}
df = pd.DataFrame(d)

sr.train(df, ['one'], 'two', deg=2)
print sr.weights
pred = pd.DataFrame()
pred['one'] = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        index=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])
predictions = sr.predict(pred)
print predictions
plt.plot(df['one'], df['two'], '.', pred['one'], predictions, '-')
plt.show()
