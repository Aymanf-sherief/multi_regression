import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multi_regression as mr

sr = mr.multi_regression()

length =500
x = np.arange(length, dtype=float)
y = x ** 2 + (np.random.rand(length) * 100)
d = {'one': pd.Series(x),
     'two': pd.Series(y)}
df = pd.DataFrame(d)

sr.train(df, ['one'], 'two', deg=2, max_iterations=1000, l2_penalty=1e1, step_size=1e-15)
pred = pd.DataFrame()
pred['one'] = pd.Series(x)
predictions = sr.predict(pred)
plt.plot(df['one'], df['two'], '.', pred['one'], predictions, '-')
plt.show()
