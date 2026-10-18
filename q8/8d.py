import numpy as np
import pandas as pd

df = pd.read_csv('pure_pursuit_square_Ld0.3_P09.csv')
o = df.iloc[-1]
t = df.dropna(subset=['true_x', 'true_y']).iloc[-1]
print(f'/odom closing error: {np.hypot(o.odom_x, o.odom_y) * 100:.1f} cm')
print(f'true closing error:  {np.hypot(t.true_x, t.true_y) * 100:.1f} cm')