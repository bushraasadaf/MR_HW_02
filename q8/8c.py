import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

runs = {
    0.15: 'pure_pursuit_square_Ld0.15_REF.csv',
    0.3: 'pure_pursuit_square_Ld0.3_REF.csv',
    0.6: 'pure_pursuit_square_Ld0.6_REF.csv',
    1.0: 'pure_pursuit_square_Ld1.0_REF.csv',
}
corners = [(0, 0), (2, 0), (2, 2), (0, 2)]

plt.figure(figsize=(6, 6))
plt.plot([0, 2, 2, 0, 0], [0, 0, 2, 2, 0], 'k--', label='reference square')

#calculates the shortest distance from each robot position to each of the four square edges:
def edge_dist(x, y):
    out = []
    for ax, ay, bx, by in [(0, 0, 2, 0), (2, 0, 2, 2), (2, 2, 0, 2), (0, 2, 0, 0)]:
        vx, vy = bx - ax, by - ay
        #t finds where the closest point lies along the edge and np.clip() clips them to the range 0-1:
        t = np.clip(((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        out.append(np.hypot(x - (ax + t * vx), y - (ay + t * vy)))
    return np.min(out, axis=0)


for ld, f in runs.items():
    df = pd.read_csv(f).dropna(subset=['true_x', 'true_y'])
    plt.plot(df['true_x'], df['true_y'], label=f'Ld = {ld} m')

    #deviaion:
    # only samples within 0.75 m of a corner
    #calculate the distance to each corner via hypot , np.min() keeps the distance to the
    #nearest of the 4 corners for every recorded position
    #cte_true 
    d = np.min([np.hypot(df['true_x'] - cx, df['true_y'] - cy) for cx, cy in corners], axis=0)
    #finds the maximum distance from the square edges among those
    #selected positions and converts it to centimetres:
    dev = edge_dist(df['true_x'].to_numpy(), df['true_y'].to_numpy())[d < 0.75].max() * 100
    print(f'Ld = {ld}: corner deviation {dev:.1f} cm, peak omega {df["omega"].abs().max():.2f} rad/s')

plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('square, true paths')
plt.grid(True)
plt.legend()
# plt.savefig('8c_paths.png', dpi=300)
plt.savefig('8c_paths_after0.15.png', dpi=300)
plt.show()