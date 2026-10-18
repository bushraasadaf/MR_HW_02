import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

runs = {
    'REF': 'pure_pursuit_circle_Ld0.5_REF.csv',
    'P09': 'pure_pursuit_circle_Ld0.5_P09.csv'
}

SKIP_START = 5.0    #skip the start-up
SKIP_END = 5.0      #skip the end where the goal runs out of path

#reference circle
#R = 1.5 m, centre = (0, 1.5)
theta = np.linspace(0, 2 * np.pi, 500)

ref_x = 1.5 * np.sin(theta)
ref_y = 1.5 - 1.5 * np.cos(theta)

for name, filename in runs.items():
    df = pd.read_csv(filename)
    #remove rows where ground truth had not arrived yet
    df = df.dropna(subset=['true_x', 'true_y', 'cte_true'])

    #steady-state error: mean signed CTE over the body of the lap
    #(+ = inside the circle, - = outside), skipping the first and last 5s
    final_time = df['time'].max()
    steady = df[(df['time'] > SKIP_START) & (df['time'] < final_time - SKIP_END)]

    ss_odom = steady['cte_odom'].mean()
    ss_true = steady['cte_true'].mean()

    print(f'{name}: /odom {ss_odom*100:+.2f} cm | ground truth {ss_true*100:+.2f} cm')
    #plot 1: reference + odom_true path
    plt.figure(figsize=(7, 7))
    plt.plot(
        ref_x,
        ref_y,
        '--',
        label='Reference circle'
    )
    plt.plot(
        df['odom_x'],
        df['odom_y'],
        label='/odom path'
    )
    plt.plot(
        df['true_x'],
        df['true_y'],
        label='Ground-truth path'
    )
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Pure Pursuit Circle - {name}, Ld = 0.5 m')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f'8b_circle_path_{name}.png',
        dpi=300
    )

    plt.show()
    #plot 2: cross track error vs time
    plt.figure(figsize=(8, 5))
    plt.plot(
        df['time'],
        df['cte_odom'],
        label='/odom CTE'
    )
    plt.plot(
        df['time'],
        df['cte_true'],
        label='Ground-truth CTE'
    )
    #desired path = zero CTE
    plt.axhline(
        0,
        linestyle='--',
        linewidth=1
    )
    plt.xlabel('Time (s)')
    plt.ylabel('Cross-track error (m)')

    plt.title(
        f'Cross-Track Error - {name}, Ld = 0.5 m'
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f'8b_cte_{name}.png',
        dpi=300
    )
    plt.show()