import pandas as pd
import matplotlib.pyplot as plt

#p controller runs
RUNS = {
    'Kp = 0.5': 'wall_follower_kp_0.5.csv',
    'Kp = 1.0': 'wall_follower_kp_1.0.csv',
    'Kp = 3.0': 'wall_follower_kp_3.0.csv',
}

D_REF = 0.5
TOLERANCE = 0.02       #+/- 2 cm settling band
T_MAX = 27.0           #only straight-wall portion (decided through csvs)

def evaluate_metrics(time, D):
    #smooth D with a 10-sample moving average (~1 s) to suppress LiDAR noise
    D_s = D.rolling(10).mean().dropna()
    time_s = time.loc[D_s.index]

    #overshoot (on smoothed D so one noisy sample doesn't set the minimum)
    overshoot = max(0.0, D_REF - D_s.min())

    #settling time (on smoothed D)
    lower = D_REF - TOLERANCE
    upper = D_REF + TOLERANCE
    settling_time = None
    for i in range(len(D_s)):
        remaining_D = D_s.iloc[i:]
        if ((remaining_D >= lower) & (remaining_D <= upper)).all():
            settling_time = time_s.iloc[i]
            break

    #remaining error:average raw D over the final 1 second
    final_time = time.iloc[-1]
    final_data = D[time >= final_time - 1.0]
    remaining_error = D_REF - final_data.mean()
    return overshoot, settling_time, remaining_error

#store results
results = []

#plot D vs time
plt.figure(figsize=(9, 5))
for label, filename in RUNS.items():
    df = pd.read_csv(filename)
    #only use first 15 seconds
    df = df[df['time'] <= T_MAX]
    time = df['time']
    D = df['D']

    overshoot, settling_time, remaining_error = evaluate_metrics(time, D)
    results.append({
        'Gain': label,
        'Overshoot (cm)': overshoot * 100,
        'Settling Time (s)': settling_time,
        'Remaining Error (cm)': remaining_error * 100
    })

    plt.plot(time, D, label=label)

# Reference line D_ref = 0.5 m
plt.axhline(D_REF, color='black', linestyle='-', linewidth=1.5, label=f'D_ref = {D_REF} m')

#lines for the +/- 2 cm settling bounds
plt.axhline(D_REF + TOLERANCE, color='gray', linestyle='--', linewidth=1.0, label='±2 cm Settling Bounds')
plt.axhline(D_REF - TOLERANCE, color='gray', linestyle='--', linewidth=1.0)

plt.title('P-Only Wall Following: Distance vs. Time')
plt.xlabel('Time (s)')
plt.ylabel('Distance to Wall D (m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('q7b_p_only_comparison.png', dpi=300)
plt.show()

#print results
results_df = pd.DataFrame(results)
print("\nProblem 7(b) Results:")
print(results_df.to_string(index=False))