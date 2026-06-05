"""
Test Hamiltonian loss: compute H(t) along a REAL trajectory
using predicted params, check if energy is conserved.

If params are correct → H(t) is constant (low variance)
If params are wrong → H(t) fluctuates (high variance)

Uses noiseless states from the long-format parquet dataset.
"""
import sys
sys.path.insert(0, '/mnt/E/github-projects/NeuroDynamics/test_suites')

import numpy as np
import pandas as pd
from _hamiltonian import DoublePendulum


def hamiltonian_loss(pred_params, states):
    """
    pred_params: (m1, m2, l1, l2) as floats
    states: (T, 4) array of [theta1, omega1, theta2, omega2] (noiseless!)
    Returns: variance of H(t) normalized by mean|H|
    """
    m1, m2, l1, l2 = pred_params
    dp = DoublePendulum(m1, m2, l1, l2)

    H_vals = np.array([
        dp.hamiltonian(s[0], s[1], s[2], s[3])
        for s in states
    ])

    H_mean = np.mean(np.abs(H_vals))
    if H_mean < 1e-10:
        return 0.0
    return float(np.var(H_vals) / H_mean)


if __name__ == "__main__":
    df = pd.read_parquet("/mnt/E/github-projects/NeuroDynamics/datasets/dataset_doublependulumpts.parquet")

    # Grab config 0 — full 5000-step trajectory
    config_ids = sorted(df['config_id'].unique())
    print(f"Total configs: {len(config_ids)}")

    traj_idx = config_ids[0]
    traj_df = df[df['config_id'] == traj_idx].sort_values('time')

    true_params = {
        'm1': traj_df['mass pendulum 1'].iloc[0],
        'm2': traj_df['mass pendulum 2'].iloc[0],
        'l1': traj_df['length pendulum 1'].iloc[0],
        'l2': traj_df['length pendulum 2'].iloc[0],
    }

    # Build state trajectory from NOISELESS states (clean dynamics)
    states = np.column_stack([
        traj_df['no noise angle pendulum 1'].values,
        traj_df['no noise angularvel pendulum 1'].values,
        traj_df['no noise angle pendulum 2'].values,
        traj_df['no noise angularvel pendulum 2'].values,
    ])

    print(f"True params: m1={true_params['m1']:.4f}, m2={true_params['m2']:.4f}, "
          f"l1={true_params['l1']:.4f}, l2={true_params['l2']:.4f}")
    print(f"Trajectory: {len(states)} timesteps")
    print()

    # Loss with TRUE params (should be ~0 since Hamiltonian is conserved)
    loss_true = hamiltonian_loss(
        [true_params['m1'], true_params['m2'], true_params['l1'], true_params['l2']],
        states
    )
    print(f"H-loss (TRUE params):     {loss_true:.6e}")

    # Loss with slightly off params
    for scale in [0.05, 0.1, 0.2, 0.5, 1.0]:
        pred = [
            true_params['m1'] + scale,
            true_params['m2'] + scale * 0.5,
            true_params['l1'] + scale,
            true_params['l2'] + scale
        ]
        loss = hamiltonian_loss(pred, states)
        print(f"H-loss (scale={scale:4.2f}):     {loss:.6e}")

    # Loss with random params
    print()
    np.random.seed(42)
    for i in range(5):
        pred = np.random.uniform([0.25, 0.25, 0.5, 0.5], [1.5, 2.5, 2.5, 2.5])
        loss = hamiltonian_loss(list(pred), states)
        print(f"H-loss (random #{i+1}):      {loss:.6e}")

    # Try a few more trajectories to see if pattern holds
    print("\n--- Cross-trajectory check ---")
    for tid in config_ids[1:6]:
        tdf = df[df['config_id'] == tid].sort_values('time')
        tp = {
            'm1': tdf['mass pendulum 1'].iloc[0],
            'm2': tdf['mass pendulum 2'].iloc[0],
            'l1': tdf['length pendulum 1'].iloc[0],
            'l2': tdf['length pendulum 2'].iloc[0],
        }
        st = np.column_stack([
            tdf['no noise angle pendulum 1'].values,
            tdf['no noise angularvel pendulum 1'].values,
            tdf['no noise angle pendulum 2'].values,
            tdf['no noise angularvel pendulum 2'].values,
        ])

        l_true = hamiltonian_loss([tp['m1'], tp['m2'], tp['l1'], tp['l2']], st)
        l_wrong = hamiltonian_loss([tp['m1'] + 0.5, tp['m2'] + 0.25, tp['l1'] + 0.5, tp['l2'] + 0.5], st)
        print(f"Config {tid}: true={l_true:.2e}, wrong={l_wrong:.2e}, ratio={l_wrong/max(l_true,1e-20):.1f}x")