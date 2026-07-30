import numpy as np
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

np.random.seed(0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(T=1500.0, dt=0.01, tau_r=0.2, tau_a=3.0, w_inhib=5.0,
             a_gain=0.5, I0=2.5, noise_sigma=0.6):
    """Two mutually-inhibiting rate units with slow self-adaptation and noise.

    Each unit represents the population encoding one interpretation of an
    ambiguous figure (e.g. 'front face down-left' vs 'front face up-right'
    for the Necker cube). Equal, constant input drives both units equally;
    with the adaptation gain used here the noise-free system is genuinely
    bistable (it settles into whichever unit wins early on and stays
    there forever). Switching is therefore noise-driven: fluctuations
    occasionally let the suppressed unit escape inhibition, with the slow
    self-adaptation of the dominant unit making that escape gradually more
    likely the longer a percept has been dominant. This noise-driven
    regime, as opposed to a purely adaptation-driven relaxation oscillator,
    follows the modeling framework in Moreno-Bote, Rinzel & Rubin (2007).
    """
    n_steps = int(T / dt)
    r1, r2 = 0.6, 0.4
    a1, a2 = 0.0, 0.0

    r1_trace = np.zeros(n_steps)
    r2_trace = np.zeros(n_steps)

    for t in range(n_steps):
        noise1 = np.random.normal(0, noise_sigma)
        noise2 = np.random.normal(0, noise_sigma)

        dr1 = (-r1 + sigmoid(I0 - w_inhib * r2 - a1 + noise1)) / tau_r
        dr2 = (-r2 + sigmoid(I0 - w_inhib * r1 - a2 + noise2)) / tau_r
        da1 = (-a1 + a_gain * r1) / tau_a
        da2 = (-a2 + a_gain * r2) / tau_a

        r1 += dt * dr1
        r2 += dt * dr2
        a1 += dt * da1
        a2 += dt * da2

        r1_trace[t] = r1
        r2_trace[t] = r2

    return r1_trace, r2_trace


def dominance_durations(r1_trace, r2_trace, dt, theta=0.05):
    """Hysteresis-based percept read-out: switch state only once one unit
    leads the other by a margin theta, rather than at every zero-crossing
    of r1 - r2. Without this, noise near a crossing produces a burst of
    spurious near-instantaneous 'switches' that are not real perceptual
    reversals, just as a subject wouldn't report a switch during an
    ambiguous, near-tied moment.
    """
    diff = r1_trace - r2_trace
    percept = np.zeros(len(diff), dtype=int)
    state = 1 if diff[0] > 0 else 0
    for t in range(len(diff)):
        if diff[t] > theta:
            state = 1
        elif diff[t] < -theta:
            state = 0
        percept[t] = state

    switch_idx = np.where(np.diff(percept) != 0)[0]
    boundaries = np.concatenate(([0], switch_idx, [len(percept) - 1]))
    durations = np.diff(boundaries) * dt
    return durations[1:-1]  # drop the first/last partial segments


r1_trace, r2_trace = simulate()
dt = 0.01
time = np.arange(len(r1_trace)) * dt
durations = dominance_durations(r1_trace, r2_trace, dt)

# --- Plot 1: activity time series with shaded dominant percept ---
fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

window = slice(0, int(150 / dt))  # first 150 time units, for readability
ax.plot(time[window], r1_trace[window], color=BLUE, lw=1.4, label="unit A (“front face down-left”)")
ax.plot(time[window], r2_trace[window], color=RED, lw=1.4, label="unit B (“front face up-right”)")

ax.set_xlabel("time (a.u.)", color=INK_SECONDARY, fontsize=10)
ax.set_ylabel("population activity", color=INK_SECONDARY, fontsize=10)
ax.tick_params(colors=INK_MUTED, labelsize=9)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")
ax.set_title("Two competing populations under equal, constant, ambiguous input",
             color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
fig.tight_layout()
fig.savefig("images/necker_bistability_timeseries.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

# --- Plot 2: distribution of dominance durations ---
fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.hist(durations, bins=30, color=BLUE, edgecolor="none", alpha=0.85, zorder=3)
ax.set_xlabel("dominance duration (a.u.)", color=INK_SECONDARY, fontsize=10)
ax.set_ylabel("count", color=INK_SECONDARY, fontsize=10)
ax.tick_params(colors=INK_MUTED, labelsize=9)
ax.yaxis.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Distribution of simulated dominance durations", color=INK_PRIMARY,
             fontsize=12, loc="left", pad=12)
fig.tight_layout()
fig.savefig("images/necker_bistability_durations.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

print(f"n switches: {len(durations)}")
print(f"mean duration: {durations.mean():.2f}, std: {durations.std():.2f}, "
      f"skew (mean/median): {durations.mean() / np.median(durations):.2f}")
print("saved images/necker_bistability_timeseries.png and images/necker_bistability_durations.png")
