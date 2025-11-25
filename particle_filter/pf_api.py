import sys
import numpy as np

# module-level state
_particles = None  # shape (3, N)
_weights = None    # shape (N,)
_N = 200
_process_std_xy = 0.05
_process_std_bearing = 0.1
_measurement_std = 0.5
_step_distance = 0.1


def init_particles(N=_N, center_x=0.0, center_y=0.0, center_bearing=None, spread=1.0):
    global _particles, _weights, _N
    _N = int(N)
    pos = np.random.normal(loc=0.0, scale=spread, size=(2, _N))
    pos[0, :] += center_x
    pos[1, :] += center_y

    if center_bearing is None:
        bearing = np.random.uniform(-np.pi, np.pi, size=_N)
    else:
        bearing = np.random.normal(loc=center_bearing, scale=0.5, size=_N)

    _particles = np.vstack((pos, bearing.reshape(1, -1)))
    _weights = np.ones((_N,)) / _N


def reset_particles(center_x=None, center_y=None, center_bearing=None, spread=1.0):
    cx = 0.0 if center_x is None else float(center_x)
    cy = 0.0 if center_y is None else float(center_y)
    cb = None if center_bearing is None else float(center_bearing)
    init_particles(N=_N, center_x=cx, center_y=cy, center_bearing=cb, spread=spread)


def _resample():
    global _particles, _weights
    indices = np.random.choice(_N, size=_N, p=_weights)
    _particles = _particles[:, indices]
    _weights.fill(1.0 / _N)


def update_position(x, y):
    global _particles, _weights
    if _particles is None:
        init_particles()

    # Predict: move along bearing + noise
    bearings = _particles[2, :]
    dx = _step_distance * np.cos(bearings)
    dy = _step_distance * np.sin(bearings)
    _particles[0, :] += dx + np.random.normal(scale=_process_std_xy, size=_N)
    _particles[1, :] += dy + np.random.normal(scale=_process_std_xy, size=_N)
    _particles[2, :] += np.random.normal(scale=_process_std_bearing, size=_N)
    # wrap bearings to [-pi, pi]
    _particles[2, :] = ((_particles[2, :] + np.pi) % (2 * np.pi)) - np.pi

    # Update: weight by position likelihood only
    diffs = _particles[0:2, :] - np.array([[x], [y]])
    sq = np.sum(diffs * diffs, axis=0)
    coeff = 1.0 / (_measurement_std * np.sqrt(2 * np.pi))
    un = coeff * np.exp(-0.5 * sq / (_measurement_std ** 2)) + 1e-12
    _weights *= un
    _weights /= np.sum(_weights)

    # Estimate position
    est = np.average(_particles[0:2, :], axis=1, weights=_weights)

    # Resample if needed
    neff = 1.0 / np.sum(_weights ** 2)
    if neff < (_N / 2.0):
        _resample()

    return float(est[0]), float(est[1])


def main():
    def process_pair(a, b):
        try:
            x = float(a)
            y = float(b)
        except Exception:
            print("could not parse numbers", file=sys.stderr)
            return
        est_x, est_y = update_position(x, y)
        print(f"meas=({x:.3f},{y:.3f}) -> est=({est_x:.4f},{est_y:.4f})")

    if len(sys.argv) == 3:
        process_pair(sys.argv[1], sys.argv[2])
        return


if __name__ == '__main__':
    main()