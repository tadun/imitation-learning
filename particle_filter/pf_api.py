import numpy as np

particles = None
weights = None
_N = 200
process_std = 0.1
measurement_std = 0.5

def init_particles(N=_N):
    global particles, weights, _N
    _N = int(N)
    particles = np.random.normal(scale=1.0, size=(2, _N))
    weights = np.ones((_N,)) / _N

def resample():
    global particles, weights
    indices = np.random.choice(_N, size=_N, p=weights)
    particles = particles[:, indices]
    weights.fill(1.0 / _N)

def update_position(x, y):
    global particles, weights
    if particles is None:
        init_particles()

    particles += np.random.normal(scale=process_std, size=particles.shape)

    diffs = particles - np.array([[x], [y]])
    sq = np.sum(diffs * diffs, axis=0)

    coeff = 1.0 / (measurement_std * np.sqrt(2 * np.pi))
    un = coeff * np.exp(-0.5 * sq / (measurement_std ** 2)) + 1e-12
    weights *= un
    weights /= np.sum(weights)

    est = np.average(particles, axis=1, weights=weights)

    neff = 1.0 / np.sum(weights ** 2)
    if neff < (_N / 2.0):
        resample()

    return float(est[0]), float(est[1])

def main():
    seq = [(0.0, 0.0), (0.1, 0.05), (0.15, 0.1), (0.2, 0.12)]
    for i, (x, y) in enumerate(seq):
        est_x, est_y = update_position(x, y)
        print(f"meas[{i}]=({x:.3f},{y:.3f}) -> est=({est_x:.4f},{est_y:.4f})")


if __name__ == '__main__':
    main()