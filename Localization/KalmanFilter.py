import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, x, r=np.eye(4), q=np.eye(4)):
        self.x = x
        self.P = np.eye(4)
        self.H = np.eye(4)
        self.Q = q
        self.R = r

        self.state_hist = np.array([self.x.copy()])

    def A(self):
        A = np.zeros((4, 4))
        A[0, 0] = 1.
        A[0, 1] = DT
        A[1, 1] = 1.
        A[2, 2] = 1.
        A[2, 3] = DT
        A[3, 3] = 1.
        return A

    def B(self):
        B = np.zeros((4, 2))
        B[0, 0] = (DT ** 2) / 2
        B[1, 0] = DT
        B[2, 1] = (DT ** 2) / 2
        B[3, 1] = DT
        return B

    def predict(self, controls=np.zeros(2)):
        A = self.A()
        B = self.B()

        self.x = A @ self.x + B @ controls
        self.P = A @ self.P @ A.T + self.Q

    def update(self, z):
        K2 = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        K = self.P @ self.H.T @ K2

        self.x += K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

        self.state_hist = np.vstack((self.state_hist, self.x))

    def __call__(self, controls, z):
        self.predict(controls)
        self.update(z)


def update(states):
    pos_noise = np.random.normal(0., .01, 2)
    vel_noise = np.random.normal(0., .01, 2)

    states[0] += (states[1] * DT) + pos_noise[0]
    states[1] = states[1] + vel_noise[0]
    states[2] += (states[3] * DT) + pos_noise[1]
    states[3] = states[3] + vel_noise[1]


# Must pass a copy of 'states' to this
def sense_state(states):
    pos_noise = np.random.normal(0., .01, 2)
    vel_noise = np.random.normal(0., .01, 2)

    states[0] += pos_noise[0]
    states[1] += vel_noise[0]
    states[2] += pos_noise[1]
    states[3] += vel_noise[1]

    return states


if __name__ == "__main__":
    DT = .1

    true_state = np.array([0., 5., 0., 2.5])
    true_state_hist = [true_state.copy()]

    cov_est = np.eye(4)
    cmd = np.array([.25, .35])

    q_mat = np.eye(4) * .1
    r_mat = np.eye(4) * .05
    init_measured_state = sense_state(true_state.copy())
    kf = KalmanFilter(init_measured_state, q_mat, r_mat)

    for i in range(50):
        update(true_state)
        measured_state = sense_state(true_state.copy())
        kf(cmd, measured_state)
        true_state_hist.append(true_state.copy())

    true_state_hist = np.array(true_state_hist)

    print("True State: ", [round(ts, 3) for ts in true_state])
    print("KF State: ", [round(xi, 3) for xi in kf.x])

    plt.plot(true_state_hist[:, 0], true_state_hist[:, 2], 'r--', label="True")
    plt.plot(kf.state_hist[:, 0], kf.state_hist[:, 2], 'b--', label="Kalman Filter")
    plt.grid()
    plt.legend()
    plt.xlim(0., 5.)
    plt.ylim(0., 5.)
    plt.show()

