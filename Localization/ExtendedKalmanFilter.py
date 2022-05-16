import numpy as np
from math import sin, cos
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, x, r=np.eye(4), q=np.eye(4)):
        self.x = x
        self.P = np.eye(3)
        self.H = np.eye(3)
        self.Q = q
        self.R = r

        self.state_hist = np.array([self.x.copy()])

    def F(self, v=0., w=0., ang=0.):
        F = np.eye(3)
        F[0, 2] = -(v * sin(ang))
        F[1, 2] = v * cos(ang)
        F[2, 2] = 0.
        return F

    def motion_model(self, prev, ctl):
        x = prev[0]
        y = prev[1]
        thet = prev[2]

        v = ctl[0]
        w = ctl[1]

        x_k1 = x + (v * cos(thet) * DT)
        y_k1 = y + (v * sin(thet) * DT)
        thet_k1 = thet + (w * DT)

        return np.array([x_k1, y_k1, thet_k1])

    def predict(self, controls=np.zeros(2)):
        F = self.F(controls[0], controls[1], self.x[2])
        self.x = self.motion_model(self.x, controls)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        K2 = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        K = self.P @ self.H.T @ K2

        self.x += K @ (z - self.H @ self.x)
        self.P = (np.eye(3) - K @ self.H) @ self.P

        self.state_hist = np.vstack((self.state_hist, self.x))

    def __call__(self, controls, z):
        self.predict(controls)
        self.update(z)


def update(states, controls):
    pos_noise = np.random.normal(0., .01, 2)
    thet_noise = np.random.normal(0., .01, 1)

    states[0] += ((controls[0] * cos(states[2])) * DT) + pos_noise[0]
    states[1] += ((controls[0] * sin(states[2])) * DT) + pos_noise[1]
    states[2] += (controls[1] * DT) + thet_noise[0]


# Must pass a copy of 'states' to this
def sense_state(states):
    pos_noise = np.random.normal(0., .075, 2)
    thet_noise = np.random.normal(0., .075, 1)

    states[0] += pos_noise[0]
    states[1] += pos_noise[1]
    states[2] += thet_noise[0]

    return states


if __name__ == "__main__":
    DT = .1
    #                      x   y  thet
    true_state = np.array([0., .5, 0.])
    true_state_hist = [true_state.copy()]

    cov_est = np.zeros((3, 3))
    cmd = np.array([1.5, 0.35])

    q_mat = np.eye(3) * .1  # Motion Model
    r_mat = np.eye(3) * .025  # Sensor Model
    init_measured_state = sense_state(true_state.copy())
    kf = KalmanFilter(init_measured_state, q_mat, r_mat)
    measured_state_hist = [init_measured_state.copy()]

    err = 0.
    measured_err = 0.

    for i in range(100):
        update(true_state, cmd)
        measured_state = sense_state(true_state.copy())
        kf(cmd, measured_state)
        true_state_hist.append(true_state.copy())
        measured_state_hist.append(measured_state.copy())
        err += np.sum(np.abs(true_state - kf.x))
        measured_err += np.sum(np.abs(true_state - measured_state))

    true_state_hist = np.array(true_state_hist)
    measured_state_hist = np.array(measured_state_hist)

    print("True State: ", [round(ts, 3) for ts in true_state])
    print("KF State: ", [round(xi, 3) for xi in kf.x])
    print("KF Covariance: ")
    print(kf.P)

    print("--------------------------------------")
    print("Kalman Filter Err: ", err)
    print("Measured Err: ", measured_err)

    plt.plot(true_state_hist[:, 0], true_state_hist[:, 1], 'r--', label="True")
    plt.plot(kf.state_hist[:, 0], kf.state_hist[:, 1], 'b--', label="Kalman Filter")
    plt.plot(measured_state_hist[:, 0], measured_state_hist[:, 1], 'gx', label="Measured")
    plt.grid()
    plt.legend()
    plt.xlim(-5., 5.)
    plt.ylim(0., 10.)
    plt.show()