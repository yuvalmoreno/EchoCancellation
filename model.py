import numpy as np
import pandas as pd


class NN_class:
    def __init__(self):
        self.hidden     = 128
        self.beta       = 6
        self.window_len = 512
        self.K          = self.window_len // 2 + 1

        self.h0 = np.zeros([self.hidden, 1])
        self.h1 = np.zeros([self.hidden, 1])

        self.load_csv()

    def set_layer_0(self):
        self.W_ir = self.weight_ih_l0[:self.hidden, :]
        self.W_hr = self.weight_hh_l0[:self.hidden, :]
        self.b_ir = self.bias_ih_l0[:self.hidden, :]
        self.b_hr = self.bias_hh_l0[:self.hidden, :]
        self.W_iz = self.weight_ih_l0[self.hidden:2 * self.hidden, :]
        self.W_hz = self.weight_hh_l0[self.hidden:2 * self.hidden, :]
        self.b_iz = self.bias_ih_l0[self.hidden:2 * self.hidden, :]
        self.b_hz = self.bias_hh_l0[self.hidden:2 * self.hidden, :]
        self.W_in = self.weight_ih_l0[2 * self.hidden:, :]
        self.W_hn = self.weight_hh_l0[2 * self.hidden:, :]
        self.b_in = self.bias_ih_l0[2 * self.hidden:, :]
        self.b_hn = self.bias_hh_l0[2 * self.hidden:, :]

    def set_layer_1(self):
        self.W_ir = self.weight_ih_l1[:self.hidden, :]
        self.W_hr = self.weight_hh_l1[:self.hidden, :]
        self.b_ir = self.bias_ih_l1[:self.hidden, :]
        self.b_hr = self.bias_hh_l1[:self.hidden, :]
        self.W_iz = self.weight_ih_l1[self.hidden:2 * self.hidden, :]
        self.W_hz = self.weight_hh_l1[self.hidden:2 * self.hidden, :]
        self.b_iz = self.bias_ih_l1[self.hidden:2 * self.hidden, :]
        self.b_hz = self.bias_hh_l1[self.hidden:2 * self.hidden, :]
        self.W_in = self.weight_ih_l1[2 * self.hidden:, :]
        self.W_hn = self.weight_hh_l1[2 * self.hidden:, :]
        self.b_in = self.bias_ih_l1[2 * self.hidden:, :]
        self.b_hn = self.bias_hh_l1[2 * self.hidden:, :]

    def load_csv(self):
        self.bias_fc1     = pd.read_csv('csv/bias_fc1.csv', sep=' ', header=None).to_numpy()
        self.bias_fc2     = pd.read_csv('csv/bias_fc2.csv', sep=' ', header=None).to_numpy()
        self.bias_hh_l0   = pd.read_csv('csv/bias_hh_l0.csv', sep=' ', header=None).to_numpy()
        self.bias_hh_l1   = pd.read_csv('csv/bias_hh_l1.csv', sep=' ', header=None).to_numpy()
        self.bias_ih_l0   = pd.read_csv('csv/bias_ih_l0.csv', sep=' ', header=None).to_numpy()
        self.bias_ih_l1   = pd.read_csv('csv/bias_ih_l1.csv', sep=' ', header=None).to_numpy()

        self.weight_fc1   = pd.read_csv('csv/weight_fc1.csv', sep=' ', header=None).to_numpy()
        self.weight_fc2   = pd.read_csv('csv/weight_fc2.csv', sep=' ', header=None).to_numpy()
        self.weight_hh_l0 = pd.read_csv('csv/weight_hh_l0.csv', sep=' ', header=None).to_numpy()
        self.weight_hh_l1 = pd.read_csv('csv/weight_hh_l1.csv', sep=' ', header=None).to_numpy()
        self.weight_ih_l0 = pd.read_csv('csv/weight_ih_l0.csv', sep=' ', header=None).to_numpy()
        self.weight_ih_l1 = pd.read_csv('csv/weight_ih_l1.csv', sep=' ', header=None).to_numpy()

    def aux_fun_1(self, r):
        idx_list    = r > 0
        r[idx_list] = 1 / (1 + np.exp(-r[idx_list]))
        r[np.logical_not(idx_list)] = np.exp(r[np.logical_not(idx_list)]) / \
                                      (np.exp(r[np.logical_not(idx_list)]) + 1)
        return r

    def aux_fun_2(self, r):
        idx_list    = r > 0
        r[idx_list] = (1 - np.exp(-2 * r[idx_list])) / (1 + np.exp(-2 * r[idx_list]))
        r[np.logical_not(idx_list)] = (np.exp(2 * r[np.logical_not(idx_list)]) - 1) / \
                                      (np.exp(2 * r[np.logical_not(idx_list)]) + 1)
        return r

    def ReLU(self, r):
        idx_list    = r < 0
        r[idx_list] = 0  # TODO: Amit: I check to simpler way
        return r

    def forward_layers(self, frame, mode):
        # Choose h
        if mode == 0:
            h = self.h0
        else:
            h = self.h1

        r = self.W_ir @ frame + self.b_ir + self.W_hr @ h + self.b_hr
        r = self.aux_fun_1(r)

        z = self.W_iz @ frame + self.b_iz + self.W_hz @ h + self.b_hz
        z = self.aux_fun_1(z)

        n = self.W_in @ frame + self.b_in + r * (self.W_hn @ h + self.b_hn)
        n = self.aux_fun_2(n)
        frame = (1 - z) * n + z * h

        # Update h
        if mode == 0:
            self.h0 = frame
        else:
            self.h1 = frame

        return frame

    def forward(self, frame, mic_buffer_coeff):
        # Load first GRU layer and pass data
        self.set_layer_0()
        frame = self.forward_layers(frame, mode=0)

        # Load second GRU layer and pass data
        self.set_layer_1()
        frame = self.forward_layers(frame, mode=1)

        # FC1
        frame = self.weight_fc1 @ frame + self.bias_fc1
        frame = self.ReLU(frame)

        # FC2
        frame = self.weight_fc2 @ frame + self.bias_fc2
        frame = self.aux_fun_1(frame)

        # Post-processing
        ft     = np.exp((frame - 1) * self.beta) * mic_buffer_coeff[:self.K, :]
        output = np.zeros([self.window_len, 1], dtype=complex)
        output[:self.K] = ft
        output[self.K:] = np.flip(ft[1:self.K - 1], axis=0).conj()

        return output


if __name__ == "__main__":
    pass