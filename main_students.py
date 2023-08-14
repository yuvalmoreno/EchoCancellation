import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from model import NN_class
import scipy.signal as signal
import scipy

# By: Yuval Moreno


def FourierCoeffGen(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    w0 = 2 * np.pi / N
    exp = np.exp(-1j * w0 * k * n)

    FourierCoeff = np.dot(exp, signal)
    return FourierCoeff


def FourierSeries(Coeffs):
    N = len(Coeffs)
    k = np.arange(N)
    n = k.reshape((N, 1))
    w0 = 2 * np.pi / N
    exp = np.exp(1j * w0 * n * k)

    signal = 1 / N * np.dot(exp, Coeffs)
    return signal


def xcorr(x, y):
    """
    Perform Cross-Correlation on x and y
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """
    corr = signal.correlate(x, y, mode="full")
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    return lags, corr


class Inference:
    def __init__(self, model):
        self.model = model

        self.epsilon = 10 ** -12
        self.alpha = 0.997
        self.window_len = 512

        self.K = self.window_len // 2 + 1

        self.M = np.zeros([self.K * 2, 1])
        self.S = np.ones([self.K * 2, 1])

        self.window = np.hamming(self.window_len)[:, np.newaxis]

    def preprocessing(self, mic_buffer, ref_signal):
        frame = np.zeros([2 * self.K, 1], dtype=complex)
        frame[:: 2, :] = mic_buffer[:self.K, :] / (2 ** 15)
        frame[1::2, :] = ref_signal[:self.K, :] / (2 ** 15)

        frame = np.log10(np.maximum(abs(frame), self.epsilon))

        self.M = self.alpha * self.M + (1 - self.alpha) * frame
        self.S = self.alpha * self.S + (1 - self.alpha) * abs(frame) ** 2

        frame = (frame - self.M) / (np.sqrt(self.S - self.M ** 2) + self.epsilon)

        return frame

    def forward(self, mic, ref):
        idx = 0
        overlap_factor = 0.75
        output = np.zeros([len(mic), 1], dtype=complex)

        while idx + self.window_len < len(mic):
            # Get input buffers
            mic_buffer = np.atleast_2d(mic[idx:(idx + self.window_len)]).T
            ref_buffer = np.atleast_2d(ref[idx:(idx + self.window_len)]).T

            # Transform inputs to the frequency domain
            # mic_buffer_coeff = FourierCoeffGen(mic_buffer * self.window) # Much slower runtime
            # ref_buffer_coeff = FourierCoeffGen(ref_buffer * self.window) # Much slower runtime
            mic_buffer_coeff = np.fft.fft((mic_buffer * self.window), axis=0)  # For faster runtime
            ref_buffer_coeff = np.fft.fft((ref_buffer * self.window), axis=0)  # For faster runtime

            # Pre-processing
            frame = self.preprocessing(mic_buffer=mic_buffer_coeff, ref_signal=ref_buffer_coeff)

            # Execute Neural Network - using frame, mic_buffer_coeff as inputs
            output_net = model.forward(frame, mic_buffer_coeff)

            # Overlap and add
            # output_net_time = FourierSeries(output_net)  # Much slower runtime
            output_net_time = np.fft.ifft(output_net, axis=0) * self.window  # For faster runtime
            output[idx:idx + self.window_len] += output_net_time

            # Update index
            idx = idx + int(self.window_len * (1 - overlap_factor))

        return output


if __name__ == "__main__":
    """
    Note!
    if there is a single code line to complete, it is marked as: 
        model = ...# TODO: complete code here
    if there is a code block to complete, it is marked as:
        # ----------
        # TODO: complete code here
        # ----------
    """

    inference_mode = 'our example'  # 'our example' or 'your record'
    if inference_mode == 'our example':
        # Load data - our example
        fs = 48000
        mic = np.memmap("2tFqz9scnUmF2PX04sNXfg_doubletalk_mic_48kHz_new.pcm", dtype='h', mode='r')
        ref = np.memmap("2tFqz9scnUmF2PX04sNXfg_doubletalk_lpb_48kHz_new.pcm", dtype='h', mode='r')
    elif inference_mode == 'your record':
        # Generate your own data - notice that you need to use your speaker and not your headphone
        fs = 48000
        ref = np.memmap("2tFqz9scnUmF2PX04sNXfg_doubletalk_lpb_48kHz_new.pcm", dtype='h', mode='r')  # 48kHz

        # for first time- record your voice from device microphone while playing reference signal from speaker
        '''myrecording = sd.playrec(ref, fs, channels=1)
        sd.wait()
        myrecording = np.squeeze(myrecording)
        scipy.io.wavfile.write('my_record.pcm', fs, myrecording.astype(np.int16))  # save as pcm file'''

        # for next time you can use the pcm file you just created with the following code
        myrecording = np.memmap("my_record.pcm", dtype='h', mode='r')  # 48kHz

        # Cancel noise from device microphone
        myrecording2 = myrecording[len(myrecording) - len(ref):]
        sd.play(myrecording2, fs)
        sd.wait()

        # Cancel delay between signal reference and your recording voice
        myrecording2_norm = myrecording2 / np.max(np.abs(myrecording2))
        ref_norm = ref / np.max(np.abs(ref))
        lags, corr = xcorr(myrecording2_norm, ref_norm)
        delay = lags[np.argmax(corr)]
        myrecording2 = myrecording2[delay:]
        ref = ref[:len(myrecording2)]
        mic = myrecording2

    # Downsample data
    # ----------
    mic = mic[::3]  # Decimate by 3
    ref = ref[::3]  # Decimate by 3
    fs = int(fs / 3)
    # ----------

    # # Play input data
    sd.play(mic, fs)
    sd.wait()
    sd.play(ref, fs)
    sd.wait()

    # Load model
    model = NN_class()

    # Define Inference, and run input through model
    inference = Inference(model)
    output = np.real(inference.forward(mic, ref))
    # Normalize the output
    output = 0.99 * output / max(abs(output))

    # Play and plot the output
    # -------------
    sd.play(output, fs)
    sd.wait()
    plt.plot(output)
    plt.show()
    # -------------

    # Save clean output
    scipy.io.wavfile.write(f'model_out_{inference_mode}.pcm', fs, output.astype(np.int16))  # save as pcm file
