import numpy as np

class Initializer:
    @staticmethod
    def he(fan_in, fan_out):
        return np.random.randn(fan_out, fan_in) * np.sqrt(2 / fan_in)

    @staticmethod
    def xavier(fan_in, fan_out):
        return np.random.randn(fan_out, fan_in) * np.sqrt(1 / fan_in)

    @staticmethod
    def xavier_uniform(fan_in, fan_out):
        return np.random.randn(fan_out, fan_in) * np.sqrt(2 / (fan_in + fan_out))
