
from .wide_residual_network import create_wide_residual_network
from .wide_resnet import WideResidualNetwork
import keras.backend as K
from keras.utils import plot_model


def get_model():
    input_shape = (32, 32, 3)
    n_classes = 100

    model = WideResidualNetwork(
        input_shape=input_shape,
        depth=28,
        width=8,
        classes=n_classes,
        weights=None,
        dropout_rate=0.0
    )

    return model


if __name__ == '__main__':
    model_test = get_model()