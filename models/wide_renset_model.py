
from .wide_residual_network import create_wide_residual_network
import keras.backend as K
from keras.utils import plot_model

def get_model():
    # For WRN-16-8 put N = 2, k = 8
    # For WRN-28-10 put N = 4, k = 10
    # For WRN-40-4 put N = 6, k = 4

    input_shape =(32, 32, 3)
    nb_classes=100


    model = create_wide_residual_network(
        input_shape,
        nb_classes=nb_classes,
        N=4,
        k=8,
        dropout=0.0
    )

    return model


if __name__ == '__main__':
    model_test = get_model()