
from .densenet import DenseNet, DenseNetFCN
import keras.backend as K
from keras.utils import plot_model

batch_size = 64
nb_classes = 100
nb_epoch = 15



def get_model():

    img_rows, img_cols = 32, 32
    img_channels = 3

    img_dim = \
        (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)

    depth = 40
    nb_dense_block = 3
    growth_rate = 12
    nb_filter_first_conv_layer = 16
    bottleneck = False
    reduction = 0.0
    dropout_rate = 0.0  # 0.0 for data augmentation

    model = DenseNet(
        img_dim,
        classes=nb_classes,
        depth=depth,
        nb_dense_block=nb_dense_block,
        growth_rate=growth_rate,
        nb_filter=nb_filter_first_conv_layer,
        dropout_rate=dropout_rate,
        bottleneck=bottleneck,
        reduction=reduction,
        weights=None,
        subsample_initial_block=False,
    )

    return model


if __name__ == '__main__':

    model = get_model()
    plot_model(model, to_file='model.eps', show_shapes=True)

    model.summary()