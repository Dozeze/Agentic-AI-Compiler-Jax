import jax


def candidate(x, w):
    return jax.lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
