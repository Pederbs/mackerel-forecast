import tensorflow as tf


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU(0.2))

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, use_bn=True):
        super(ResidualBlock, self).__init__()
        self.use_bn = use_bn
        initializer = tf.random_normal_initializer(0., 0.02)
        
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.use_bn:
            self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides=1,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.use_bn:
            self.bn2 = tf.keras.layers.BatchNormalization()

        # If input and output channels differ or strides > 1, project with 1x1 conv
        if strides != 1:
            self.proj = tf.keras.layers.Conv2D(filters, 1, strides=strides,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False)
            if self.use_bn:
                self.bn_proj = tf.keras.layers.BatchNormalization()
        else:
            self.proj = None

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        if self.use_bn:
            x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x, training=training)

        # Shortcut
        if self.proj:
            shortcut = self.proj(inputs)
            if self.use_bn:
                shortcut = self.bn_proj(shortcut, training=training)
        else:
            shortcut = inputs

        x = tf.keras.layers.add([x, shortcut])
        return self.relu(x)

class BottleneckResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, bottleneck_ratio=4, strides=1, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        bottleneck_filters = filters // bottleneck_ratio
        initializer = tf.random_normal_initializer(0., 0.02)

        # 1×1 reduce
        self.conv_reduce = tf.keras.layers.Conv2D(
            bottleneck_filters, 1, strides=strides, padding='same',
            kernel_initializer=initializer, use_bias=False)
        # 3×3 spatial
        self.conv3x3 = tf.keras.layers.Conv2D(
            bottleneck_filters, 3, strides=1, padding='same',
            kernel_initializer=initializer, use_bias=False)
        # 1×1 expand
        self.conv_expand = tf.keras.layers.Conv2D(
            filters, 1, strides=1, padding='same',
            kernel_initializer=initializer, use_bias=False)

        if use_bn:
            self.bn_reduce = tf.keras.layers.BatchNormalization()
            self.bn_3x3    = tf.keras.layers.BatchNormalization()
            self.bn_expand = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()

        # projection if needed
        if strides != 1:
            self.proj = tf.keras.layers.Conv2D(
                filters, 1, strides=strides, padding='same',
                kernel_initializer=initializer, use_bias=False)
            if use_bn:
                self.bn_proj = tf.keras.layers.BatchNormalization()
        else:
            self.proj = None

    def call(self, inputs, training=False):
        x = self.conv_reduce(inputs)
        if self.use_bn: x = self.bn_reduce(x, training=training)
        x = self.relu(x)

        x = self.conv3x3(x)
        if self.use_bn: x = self.bn_3x3(x, training=training)
        x = self.relu(x)

        x = self.conv_expand(x)
        if self.use_bn: x = self.bn_expand(x, training=training)

        # shortcut
        if self.proj:
            shortcut = self.proj(inputs)
            if self.use_bn: shortcut = self.bn_proj(shortcut, training=training)
        else:
            shortcut = inputs

        return self.relu(x + shortcut)



def downsample_res(filters, size, use_bn=True, bottleneck=False):
    """
    Downsampling block: Conv2D with stride 2 followed by optional BatchNorm/InstanceNorm, LeakyReLU,
    then one ResidualBlock.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    sequential = tf.keras.Sequential()
    sequential.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False)
    )
    if use_bn:
        sequential.add(tf.keras.layers.BatchNormalization())
    sequential.add(tf.keras.layers.LeakyReLU(0.2))
    if bottleneck:
        sequential.add(BottleneckResidualBlock(filters, bottleneck_ratio=4, strides=1, use_bn=use_bn))
    else:
        sequential.add(ResidualBlock(filters, kernel_size=3, strides=1, use_bn=use_bn))
    return sequential

def upsample_res(filters, size, apply_dropout=False, use_bn=True, bottleneck=False):
    """
    Upsampling block: Conv2DTranspose with stride 2, BatchNorm/InstanceNorm, optional Dropout,
    ReLU, then one ResidualBlock.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    sequential = tf.keras.Sequential()
    sequential.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False)
    )
    if use_bn:
        sequential.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        sequential.add(tf.keras.layers.Dropout(0.5))
    sequential.add(tf.keras.layers.ReLU())
    if bottleneck:
        sequential.add(BottleneckResidualBlock(filters, bottleneck_ratio=4, strides=1, use_bn=use_bn))
    else:
        sequential.add(ResidualBlock(filters, kernel_size=3, strides=1, use_bn=use_bn))
    return sequential


def Generator(height: int = 256, width: int = 256, input_channels: int = 3, output_channels: int = 3) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

    down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator(height: int = 256, width: int = 256, input_channles: int = 3, output_channels: int = 3) -> tf.keras.Model:
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[height, width, input_channles], name='input_image')
    tar = tf.keras.layers.Input(shape=[height, width, output_channels], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU(0.2)(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


def Generator_minimal_residual(height: int = 256,
                               width: int = 256,
                               input_channels: int = 3,
                               output_channels: int = 3) -> tf.keras.Model:
    """
    U-Net generator with integrated residual blocks in the 4 deepest layers.
    """
    inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

    # Encoder (Downsampling)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),                   # (bs, 128, 128, 64)
        downsample(128, 4),                                         # (bs, 64, 64, 128)
        downsample(256, 4),                                         # (bs, 32, 32, 256)
        downsample(512, 4),                                         # (bs, 16, 16, 512)
        downsample(512, 4),                                         # (bs, 8, 8, 512)
        downsample_res(512, 4),                                     # (bs, 4, 4, 512)
        downsample_res(512, 4, bottleneck=True),                    # (bs, 2, 2, 512)
        downsample_res(512, 4, bottleneck=True),                    # (bs, 1, 1, 512)
    ]

    # Decoder (Upsampling)
    up_stack = [
        upsample_res(512, 4, apply_dropout=True, bottleneck=True),  # (bs, 2, 2, 512)
        upsample_res(512, 4, apply_dropout=True),                   # (bs, 4, 4, 512)
        upsample_res(512, 4, apply_dropout=True),                   # (bs, 8, 8, 512)
        upsample(512, 4),                                           # (bs, 16, 16, 512)
        upsample(256, 4),                                           # (bs, 32, 32, 256)
        upsample(128, 4),                                           # (bs, 64, 64, 128)
        upsample(64, 4),                                            # (bs, 128, 128, 64)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 256, 256, out)

    x = inputs
    skips = []

    # Apply downsampling
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # Reverse skips, drop the bottleneck
    skips = reversed(skips[:-1])

    # Apply upsampling and skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


##########################################################################################################
######################################          DEPRICATED          ######################################
##########################################################################################################


# # unet-with-residual: used batch normalization
# def Generator_residual(height: int = 256,
#                        width: int = 256,
#                        input_channels: int = 3,
#                        output_channels: int = 3) -> tf.keras.Model:
#     """
#     U-Net generator with integrated residual blocks in each down- and up-sampling stage.
#     """
#     inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

#     # Encoder (Downsampling)
#     down_stack = [
#         downsample_res(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
#         downsample_res(128, 4),                        # (bs, 64, 64, 128)
#         downsample_res(256, 4),                        # (bs, 32, 32, 256)
#         downsample_res(512, 4),                        # (bs, 16, 16, 512)
#         downsample_res(512, 4),                        # (bs, 8, 8, 512)
#         downsample_res(512, 4),                        # (bs, 4, 4, 512)
#         downsample_res(512, 4),                        # (bs, 2, 2, 512)
#         downsample_res(512, 4),                        # (bs, 1, 1, 512)
#     ]

#     # Decoder (Upsampling)
#     up_stack = [
#         upsample_res(512, 4, apply_dropout=True),  # (bs, 2, 2, 512)
#         upsample_res(512, 4, apply_dropout=True),  # (bs, 4, 4, 512)
#         upsample_res(512, 4, apply_dropout=True),  # (bs, 8, 8, 512)
#         upsample_res(512, 4),                     # (bs, 16, 16, 512)
#         upsample_res(256, 4),                     # (bs, 32, 32, 256)
#         upsample_res(128, 4),                     # (bs, 64, 64, 128)
#         upsample_res(64, 4),                      # (bs, 128, 128, 64)
#     ]

#     initializer = tf.random_normal_initializer(0., 0.02)
#     last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
#                                            strides=2,
#                                            padding='same',
#                                            kernel_initializer=initializer,
#                                            activation='tanh')  # (bs, 256, 256, out)

#     x = inputs
#     skips = []

#     # Apply downsampling
#     for down in down_stack:
#         x = down(x)
#         skips.append(x)

#     # Reverse skips, drop the bottleneck
#     skips = reversed(skips[:-1])

#     # Apply upsampling and skip connections
#     for up, skip in zip(up_stack, skips):
#         x = up(x)
#         x = tf.keras.layers.Concatenate()([x, skip])

#     x = last(x)

#     return tf.keras.Model(inputs=inputs, outputs=x)



# Depricated: used for another dataset
# Discriminator/Generator for 128x512 images, used for Lofoten dataset


def Discriminator286(
    height: int = 256,
    width: int = 256,
    input_channels: int = 3,
    target_channels: int = 3,
) -> tf.keras.Model:
    
    init = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input([height, width, input_channels], name="inp")
    tar = tf.keras.layers.Input([height, width, target_channels], name="tar")
    x = tf.keras.layers.Concatenate()([inp, tar])  # → 256×256

    # -- five downsampling convs (stride=2), doubling filters up to 512
    d = downsample( 64, 4, apply_batchnorm=False)(x)  # 256→128, 64 channels
    d = downsample(128, 4)(d)                        # 128→ 64,128 ch
    d = downsample(256, 4)(d)                        # 64 → 32,256 ch
    d = downsample(512, 4)(d)                        # 32 → 16,512 ch
    d = downsample(512, 4)(d)                        # 16 →  8,512 ch

    # -- one RF-expanding conv (stride=1)
    d = tf.keras.layers.ZeroPadding2D()(d)           # 8→10
    d = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=init, use_bias=False
    )(d)                                             # 10→ 7, 512 ch
    d = tf.keras.layers.BatchNormalization()(d)
    d = tf.keras.layers.LeakyReLU(0.2)(d)

    # -- final 1-channel conv (stride=1)
    d = tf.keras.layers.ZeroPadding2D()(d)           # 7→9
    out = tf.keras.layers.Conv2D(
        1, 4, strides=1, kernel_initializer=init
    )(d)                                             # 9→ 6, 1 ch

    return tf.keras.Model(inputs=[inp, tar], outputs=out)



# for the Lofoten dataset, which has a different size   

def Generator_128_512(height: int = 128, width: int = 512, input_channels: int = 3, output_channels: int = 3) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=[height, width, input_channels])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 64, 256, 64)
        downsample(128, 4),  # (batch_size, 32, 128, 128)
        downsample(256, 4),  # (batch_size, 16, 64, 256)
        downsample(512, 4),  # (batch_size, 8, 32, 512)
        downsample(512, 4),  # (batch_size, 4, 16, 512)
        downsample(512, 4),  # (batch_size, 2, 8, 512)
        downsample(512, 4),  # (batch_size, 1, 4, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 8, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 16, 1024)
        upsample(512, 4),  # (batch_size, 8, 32, 1024)
        upsample(256, 4),  # (batch_size, 16, 64, 512)
        upsample(128, 4),  # (batch_size, 32, 128, 256)
        upsample(64, 4),  # (batch_size, 64, 256, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def Discriminator_128_512(height: int = 128, width: int = 512, input_channels: int = 3, output_channels: int = 3) -> tf.keras.Model:
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[height, width, input_channels], name='input_image')
    tar = tf.keras.layers.Input(shape=[height, width, output_channels], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 128, 512, channels*2)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 64, 256, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 32, 128, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 16, 64, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 18, 66, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  # (batch_size, 17, 65, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU(0.2)(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 18, 66, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)
