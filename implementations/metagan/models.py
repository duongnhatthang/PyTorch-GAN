import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class MultiInputGeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(MultiInputGeneratorResNet, self).__init__()

        channels = input_shape[0]

        def _get_encoder():
            # Initial convolution block
            out_features = 64
            model = [
                nn.ReflectionPad2d(channels),
                nn.Conv2d(channels, out_features, 7),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

            # Downsampling
            for _ in range(2):
                out_features *= 2
                model += [
                    nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                ]
                in_features = out_features
            return model, out_features
        model_T, out_features_T = _get_encoder()
        self.encoder_T = nn.Sequential(*model_T) #Target
        model_S, out_features_S = _get_encoder()
        self.encoder_S = nn.Sequential(*model_S) #Source
        model_S_meta, out_features_S_meta = _get_encoder() #For meta target generation
        model_S_meta2, out_features_S_meta2 = _get_encoder()
        self.encoder_S_meta2 = nn.Sequential(*model_S_meta) #For final target generation

        def _get_decoder(out_features, c):
            in_features = out_features
            model = []
            # Residual blocks
            for _ in range(num_residual_blocks):
                model += [ResidualBlock(out_features)]

            # Upsampling
            for _ in range(2):
                out_features //= 2
                model += [
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                    nn.InstanceNorm2d(out_features),
                    nn.ReLU(inplace=True),
                ]
                in_features = out_features

            # Output layer
            model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, c, 7), nn.Tanh()]
            return model
        model = model_S_meta + _get_decoder(out_features_S_meta, c=1)
        self.T_meta_generator = nn.Sequential(*model)
        self.decoder_all = nn.Sequential(*_get_decoder(out_features_T+out_features_S+out_features_S_meta2, c=channels))

    def forward(self, S, S_meta):
        T_meta_image = self.T_meta_generator(S_meta)
        T_meta_image = T_meta_image.repeat(1, 3, 1, 1)
        enc_T = self.encoder_T(T_meta_image)
        enc_S = self.encoder_S(S)
        enc_S_meta2 = self.encoder_S_meta2(S_meta)
        combine = torch.cat((enc_T, enc_S, enc_S_meta2), dim=1) #double check, (batch, channels, w, h)
        T = self.decoder_all(combine)
        return T, T_meta_image

##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class MultiInputDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(MultiInputDiscriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.encoder = nn.Sequential(*discriminator_block(channels, 32, normalize=False))
        self.encoder_meta = nn.Sequential(*discriminator_block(channels, 32, normalize=False))
        self.decoder = nn.Sequential(
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img, img_meta):
        encoder_output = self.encoder(img)
        encoder_meta_output = self.encoder_meta(img_meta)
        combine = torch.cat((encoder_output, encoder_meta_output), dim=1) #double check, (batch, channels, w, h)
        return self.decoder(combine)