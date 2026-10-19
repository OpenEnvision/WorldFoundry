# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
# License text: THIRD-PARTY-NOTICES, BigVGAN HiFi-GAN LICENSE_1.

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
