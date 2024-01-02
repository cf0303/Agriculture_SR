from torchinfo import summary

from DLCs.FLOPs import profile

import torch.nn as nn

from DLCs.model_imdn            import IMDN
from DLCs.model_rfdn            import RFDN
from DLCs.model_bsrn            import BSRN
from DLCs.model_esrt            import ESRT


from _private_models.Prop_9_NEW_Ab_53 import model_proposed

model = IMDN(upscale=4)
#model = RFDN(upscale=4)
#model = BSRN(upscale=4)
#model = ESRT(upscale=4)
#model = CNCAN()
'''
class _model(nn.Module):
    def __init__(self):
        super(_model, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(96, 384, 1, padding=0),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.Conv2d(384, 96, 1, padding=0)
        )

    def forward(self, x):
        return self.layer_1(x)

model = _model()
'''
#_B, _C, _H, _W = 1, 3, 160, 120
#_B, _C, _H, _W = 1, 3, 640, 480

#_B, _C, _H, _W = 1, 3, 120, 160
#_B, _C, _H, _W = 1, 3, 480, 640

print("\n --- Info ---\n")
summary(model, input_size=(_B, _C, _H, _W))
print("\n --- FLOPs ---\n")
flops, params = profile(model, input_size=(_B, _C, _H, _W))
print('Input: {} x {}, flops: {:.10f} GFLOPs, params: {}'.format( _H, _W,flops/(1e9),params))
#print("Prop")
