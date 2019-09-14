import torch
import torch.nn as nn

from pytorch_wavelets import DWTForward, DWTInverse


# ##############################################################################################3
# 原图的首次进行特征的提取，保持feature map 数量160
class Block_of_DMT1(nn.Module):
    def __init__(self):
        super(Block_of_DMT1, self).__init__()

        # DMT1
        self.conv1_1 = nn.Conv2d(
            in_channels=160, out_channels=160, kernel_size=3, stride=1, padding=1
        )
        self.bn1_1 = nn.BatchNorm2d(160, affine=True)
        self.relu1_1 = nn.ReLU()

    def forward(self, x):
        output = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        return output


# 第一次下采样后保持256的feature map
class Block_of_DMT2(nn.Module):
    def __init__(self):
        super(Block_of_DMT2, self).__init__()

        # DMT1
        self.conv2_1 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn2_1 = nn.BatchNorm2d(256, affine=True)
        self.relu2_1 = nn.ReLU()

    def forward(self, x):
        output = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        return output


# 第三次下采样，保持feature map为256
class Block_of_DMT3(nn.Module):
    def __init__(self):
        super(Block_of_DMT3, self).__init__()

        # DMT1
        self.conv3_1 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn3_1 = nn.BatchNorm2d(256, affine=True)
        self.relu3_1 = nn.ReLU()

    def forward(self, x):
        output = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        return output


# MWCNN
class MWCNN(nn.Module):
    def __init__(self):  ##play attention the upscales
        super(MWCNN, self).__init__()

        self.DWT = DWTForward(J=1, wave="haar").cuda()
        self.IDWT = DWTInverse(wave="haar").cuda()

        # DMT1 operation
        # DMT1
        # 因为对原图首先做了一次DWT，导致通道数变为原来的4倍调用保持的feature map的函数可以保持通道的正常
        self.conv_DMT1 = nn.Conv2d(
            in_channels=3 * 4, out_channels=160, kernel_size=3, stride=1, padding=1
        )
        self.bn_DMT1 = nn.BatchNorm2d(160, affine=True)
        self.relu_DMT1 = nn.ReLU()
        # IDMT1  逆变换
        self.conv_IDMT1 = nn.Conv2d(
            in_channels=160, out_channels=3 * 4, kernel_size=3, stride=1, padding=1
        )
        # feature map 保持
        self.blockDMT1 = self.make_layer(Block_of_DMT1, 3)

        # DMT2 operation
        # DMT2
        self.conv_DMT2 = nn.Conv2d(
            in_channels=640, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn_DMT2 = nn.BatchNorm2d(256, affine=True)
        self.relu_DMT2 = nn.ReLU()
        # IDMT2
        self.conv_IDMT2 = nn.Conv2d(
            in_channels=256, out_channels=640, kernel_size=3, stride=1, padding=1
        )
        self.bn_IDMT2 = nn.BatchNorm2d(640, affine=True)
        self.relu_IDMT2 = nn.ReLU()

        self.blockDMT2 = self.make_layer(Block_of_DMT2, 3)

        # DMT3 operation
        # DMT3
        self.conv_DMT3 = nn.Conv2d(
            in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn_DMT3 = nn.BatchNorm2d(256, affine=True)
        self.relu_DMT3 = nn.ReLU()
        # IDMT3
        self.conv_IDMT3 = nn.Conv2d(
            in_channels=256, out_channels=1024, kernel_size=3, stride=1, padding=1
        )
        self.bn_IDMT3 = nn.BatchNorm2d(1024, affine=True)
        self.relu_IDMT3 = nn.ReLU()

        self.blockDMT3 = self.make_layer(Block_of_DMT3, 3)
        self.try_conv1 = nn.Conv2d(160, 80, 3, 1, 1)
        # self.try_bn = nn.BatchNorm2d(20)
        self.try_conv2 = nn.Conv2d(80, 12, 3, 1, 1)
        # self.try_conv3 = nn.Conv2d(10, 3, 3, 1, 1)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    # 对小波变换后进行融合，但是融合方法还带确定
    def _transformer(self, DMT1_yl, DMT1_yh):
        # print("trans,", DMT1_yh)
        list_tensor = []
        # (yl, yh)
        #                 tuple of lowpass (yl) and bandpass (yh)
        #                 coefficients. yh is a list of length J with the first entry
        #                 being the finest scale coefficients. yl has shape
        #                 :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
        #                 :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
        #                 dimension in yh iterates over the LH, HL and HH coefficients
        #
        for i in range(3):
            list_tensor.append(DMT1_yh[0][:, :, i, :, :])
        list_tensor.append(DMT1_yl)
        return torch.cat(list_tensor, 1)

    def _Itransformer(self, out):
        # w = pywt.Wavelet('haar')

        # sz=2*(len(w.dec_lo) // 2 - 1)
        # if yl.shape[-2] % 2 == 1 and yl.shape[-1] % 2 == 1:
        # yl = F.pad(yl, (sz, sz+1, sz, sz+1), mode='reflect')
        # elif yl.shape[-2] % 2 == 1:
        # yl = F.pad(yl, (sz, sz+1, sz, sz), mode='reflect')
        # elif yl.shape[-1] % 2 == 1:
        # yl = F.pad(yl, (sz, sz, sz, sz+1), mode='reflect')
        # else:
        # yl = F.pad(yl, (sz, sz, sz, sz), mode='reflect')
        # 原本的解决方案
        yh = []
        C = out.shape[1] / 4
        # print("c", C)
        y = out.reshape((out.shape[0], int(C), 4, out.shape[-2], out.shape[-1]))
        yl = y[:, :, 0].contiguous()
        yh.append(y[:, :, 1:].contiguous())

        return yl, yh

    def forward(self, x):  #
        print("int_size", x.shape)
        DMT1_p = x
        # DMT1
        DMT1_yl, DMT1_yh = self.DWT(x)
        DMT1 = self._transformer(DMT1_yl, DMT1_yh)
        # print("dmt1 size",DMT1.shape)
        out = self.relu_DMT1(self.bn_DMT1(self.conv_DMT1(DMT1)))
        # 保持160 的feature map,经过3次的特征提取
        out = self.blockDMT1(out)
        # ##160，

        DMT2_p = out
        # print("DMT2", DMT2_p.shape)
        # DMT2
        DMT2_yl, DMT2_yh = self.DWT(out)
        # 经过小波变换后 fearture map 升到640，符合论文
        DMT2 = self._transformer(DMT2_yl, DMT2_yh)
        # print("dmt2 size",DMT2.shape)
        out = self.relu_DMT2(self.bn_DMT2(self.conv_DMT2(DMT2)))
        # keep 256
        out = self.blockDMT2(out)  ###256

        DMT3_p = out
        # print("DMT3", DMT3_p.shape)
        # DMT3
        DMT3_yl, DMT3_yh = self.DWT(out)
        DMT3 = self._transformer(DMT3_yl, DMT3_yh)
        out = self.relu_DMT3(self.bn_DMT3(self.conv_DMT3(DMT3)))
        # keep 256
        out = self.blockDMT3(out)
        ###256
        # print("OUT", out.shape)

        # IDMT3
        out = self.blockDMT3(out)  # DMT4
        # 256->1024
        out = self.relu_IDMT3(self.bn_IDMT3(self.conv_IDMT3(out)))
        out = self._Itransformer(out)  ###########
        IDMT3 = self.IDWT(out)
        out = IDMT3 + DMT3_p
        # print("IDMT3", IDMT3.shape)

        # IDMT2
        # 1024-256
        # 先keep 256
        out = self.blockDMT2(out)
        # 升640
        out = self.relu_IDMT2(self.bn_IDMT2(self.conv_IDMT2(out)))
        out = self._Itransformer(out)  ##############
        # 降160
        IDMT2 = self.IDWT(out)
        out = IDMT2 + DMT2_p
        # print("out_size2", IDMT2.shape)

        # IDMT1
        # keep 160
        out = self.blockDMT1(out)
        out = self.try_conv1(out)
        out = self.try_conv2(out)

        # out = self.relu_IDMT2(self.bn_IDMT2(self.conv_IDMT1(out)))

        out = self._Itransformer(out)  ###############
        IDMT1 = self.IDWT(out)
        # IDMT1 = self.try_conv1(IDMT1)
        # IDMT1 = self.try_conv2(IDMT1)
        # IDMT1 = self.try_conv3(IDMT1)
        # print("out_size", IDMT1.shape)
        out = IDMT1 + DMT1_p

        # out = self.up(out)

        return out


##########################################
if __name__ == "__main__":
    model = MWCNN()
    print(model)
