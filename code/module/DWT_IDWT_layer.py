import numpy as np
import math
from torch.nn import Module
from DWT_IDWT_Functions import *
import pywt
class IDWTFunction_3D(Function):
    @staticmethod
    def forward(ctx, input_LLL, input_LLH, input_LHL, input_LHH,
                     input_HLL, input_HLH, input_HHL, input_HHH,
                     matrix_Low_0, matrix_Low_1, matrix_Low_2,
                     matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
        input_LL = torch.add(torch.matmul(matrix_Low_2.t(), input_LLL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HLL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_LH = torch.add(torch.matmul(matrix_Low_2.t(), input_LLH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HLH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_HL = torch.add(torch.matmul(matrix_Low_2.t(), input_LHL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HHL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_HH = torch.add(torch.matmul(matrix_Low_2.t(), input_LHH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HHH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()), torch.matmul(input_LH, matrix_High_1.t()))
        input_H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()), torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), input_L), torch.matmul(matrix_High_0.t(), input_H))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        grad_LH = torch.matmul(grad_L, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        grad_HL = torch.matmul(grad_H, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        grad_HH = torch.matmul(grad_H, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        grad_LLL = torch.matmul(matrix_Low_2, grad_LL).transpose(dim0 = 2, dim1 = 3)
        grad_LLH = torch.matmul(matrix_Low_2, grad_LH).transpose(dim0 = 2, dim1 = 3)
        grad_LHL = torch.matmul(matrix_Low_2, grad_HL).transpose(dim0 = 2, dim1 = 3)
        grad_LHH = torch.matmul(matrix_Low_2, grad_HH).transpose(dim0 = 2, dim1 = 3)
        grad_HLL = torch.matmul(matrix_High_2, grad_LL).transpose(dim0 = 2, dim1 = 3)
        grad_HLH = torch.matmul(matrix_High_2, grad_LH).transpose(dim0 = 2, dim1 = 3)
        grad_HHL = torch.matmul(matrix_High_2, grad_HL).transpose(dim0 = 2, dim1 = 3)
        grad_HHH = torch.matmul(matrix_High_2, grad_HH).transpose(dim0 = 2, dim1 = 3)
        return grad_LLL, grad_LLH, grad_LHL, grad_LHH, grad_HLL, grad_HLH, grad_HHL, grad_HHH, None, None, None, None, None, None
class DWT_3D(Module):
    """
    input: the 3D data to be decomposed -- (N, C, D, H, W)
    output: lfc -- (N, C, D/2, H/2, W/2)
            hfc_llh -- (N, C, D/2, H/2, W/2)
            hfc_lhl -- (N, C, D/2, H/2, W/2)
            hfc_lhh -- (N, C, D/2, H/2, W/2)
            hfc_hll -- (N, C, D/2, H/2, W/2)
            hfc_hlh -- (N, C, D/2, H/2, W/2)
            hfc_hhl -- (N, C, D/2, H/2, W/2)
            hfc_hhh -- (N, C, D/2, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        3D discrete wavelet transform (DWT) for 3D data decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )# 1,3
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )# 2,3
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        # print(self.band_length,"matrix_h",matrix_h.shape,"matrix_g",matrix_g.shape)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):#2
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, input):
        """
        :param input: the 3D data to be decomposed
        :return: the eight components of the input data, one low-frequency and seven high-frequency components
        """
        # assert len(input.size()) == 8
        # print("input",input.shape)
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_3D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                           self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)


class IDWT_3D(Module):
    """
    input:  lfc -- (N, C, D/2, H/2, W/2)
            hfc_llh -- (N, C, D/2, H/2, W/2)
            hfc_lhl -- (N, C, D/2, H/2, W/2)
            hfc_lhh -- (N, C, D/2, H/2, W/2)
            hfc_hll -- (N, C, D/2, H/2, W/2)
            hfc_hlh -- (N, C, D/2, H/2, W/2)
            hfc_hhl -- (N, C, D/2, H/2, W/2)
            hfc_hhh -- (N, C, D/2, H/2, W/2)
    output: the original 3D data -- (N, C, D, H, W)
    """
    def __init__(self, wavename):
        """
        3D inverse DWT (IDWT) for 3D data reconstruction
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        """
        :param LLL: the low-frequency component, lfc
        :param LLH: the high-frequency componetn, hfc_llh
        :param LHL: the high-frequency componetn, hfc_lhl
        :param LHH: the high-frequency componetn, hfc_lhh
        :param HLL: the high-frequency componetn, hfc_hll
        :param HLH: the high-frequency componetn, hfc_hlh
        :param HHL: the high-frequency componetn, hfc_hhl
        :param HHH: the high-frequency componetn, hfc_hhh
        :return: the original 3D input data
        """
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == 5
        assert len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        self.input_depth = LLL.size()[-3] + HHH.size()[-3]
        self.input_height = LLL.size()[-2] + HHH.size()[-2]
        self.input_width = LLL.size()[-1] + HHH.size()[-1]
        self.get_matrix()
        return IDWTFunction_3D.apply(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                                     self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                     self.matrix_high_0, self.matrix_high_1, self.matrix_high_2)


if __name__ == '__main__':
    from datetime import datetime
    from torch.autograd import gradcheck
    wavelet = pywt.Wavelet('bior1.1')
    h = wavelet.rec_lo
    g = wavelet.rec_hi
    h_ = wavelet.dec_lo
    g_ = wavelet.dec_hi
    h_.reverse()
    g_.reverse()
