import numpy as np

class OperatorKit:
    def __init__(self, P, m1, n1, upsampling_ratio = 4):
        
        self.center_x_low = int(n1 / 2)
        self.center_y_low = int(m1 / 2)      
        
        self.P = P
        
        P_inv = P.copy()
        self.mask_non_zero = (P != 0)
        P_inv[self.mask_non_zero] = 1.0 / P[self.mask_non_zero]
        self.P_inv = P_inv
        
        self.m1 = m1
        self.n1 = n1
        
        self.m = self.m1 * upsampling_ratio
        self.n = self.n1 * upsampling_ratio
        
        self.upsampling_ratio = upsampling_ratio

    def get_bbx_indices(self, xc, yc):
        yl = yc - self.center_y_low
        yh = yl + self.m1
        
        xl = xc - self.center_x_low
        xh = xl + self.n1
        return xl, xh, yl, yh
    
    def get_diag_of_gram(self, shifted_center_x, shifted_center_y):
        # compute gram: mat(diag(H' * H))
        
        gram = np.zeros((self.m, self.n))
        
        P_square = np.absolute(self.P) ** 2
        
        for i in range(shifted_center_x.shape[0]):
            xc = shifted_center_x[i]
            yc = shifted_center_y[i]
        
            xl, xh, yl, yh = self.get_bbx_indices(xc, yc)
        
            gram[yl:yh, xl:xh] += P_square
        gram /= (self.upsampling_ratio ** 4) * self.m1 * self.n1
        return gram
    
    def FFT(self, o):
        # fft + shift
        return np.fft.fftshift(np.fft.fft2(o), axes = (-2, -1))
    
    def iFFT(self, o_hat):
        # shift + ifft
        return np.fft.ifft2(np.fft.ifftshift(o_hat, axes = (-2, -1)))

    def H_l_operator(self, o_hat, xc, yc):
        xl, xh, yl, yh = self.get_bbx_indices(xc, yc)
        
        o_hat_cropped = o_hat[yl:yh, xl:xh]
        return self.iFFT(self.P * o_hat_cropped) / (self.upsampling_ratio ** 2)
    
    def H_l_inv_operator(self, y_l, xc, yc, o_hat_init = 0):
        xl, xh, yl, yh = self.get_bbx_indices(xc, yc)
        
        pupil_mask = np.zeros((self.m, self.n)).astype(np.complex128)
        pupil_mask[yl:yh, xl:xh] = self.mask_non_zero.astype(np.complex128)
        
        o_hat = np.zeros((self.m, self.n)).astype(np.complex128)
        o_hat[yl:yh, xl:xh] = (self.P_inv * self.FFT(y_l)) * (self.upsampling_ratio ** 2)
        return o_hat_init * (1 - pupil_mask) + o_hat
        
    def amplitude_projector(self, y_l, am_measurement):
        return am_measurement * np.exp(1j * np.angle(y_l))
    