import numpy as np
import torch
import torch.nn as nn
import fp_utils
import utils


class FP_model(nn.Module):

    def __init__(self, P, shifted_center_x, shifted_center_y, m1, n1, 
                 upsampling_ratio = 4,
                 pupil_parametric_model = None,
                 RUN_ON_GPU = False):
        super(FP_model, self).__init__()
        
        self.upsampling_ratio = upsampling_ratio
        
        self.optimize_pupil = pupil_parametric_model is not None
        
        if self.optimize_pupil:
            non_zero_mask = pupil_parametric_model['non_zero_mask'].astype(np.complex64)
            non_zero_mask = torch.from_numpy(non_zero_mask)
            if RUN_ON_GPU:
                non_zero_mask = non_zero_mask.cuda()
            self.non_zero_mask = non_zero_mask
            modes = utils.np_to_var(pupil_parametric_model['modes'], RUN_ON_GPU = RUN_ON_GPU)
            omega_z = utils.np_to_var(pupil_parametric_model['omega_z'], RUN_ON_GPU = RUN_ON_GPU)
            all_modes = modes
            self.all_modes = torch.moveaxis(all_modes, 0, -1)
        self.P = utils.np_to_var(P, RUN_ON_GPU = RUN_ON_GPU)
        
        self.shifted_center_x = shifted_center_x
        self.shifted_center_y = shifted_center_y
        
        self.indices_x_all, self.indices_y_all = fp_utils.get_cropping_indices(m1, n1, shifted_center_x, shifted_center_y)    
        self.num_led = shifted_center_x.shape[0]

    def get_pupil(self, coef = None):
        if (not self.optimize_pupil) or (coef is None):
            return self.P
        return torch.exp(1j * torch.matmul(self.all_modes, coef)) * self.non_zero_mask
        
    def forward(self, o_am, o_ph, selected_LED_ids, indices_frame = np.array([0]), coef = None):
        # compute intensity low-res measurement
        # selected_LED_ids: num_frame x L'
        
        self.o_ph = o_ph
        self.o_am = o_am
        self.o_complex = self.o_am * torch.exp(1j * self.o_ph)

        o_hat = torch.fft.fftshift(torch.fft.fft2(self.o_complex), dim = (-2, -1))
        
        indices_y = self.indices_y_all[selected_LED_ids] # num_frame x L' x m1 x 1
        indices_x = self.indices_x_all[selected_LED_ids] # num_frame x L' x 1 x n1
        
        indices_frame_tensor = indices_frame.reshape((-1, 1, 1, 1))
        
        o_hat = torch.view_as_real(o_hat)
        o_hat_real = torch.unsqueeze(o_hat[:, :, :, 0][indices_frame_tensor, indices_y, indices_x], dim = -1)
        o_hat_imag = torch.unsqueeze(o_hat[:, :, :, 1][indices_frame_tensor, indices_y, indices_x], dim = -1)
        o_hat_cropped = torch.view_as_complex(torch.cat([o_hat_real, o_hat_imag], dim = -1))
        
        P = self.get_pupil(coef = coef)

        g = torch.fft.ifft2(torch.fft.ifftshift(P * o_hat_cropped, dim = (-2, -1)))
        
        g = torch.view_as_real(g)
        
        return (1 / (self.upsampling_ratio ** 4)) * (g ** 2).sum(dim = -1)
    
    def forwardLR(self, o_am, o_ph, selected_LED_ids, indices_frame = np.array([0]), coef = None):
        # compute intensity low-res measurement
        # selected_LED_ids: num_frame x L'
        
        self.o_ph = o_ph
        self.o_am = o_am
        self.o_complex = self.o_am * torch.exp(1j * self.o_ph)

        o_hat = torch.fft.fftshift(torch.fft.fft2(self.o_complex), dim = (-2, -1))
        
        indices_y = self.indices_y_all[selected_LED_ids] # num_frame x L' x m1 x 1
        indices_x = self.indices_x_all[selected_LED_ids] # num_frame x L' x 1 x n1
        
        indices_frame_tensor = indices_frame.reshape((-1, 1, 1, 1))
        
        o_hat = torch.view_as_real(o_hat)
        o_hat_real = torch.unsqueeze(o_hat[:, :, :, 0][indices_frame_tensor, indices_y, indices_x], dim = -1)
        o_hat_imag = torch.unsqueeze(o_hat[:, :, :, 1][indices_frame_tensor, indices_y, indices_x], dim = -1)
        o_hat_cropped = torch.view_as_complex(torch.cat([o_hat_real, o_hat_imag], dim = -1))
        
        P = self.get_pupil(coef = coef)

        g = torch.fft.ifft2(torch.fft.ifftshift(P * o_hat_cropped, dim = (-2, -1)))
        
        g = torch.view_as_real(g)
        
        return (1 / (self.upsampling_ratio ** 4)) * (g).sum(dim = -1) #no magnitude


class Cropping(torch.autograd.Function):
    # ctx is a context object
    @staticmethod
    def forward(ctx, o_hat, indices_frame, indices_y, indices_x):
        ctx.set_materialize_grads(False)
        ctx.high_res_size = o_hat.shape
        ctx.indices_y = indices_y
        ctx.indices_x = indices_x
        return o_hat[indices_frame, indices_y, indices_x]

    @staticmethod
    def backward(ctx, grad_output):
        high_res_size = ctx.high_res_size
        indices_y = ctx.indices_y
        indices_x = ctx.indices_x
        
        grad_input = torch.zeros(*high_res_size, dtype = torch.complex64)
        grad_input = grad_input.to(grad_output.device)

        view_num = indices_y.shape[1]
        
        indices_frame = np.arange(grad_output.shape[0]).reshape((-1, 1, 1))
        for i in range(view_num):
            grad_input[indices_frame, indices_y[:, i, :, :], indices_x[:, i, :, :]] += grad_output[:, i, :, :]
        return grad_input, None, None, None


class AmpDataLoss(nn.Module):
    def __init__(self):
        super(AmpDataLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target, epsilon = 1e-10):
        return self.loss(torch.sqrt(pred + epsilon), torch.sqrt(target + epsilon))


class RegLoss(nn.Module):
    def __init__(self, weight_TV, TVtime=False):
        super(RegLoss, self).__init__()
        if type(weight_TV) is tuple:
            self.weight_amp = weight_TV[0]
            self.weight_phase = weight_TV[1]
        else:
            self.weight_amp = weight_TV
            self.weight_phase = weight_TV
        self.TVtime = TVtime
        if self.TVtime:
            if type(weight_TV) is tuple:
                self.weight_time = weight_TV[2:]
            else:
                self.weight_time = weight_TV

    def TV_complex(self, o_complex, epsilon = 1e-8):
        variation_y = (o_complex[:, 1:, :] - o_complex[:, :-1, :] + epsilon).abs()
        variation_x = (o_complex[:, :, 1:] - o_complex[:, :, :-1] + epsilon).abs()
        lossTV = self.weight_amp * variation_x.mean() + self.weight_phase * variation_y.mean()
        if self.TVtime: #expects framewise consecutive samples
            lossTV += self.weight_time*(o_complex[1:] - o_complex[:-1] + epsilon).abs().mean()
        return lossTV
    
    def TV_separate(self, o_am, o_ph):
        o_am_x = (o_am[:, :, 1:] - o_am[:, :, :-1]).abs()
        o_am_y = (o_am[:, 1:, :] - o_am[:, :-1, :]).abs()
        
        o_ph_x = (o_ph[:, :, 1:] - o_ph[:, :, :-1]).abs()  
        o_ph_y = (o_ph[:, 1:, :] - o_ph[:, :-1, :]).abs()
        
        lossTV = self.weight_amp * (o_am_x.mean() + o_am_y.mean()) + self.weight_phase * (o_ph_x.mean() + o_ph_y.mean())
        
        if self.TVtime: #expects framewise consecutive samples
            lossTV += self.weight_time[0]*((o_am[1:] - o_am[:-1]).abs().mean()) + self.weight_time[1]*((o_ph[1:] - o_ph[:-1]).abs().mean())
        
        return lossTV
    
    def phase_penalty(self, o_ph, mu = 0):
        # for solving phase shift
        return (o_ph.mean() - mu).abs()


class FCBlock(nn.Module):
    def __init__(self, in_size, out_size, nl_layer = nn.ReLU(), use_batchnorm = False):
        super(FCBlock, self).__init__()
        if use_batchnorm:
            module_list = [nn.Linear(in_size, out_size, bias = False), 
                           nn.BatchNorm1d(out_size), nl_layer]
        else:
            module_list = [nn.Linear(in_size, out_size, bias = True), 
                           nl_layer]
        self.fc_block = nn.Sequential(*module_list)
        
    def forward(self, x):
        return self.fc_block(x)


def conv3x3(in_planes, out_planes, nl_layer = nn.ReLU(), use_batchnorm = True, padding_mode = 'reflect'):
    """3x3 convolution with zero padding"""

    if use_batchnorm: 
        conv = nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = 1, 
                            padding = 1, bias = False, padding_mode = padding_mode)
        layers = [conv]
        layers.append(nn.BatchNorm2d(out_planes, affine = True))
    else:
        layers = [nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = 1, 
                            padding = 1, bias = True, padding_mode = padding_mode)]  

    layers.append(nl_layer)
    return nn.Sequential(*layers)


class DecoderBlock(nn.Module):
    
    def __init__(self, in_planes, out_planes, nl_layer = nn.ReLU(), up_scale_factor = 2, 
                 scaled_size = None,
                 interp_mode = 'nearest', block_num = 1, use_batchnorm = True):
        super(DecoderBlock, self).__init__()
        if scaled_size is None:
            up_sampling = nn.Upsample(scale_factor = up_scale_factor, mode = interp_mode)
        else:
            up_sampling = nn.Upsample(size = scaled_size, mode = interp_mode)
        
        layers = [up_sampling, conv3x3(in_planes, out_planes, nl_layer = nl_layer, use_batchnorm = use_batchnorm)]
        
        for i in range(block_num - 1):
            layers.append(conv3x3(out_planes, out_planes, nl_layer = nl_layer, use_batchnorm = use_batchnorm))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)
    

class ImageGenerator(nn.Module):
    def __init__(self, image_size_high_res, constant_width = 128, 
                 interp_mode = 'nearest', nl_layer = nn.ReLU(), use_batchnorm = True, 
                 block_num = 2):
        super(ImageGenerator, self).__init__()

        # hidden_dim = 64 
        self.block_num = block_num # for decoder
        self.constant_width = constant_width
        self.interp_mode = interp_mode
        self.nl_layer = nl_layer
        self.use_batchnorm = use_batchnorm

        (m, n) = image_size_high_res
        up_repeat_num = round(np.log2(min(m, n) / 8))

        # Decoder: from 8x8 to mxn
        scaled_size_before_final = 8 * (2 ** (up_repeat_num - 1))
        self.scale_list = []
        
        decoder_layers = self.get_first_conv_layer_list()
            
        for i in range(up_repeat_num):
            if i < (up_repeat_num - 1):
                up_scale_factor = 2
                scaled_size = None
            else:
                up_scale_factor = (m / scaled_size_before_final, n / scaled_size_before_final)
                scaled_size = (m, n)
            self.scale_list.append(up_scale_factor)
            
            decoder_layers.append(self.get_decoder(up_scale_factor, scaled_size))
             
        self.decoder = nn.Sequential(*decoder_layers)
            
        padding_mode = 'reflect'
        self.final_conv1 = nn.Conv2d(constant_width, 1, kernel_size = 3, stride = 1, 
                                     padding = 1, bias = True, padding_mode = padding_mode)
        self.final_conv2 = nn.Conv2d(constant_width, 1, kernel_size = 3, stride = 1, 
                                     padding = 1, bias = True, padding_mode = padding_mode)        
        
    def get_first_conv_layer_list(self):
        return [conv3x3(1, self.constant_width, nl_layer = self.nl_layer, use_batchnorm = self.use_batchnorm), 
                conv3x3(self.constant_width, self.constant_width, nl_layer = self.nl_layer, use_batchnorm = self.use_batchnorm)]

    def get_decoder(self, up_scale_factor, scaled_size):
        return DecoderBlock(self.constant_width, self.constant_width, 
                            nl_layer = self.nl_layer, up_scale_factor = up_scale_factor, 
                            scaled_size = scaled_size,
                            interp_mode = self.interp_mode, 
                            block_num = self.block_num, use_batchnorm = self.use_batchnorm)

    def forward(self, z):
        z = z[0]
        z = z.view(-1, 1, 8, 8)
        z = self.decoder(z)
        amp = self.final_conv1(z)
        phase = self.final_conv2(z)
        
        return DReLU(amp), np.pi * torch.tanh(phase)

    
def DReLU(x, epsilon = 0.1):
    x_min = torch.clamp(x, min = None, max = epsilon)
    x_max = torch.clamp(x, min = epsilon, max = None)
    return epsilon * torch.exp(x_min / epsilon - 1) + x_max - epsilon