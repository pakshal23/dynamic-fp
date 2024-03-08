"""
Adapted from the Matlab codes of
"Implementations and applications of Fourier ptychography, Zheng 2021"
"""

import numpy as np
from zernike import RZern


def spiral_LED(size):
    steps = np.append(np.repeat(np.arange(1, size), 2), size - 1)
    
    num = size ** 2

    go_x = 1 # go x direction
    go_x_positive = 1
    go_y_positive = 1
    
    x_location = np.zeros((num, ))
    y_location = np.zeros((num, ))
    
    i = 1
    for step in steps:
        i_next = i + step
        if go_x:
            y_location[i:i_next] = y_location[i - 1]
            if go_x_positive:
                x_location[i:i_next] = np.arange(x_location[i - 1] + 1, x_location[i - 1] + step + 1)
            else:
                x_location[i:i_next] = np.arange(x_location[i - 1] - 1, x_location[i - 1] - step - 1, -1)
            go_x_positive = 1 - go_x_positive
        else:
            x_location[i:i_next] = x_location[i - 1]
            if go_y_positive:
                y_location[i:i_next] = np.arange(y_location[i - 1] + 1, y_location[i - 1] + step + 1)
            else:
                y_location[i:i_next] = np.arange(y_location[i - 1] - 1, y_location[i - 1] - step - 1, -1)
            go_y_positive = 1 - go_y_positive
        go_x = 1 - go_x
        i = i_next
    return x_location, y_location

def line_LED(size):
    num = size ** 2
    
    c = np.floor((size - 1) / 2)
    y = -c
    
    go_right = 1
    
    x_location = np.zeros((num, ))
    y_location = np.zeros((num, ))
    
    for i in range(size):
        if go_right:
            x_location[(i * size):((i + 1) * size)] = np.arange(-c, size - c)
        else:
            x_location[(i * size):((i + 1) * size)] = np.arange(size - c - 1, -c - 1, -1)
        y_location[(i * size):((i + 1) * size)] = y
        
        go_right = 1 - go_right
        y += 1
    return x_location, y_location

def compute_k_vector_normalized(xi, yi, H, LED_d, theta = 0, xint = 0, yint = 0):

    X0 = np.stack([xi, yi]) * LED_d + np.array([[xint], [yint]])
    
    theta = theta * np.pi / 180
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    X = R.dot(X0)
    
    theta_plane = np.arctan2(X[1, :], X[0, :])
    
    xoff = 0 # glass substrate ignored
        
    l = np.sqrt((X ** 2).sum(axis = 0))
    
    NA_illum = np.abs((l - xoff) / np.sqrt((l - xoff) ** 2 + H ** 2))
    kx_normal = -NA_illum * np.cos(theta_plane)
    ky_normal = -NA_illum * np.sin(theta_plane)
    
    return kx_normal, ky_normal, NA_illum
    
def compute_pupil_and_shift(kx_normal, ky_normal, NA_obj, wave_length, pixel_size_for_low_res, 
                            image_size_low_res = (201, 201), upsampling_ratio = 4, z0 = 0, zernike_coef = None):
    # image_size_low_res is a tuple
    
    pixel_size_for_high_res = pixel_size_for_low_res / upsampling_ratio
    
    m1, n1 = image_size_low_res
    m = upsampling_ratio * m1
    n = upsampling_ratio * n1
    
    k0 = 2 * np.pi / wave_length
        
    omega_cutoff = 2 * np.pi * (NA_obj / wave_length)
    
    T_x = n * pixel_size_for_high_res
    T_y = m * pixel_size_for_high_res
    
    omega0_x = 2 * np.pi / T_x
    omega0_y = 2 * np.pi / T_y    
    
    freq_cutoff_x_in_pixel_unit = omega_cutoff / omega0_x
    freq_cutoff_y_in_pixel_unit = omega_cutoff / omega0_y
    
    center_x = int(n / 2)
    center_y = int(m / 2)
    
    omega_x_all = np.arange(-center_x, -center_x + n) * omega0_x
    omega_y_all = np.arange(-center_y, -center_y + m) * omega0_y
    
    
    omega_x_all = omega_x_all.reshape((1, -1))
    omega_y_all = omega_y_all.reshape((-1, 1))
    
    center_x_low = int(n1 / 2)
    center_y_low = int(m1 / 2)

    omega_z_mat = np.sqrt(k0 ** 2 - omega_x_all ** 2 - omega_y_all ** 2)
    omega_z = omega_z_mat[(center_y - center_y_low):(center_y - center_y_low + m1), 
                          (center_x - center_x_low):(center_x - center_x_low + n1)].astype(np.float32)

    if z0 is None or z0 == 0:
        H_defocus = 1
    else:
        H_defocus = np.exp(1j * z0 * omega_z)
        
    i_array = np.arange(n1).reshape((1, -1))
    j_array = np.arange(m1).reshape((-1, 1))
    dist = ((i_array - center_x_low) / freq_cutoff_x_in_pixel_unit) ** 2 + ((j_array - center_y_low) / freq_cutoff_y_in_pixel_unit) ** 2
    
    P = (dist <= 1).astype(np.float32)
    
    if z0 is not None and z0 != 0:
        H_defocus[dist > 1] = 0
        
    if zernike_coef is not None:
        P = get_pupil_from_zernike(m1, n1, freq_cutoff_x_in_pixel_unit, freq_cutoff_y_in_pixel_unit, zernike_coef)

    P = P * H_defocus
    
    kx = k0 * kx_normal
    ky = k0 * ky_normal
    
    kx_in_pixel_unit = kx / omega0_x
    ky_in_pixel_unit = ky / omega0_y
    
    shifted_center_x = np.round(center_x - kx_in_pixel_unit).astype(np.int32)
    shifted_center_y = np.round(center_y - ky_in_pixel_unit).astype(np.int32)
    
    return P, shifted_center_x, shifted_center_y, freq_cutoff_x_in_pixel_unit, freq_cutoff_y_in_pixel_unit, omega_z

def get_zernike_modes(m1, n1, freq_cutoff_x_in_pixel_unit, freq_cutoff_y_in_pixel_unit, num_coef):
    freq_cutoff_x_in_pixel_unit_int = int(freq_cutoff_x_in_pixel_unit)
    freq_cutoff_y_in_pixel_unit_int = int(freq_cutoff_y_in_pixel_unit)    
    rx = freq_cutoff_x_in_pixel_unit_int / freq_cutoff_x_in_pixel_unit
    ry = freq_cutoff_y_in_pixel_unit_int / freq_cutoff_y_in_pixel_unit
    
    center_x_low = int(n1 / 2)
    center_y_low = int(m1 / 2)
    
    out = np.ones((num_coef, m1, n1)).astype(np.complex64) * np.nan
    
    cart = RZern(int(np.ceil((0.25 + 2 * (num_coef + 1)) ** 0.5 - 1.5)))
    ddx = np.linspace(-rx, rx, 2 * freq_cutoff_x_in_pixel_unit_int + 1)
    ddy = np.linspace(-ry, ry, 2 * freq_cutoff_y_in_pixel_unit_int + 1)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)
    
    c = np.zeros(cart.nk)
    for i in range(1, num_coef + 1):
        c *= 0.0
        c[i] = 1.0
        mode = cart.eval_grid(c, matrix = True).astype(np.complex64)
        out[i - 1][(center_y_low - freq_cutoff_y_in_pixel_unit_int):(center_y_low + freq_cutoff_y_in_pixel_unit_int + 1), 
               (center_x_low - freq_cutoff_x_in_pixel_unit_int):(center_x_low + freq_cutoff_x_in_pixel_unit_int + 1)] = mode.copy()
    return out

def get_pupil_from_zernike(m1, n1, freq_cutoff_x_in_pixel_unit, freq_cutoff_y_in_pixel_unit, coef):
    num_coef = coef.shape[0]
    modes = get_zernike_modes(m1, n1, freq_cutoff_x_in_pixel_unit, freq_cutoff_y_in_pixel_unit, num_coef)
    pupil = np.zeros((m1, n1)).astype(np.complex64)
    for i in range(num_coef):
        pupil += coef[i] * modes[i]
    mask = np.isnan(pupil)
    pupil = np.exp(1j * pupil)
    pupil[mask] = 0.0
    return pupil
    
def get_cropping_indices(m1, n1, shifted_center_x, shifted_center_y):
    # cropping the object in Fourier space, according to the given shift
    
    center_x_low = int(n1 / 2)
    center_y_low = int(m1 / 2)
    
    num = shifted_center_x.shape[0]
    
    if num != shifted_center_y.shape[0]:
        raise ValueError
    
    indices_y = np.zeros((num, m1, 1), dtype = np.int32)
    indices_x = np.zeros((num, 1, n1), dtype = np.int32)
    
    for i in range(num):
        xc = shifted_center_x[i]
        yc = shifted_center_y[i]
        
        indices_x[i] = np.arange(xc - center_x_low, xc - center_x_low + n1).reshape((1, -1))
        indices_y[i] = np.arange(yc - center_y_low, yc - center_y_low + m1).reshape((-1, 1))
    return indices_x, indices_y

def compute_overlap_ratio(P, shifted_center_x, shifted_center_y):
    m1, n1 = P.shape
    
    mask = (np.absolute(P) > 0).astype(np.float32)
    pupil_area_unit = mask.sum()
    
    num = shifted_center_x.shape[0]
    ratio_mat = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if i != j:
                xc1 = shifted_center_x[i]
                yc1 = shifted_center_y[i]
                
                xc2 = shifted_center_x[j]
                yc2 = shifted_center_y[j]
                
                x_diff = abs(xc2 - xc1)
                y_diff = abs(yc2 - yc1)
                
                test_im = np.zeros((m1 + y_diff, n1 + x_diff))
                test_im[:m1, :n1] += mask
                test_im[y_diff:(m1 + y_diff), x_diff:(n1 + x_diff)] += mask
        
                overlapped_area_unit = (test_im == 2).sum()
                ratio_mat[i, j] = overlapped_area_unit / pupil_area_unit
    return ratio_mat