import os
import numpy as np
import cv2
import torch
import torch.utils.data as utils_data
from scipy.io import savemat
import skimage.draw as draw
from skimage.io import imsave
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

def config_parameters(fact=1):
    #fact = 1 #don't change when main_TVtime is runnign
    NA_obj = fact*0.1
    wave_length = 5.32e-7
    upsampling_ratio = int(4/fact)
    z0 = 20
    pixel_size_for_low_res = 1.33e-6/fact #1.845e-6 #5.32e-7/(4*0.1)
    image_size_low_res = (int(64*fact), int(64*fact))
    H = 90.88
    LED_d = 4
    theta = 0
    xint = 0
    yint = 0
    return NA_obj, wave_length, upsampling_ratio, z0, pixel_size_for_low_res, image_size_low_res, H, LED_d, theta, xint, yint


def np_to_var(x, RUN_ON_GPU):
    '''
    Convert numpy array to Torch variable.
    '''
    
    if np.all(np.isreal(x)):
        x = x.astype(np.float32)
        # x = x.astype(np.float64)
    else:
        x = x.astype(np.complex64)
        # x = x.astype(np.complex128)
    
    x = torch.from_numpy(x)
    if RUN_ON_GPU:
        x = x.cuda()
    return x


def var_to_np(x, RUN_ON_GPU):
    '''
    Convert Torch variable to numpy array.
    '''
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()


def move_model(net, RUN_ON_GPU):

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net


def l2_norm_squared(x):
    return np.absolute((x * np.conjugate(x)).sum())


def snr(x_true, x_estimate):
    return 10 * np.log10(l2_norm_squared(x_true) / l2_norm_squared(x_true - x_estimate))


def optimal_scale(x_true, x_estimate):
    return (x_true * np.conjugate(x_estimate)).sum() / l2_norm_squared(x_estimate)


def rsnr(x_true, x_estimate):
    a = optimal_scale(x_true, x_estimate)
    return 10 * np.log10(l2_norm_squared(x_true) / l2_norm_squared(a * x_estimate - x_true))


def rsnr_for_dynamic(x_true, x_estimate):
    num_frames = x_true.shape[0]
    x_estimate_scaled = x_estimate.copy()
    for i in range(num_frames):
        x_estimate_scaled[i] *= optimal_scale(x_true[i], x_estimate[i])
    return snr(x_true, x_estimate_scaled), x_estimate_scaled


def load_image(image_dir, image_size, value_range = (0, 1)):
    img = cv2.imread(image_dir, 0)
    img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_CUBIC).astype(float)
    if value_range is not None:
        img = cv2.normalize(img, None, value_range[0], value_range[1], cv2.NORM_MINMAX)    
    return img


def load_dynamic_dataset(dataset_dir, image_size, num_max_frames = 10, iter_start = 10):
    file_list = os.listdir(dataset_dir)
    file_list = np.sort(file_list)
    frames = []
    count = 0
    for i in range(len(file_list)):
        image_dir = dataset_dir + '/' + file_list[i + iter_start]
        img = cv2.imread(image_dir, 0)
        if img is not None:
            img = cv2.resize(img, (image_size[1], image_size[0]), interpolation=cv2.INTER_CUBIC).astype(float)
            frames.append(img)
            count += 1
        if count == num_max_frames:
            break
    return np.stack(frames)


def normalize_img(img, value_range = (0, 1)):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min) * (value_range[1] - value_range[0]) + value_range[0]


def unnormalize_img(img, value_range = (0, 1)):
    # to range (0, 1)
    img_normal = (img - value_range[0]) / (value_range[1] - value_range[0])
    return np.maximum(np.minimum(img_normal, 1), 0)


def save_synthetic_data(file_name, im_low_res_seq, wave_length, z0 = 0, theta = 0, xint = 0, yint = 0, aberration = 0):
    out_dict = {'aberration': float(aberration), 
                'imlow_HDR': np.moveaxis(im_low_res_seq, 0, 2), 
                'theta': float(theta), 'wlength': wave_length, 
                'z': float(z0), 
                'xint': float(xint), 
                'yint': float(yint)}
    savemat('synthetic_seq.mat', out_dict)    
    

def plot_object_and_pupils_in_Fourier(o, shifted_center_x, shifted_center_y, 
                                 freq_cutoff_x_in_pixel_unit, freq_cutoff_y_in_pixel_unit,
                                 image_size_high_res,
                                 check_id = 0, show_all_pupil = 0, file_name = None, dirres = './results'):
    
    o_hat = np.fft.fftshift(np.fft.fft2(o), axes = (-2, -1))
    
    o_hat_abs = np.absolute(o_hat)
    o_hat_abs_log = np.log10(o_hat_abs)
    
    max_value = o_hat_abs_log.max()
    
    if show_all_pupil:
        plot_id_list = np.arange(shifted_center_x.shape[0])
        # title = 'All covered regions in Fourier domain'
    else:
        plot_id_list = [check_id]
        # title = 'Covered region in Fourier domain'

    for l in plot_id_list:
        ellipse_y, ellipse_x = draw.ellipse_perimeter(shifted_center_y[l], shifted_center_x[l], round(freq_cutoff_y_in_pixel_unit), round(freq_cutoff_x_in_pixel_unit), shape = image_size_high_res)
        o_hat_abs_log[ellipse_y, ellipse_x] = max_value
    plt.figure()
    plt.imshow(o_hat_abs_log, cmap='gray', vmin = 0, vmax = max_value)
    plt.axis('off')
    if file_name is None:
        plt.savefig('{}/pupil_cover.png'.format(dirres)) 
    else:
        plt.savefig(file_name) 
    plt.close()
    #plt.show()


def plot_spiral_array(xi, yi, LED_d, xint = 0, yint = 0, dirres = './results'):
    # plt.plot(xi, yi, '*r')
    X = np.stack([xi, yi]) * LED_d + np.array([[xint], [yint]])
    plt.figure()
    plt.plot(X[0, :], X[1, :], '*r')
    
    num = xi.shape[0]
    
    for i in range(num - 1):
        x = X[0, i]
        y = X[1, i]
        dx = X[0, i + 1] - x
        dy = X[1, i + 1] - y
        plt.arrow(x, y, dx, dy, width = 0.05)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('{}/spiral_array.png'.format(dirres)) 
    plt.show()


def add_noise(x, noise_std = 0.01):
    if noise_std == 0:
        return x
    
    if len(x.shape) == 3:
        out = x.copy()
        for i in range(x.shape[0]):
            v = np.random.randn(x.shape[1], x.shape[2])
            v /= np.sqrt(np.sum(v ** 2, axis = 1)).reshape((-1, 1))
            out[i] += noise_std * np.random.randn(x.shape[1], 1) * v
        return out
    else:
        return x + noise_std * np.random.randn(*x.shape)


def plot_activation(act, v_max = 3, step = 0.01):
    x = torch.arange(-v_max, v_max, step)
    y = act(x)
    plt.plot(x, y)
    plt.show()


def manifold_embedding(num_frames, noise_std, dim = 2, 
                       manifold_name = 'line_order', min_safety_distance = 0.1, 
                       im_low_res_seq = None, o_high_res0 = None):
    # manifold design: Fourier encoding, low resolution
    
    safety_ratio = 8
    safety_distance = safety_ratio * noise_std
    safety_distance = max(safety_distance, min_safety_distance)
    
    if manifold_name == 'random':
        z_fixed = np.random.uniform(low = 0, high = 0.1, size = (num_frames, dim))
        # z_fixed = np.random.randn(num_frames, dim)
    elif manifold_name == 'line_order':
        step_left = np.floor(num_frames / 2)
        z0 = np.random.randn(1, dim)
        z_fixed = np.arange(-step_left, -step_left + num_frames).reshape((-1, 1)).dot(z0)
        z_fixed = z_fixed / np.linalg.norm(z0) * safety_distance
        z_fixed += np.random.uniform(low = 0, high = 0.1, size = (1, dim))
    else:
        if im_low_res_seq is not None and manifold_name == "low_res":
            im_size_low = round(np.sqrt(dim))
            z_fixed = []
            for i in range(num_frames):
                im_low = cv2.resize(im_low_res_seq[i][0], (im_size_low, im_size_low), 
                                    interpolation = cv2.INTER_CUBIC).astype(float)
                z_fixed.append(im_low.flatten())
            z_fixed = np.stack(z_fixed)
        elif o_high_res0 is not None and manifold_name == "low_res_double":
            im_size_low = round(np.sqrt(dim))
            z_fixed1 = []
            z_fixed2 = []
            amp = np.absolute(o_high_res0)
            
            phase = np.angle(o_high_res0)
            for i in range(num_frames):
                im_low1 = cv2.resize(amp[i], (im_size_low, im_size_low), 
                                    interpolation = cv2.INTER_CUBIC).astype(float)
                z_fixed1.append(im_low1.flatten())
                im_low2 = cv2.resize(phase[i], (im_size_low, im_size_low), 
                                    interpolation = cv2.INTER_CUBIC).astype(float)
                z_fixed2.append(im_low2.flatten())
            z_fixed1 = np.stack(z_fixed1)
            z_fixed2 = np.stack(z_fixed2)
            z_fixed1 = (z_fixed1 - z_fixed1.min()) / (z_fixed1.max() - z_fixed1.min()) * 0.1
            z_fixed2 = (z_fixed2 - z_fixed2.min()) / (z_fixed2.max() - z_fixed2.min()) * 0.1
            z_fixed = np.stack([z_fixed1, z_fixed2])
        elif im_low_res_seq is not None and manifold_name == "mds":
            X = []
            for i in range(num_frames):
                X.append(im_low_res_seq[i][0].flatten())
            X = np.stack(X)
            embedding = MDS(n_components = dim, max_iter = 1000)
            X_transformed = embedding.fit_transform(X)
            
            dist_near_by = np.sqrt(np.sum((X_transformed[1:] - X_transformed[:-1]) ** 2, axis = 1))
            z_fixed = X_transformed / dist_near_by.mean() * min_safety_distance
        elif o_high_res0 is not None and manifold_name == "mds_new":
            X = []
            for i in range(num_frames):
                X.append(o_high_res0[i].flatten())
            X = np.stack(X)
            embedding = MDS(n_components = dim, dissimilarity = "precomputed", max_iter = 1000)
            X_transformed = embedding.fit_transform(compute_dist_matrix(X))
            
            dist_near_by = np.sqrt(np.sum((X_transformed[1:] - X_transformed[:-1]) ** 2, axis = 1))
            z_fixed = X_transformed / dist_near_by.mean() * min_safety_distance
        elif o_high_res0 is not None and manifold_name == "mds_double":
            X = []
            for i in range(num_frames):
                X.append(o_high_res0[i].flatten())
            X = np.stack(X)
            embedding = MDS(n_components = dim, max_iter = 1000)
            X_transformed1 = embedding.fit_transform(np.absolute(X))
            X_transformed2 = embedding.fit_transform(np.angle(X))
            dist_near_by1 = np.sqrt(np.sum((X_transformed1[1:] - X_transformed1[:-1]) ** 2, axis = 1))
            z_fixed1 = X_transformed1 / dist_near_by1.mean() * min_safety_distance
            dist_near_by2 = np.sqrt(np.sum((X_transformed2[1:] - X_transformed2[:-1]) ** 2, axis = 1))
            z_fixed2 = X_transformed2 / dist_near_by2.mean() * min_safety_distance
            z_fixed = np.stack([z_fixed1, z_fixed2])
        elif manifold_name == "gaussian":
            step = 0.8
            sigma = 2
            z_fixed = []
            t_list = np.arange(-num_frames // 2, num_frames - num_frames // 2)
            for t in t_list:
                tem = gaussian_kernel(size = round(np.sqrt(dim)), mu = (t * step, t * step), sigma = sigma)
                z_fixed.append(tem.flatten())
            z_fixed = np.stack(z_fixed)
        else:
            # line, with end points not fixed
            z_start = np.random.randn(dim)
            z_end = np.random.randn(dim)
            
            t = np.linspace(0, 1, num = num_frames)
            
            z_fixed = []
            for i in range(num_frames):
                z_fixed.append((1 - t[i]) * z_start + t[i] * z_end)
            z_fixed = np.stack(z_fixed)
    return z_fixed


def compute_dist_matrix(X):
    n = X.shape[0]
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = np.sqrt(l2_norm_squared(optimal_scale(X[j], X[i]) * X[i] - X[j]))
    return (dist_mat + dist_mat.T) / 2
            

def gaussian_kernel(size = 8, mu = (0, 0), sigma = 1):
    x = np.arange(-size // 2, -size // 2 + size).reshape((1, -1))
    exp_x = np.exp(-((x - mu[0]) / sigma) ** 2 / 2)
    exp_y = np.exp(-((x - mu[1]) / sigma) ** 2 / 2).T
    kernel = exp_y.dot(exp_x)
    # return kernel / kernel.sum()
    return kernel


def plot_loss_track(collection_of_loss_track, name_list, xlabel = 'Iteration number', ylabel = 'Loss', dirres = './results'):

    name_list_new = [name for name in name_list if name != 'GS']
    
    fig, ax = plt.subplots()
    
    for i, name in enumerate(name_list_new):
        loss_track = collection_of_loss_track[i]
        ax.plot(np.arange(1, len(loss_track) + 1), np.array(loss_track), label = name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    leg = ax.legend()
    plt.savefig('{}/loss_track.png'.format(dirres))


def plot_coef_error_track(coef_collection, coef_GT, name_list, check_period = 10, dirres = './results'):

    name_list_new = [name for name in name_list if name != 'GS']
    
    fig, ax = plt.subplots()
    
    for i, name in enumerate(name_list_new):
        num = len(coef_collection[i])
        err_coef = np.abs(coef_collection[i] - coef_GT).sum(axis = 1)
        ax.plot(np.arange(num) * check_period, err_coef, label = name)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SAE')
    leg = ax.legend()
    plt.savefig('{}/coef_track.png'.format(dirres))
    

def plot_coef(coef_collection, coef_GT, name_list, dirres = './results'):
    indices = np.arange(1, coef_GT.shape[-1] + 1)
    
    name_list_new = [name for name in name_list if name != 'GS']
    fig, ax = plt.subplots()
    for i, name in enumerate(name_list_new):
        ax.plot(indices, coef_collection[i][-1], label = name)
        # print('z0 inferred by %s is: %.7f'%(name, coef_collection[i][-1][-1]))
    ax.plot(indices, coef_GT, label = 'GT')
    # print('GT z0 is: %.7f'%(coef_GT[-1]))
    ax.set_xlabel('Mode Index')
    ax.set_ylabel('Value')    
    leg = ax.legend()
    plt.savefig('{}/coef_result.png'.format(dirres))
    

def plot_rsnr(rsnr_collection, name_list, use_rsnr = True, check_period = 10, dirres = './results'):
    fig, ax = plt.subplots()
    
    for i, name in enumerate(name_list):
        num = len(rsnr_collection[i])
        ax.plot(np.concatenate((np.array([0]), np.arange(num - 1) * check_period)), np.array(rsnr_collection[i]), label = name)
    
    ax.set_xlabel('Epoch')
    if use_rsnr:
        ax.set_ylabel('RSNR')
    else:
        ax.set_ylabel('SNR')
    leg = ax.legend()    
    plt.savefig('{}/rsnr_plot.png'.format(dirres))
    

def compute_separate_snr(o_high_res_scaled, o_ground_truth_frames):
    out = []
    for i in range(o_ground_truth_frames.shape[0]):
        out.append(snr(o_ground_truth_frames[i], o_high_res_scaled[i]))
    return out


def compute_error_map(o_pred, o_GT, name = 'GS', high_am = 0.3, high_ph = 1.0, dirres = './results'):
    if len(o_pred.shape) == 2:
        err_map_amp = np.absolute(np.absolute(o_pred) - np.absolute(o_GT))
        err_map_normal_amp = unnormalize_img(err_map_amp, (0, high_am))
        imsave('{}/amp_error_map_'.format(dirres) + name + '.png', (255 * err_map_normal_amp).astype(np.uint8))
        
        err_map_ph = np.absolute(np.angle(o_pred) - np.angle(o_GT))
        err_map_normal_ph = unnormalize_img(err_map_ph, (0, high_ph))
        imsave('{}/phase_error_map_'.format(dirres) + name + '.png', (255 * err_map_normal_ph).astype(np.uint8))
    else:
        new_dir = '{}/error_maps_'.format(dirres) + name
        os.mkdir(new_dir)
        
        os.mkdir(new_dir + '/amplitude')
        os.mkdir(new_dir + '/phase')
        
        num_frames = o_pred.shape[0]
        err_map_amp = np.real(o_pred).copy()
        err_map_ph = err_map_amp.copy()
        for i in range(num_frames):
            err_map_amp[i] = np.absolute(np.absolute(o_pred[i]) - np.absolute(o_GT[i]))
            err_map_normal_amp = unnormalize_img(err_map_amp[i], (0, high_am))
            imsave(new_dir + '/amplitude/error_map_' + str(i) + '.png', (255 * err_map_normal_amp).astype(np.uint8))
            
            err_map_ph[i] = np.absolute(np.angle(o_pred[i]) - np.angle(o_GT[i]))
            err_map_normal_ph = unnormalize_img(err_map_ph[i], (0, high_ph))
            imsave(new_dir + '/phase/error_map_' + str(i) + '.png', (255 * err_map_normal_ph).astype(np.uint8))
    return err_map_amp, err_map_ph

'''
def compare_pupils(FPM, P, collection_of_coef_list, name_list_new, RUN_ON_GPU, dirres = './results'):
    name_list_new = [name for name in name_list_new if name != 'GS']
    pred_list = []
    for i, name in enumerate(name_list_new):
        coef = collection_of_coef_list[i][-1]
        P_pred = var_to_np(FPM.get_pupil(np_to_var(coef, RUN_ON_GPU)), RUN_ON_GPU)
        
        P_amp = np.absolute(P)
        P_ph = np.angle(P)
        
        plt.figure(), plt.imshow(np.absolute(P_pred), cmap='gray', vmin=P_amp.min(), vmax=P_amp.max()), plt.title('Amplitude of pupil')
        plt.axis('off')
        plt.colorbar()
        plt.savefig('{}/pred_'.format(dirres) + name + "_pupil_amplitude.png") 
        plt.show()
        
        plt.figure(), plt.imshow(np.angle(P_pred), cmap='gray', vmin=P_ph.min(), vmax=P_ph.max()), plt.title('Phase of pupil')
        plt.axis('off')
        plt.colorbar()
        plt.savefig('{}/pred_'.format(dirres) + name + "_pupil_phase.png") 
        plt.show()
        
        print("RSNR of %s pupil prediction is %.7f"%(name, rsnr(P, P_pred)))
        pred_list.append(P_pred)
    return pred_list
'''
    
def compare_pupils(FP, P, coef, name, RUN_ON_GPU, dirres = './results'):
    
    P_pred = var_to_np(FP.get_pupil(np_to_var(coef[-1], RUN_ON_GPU)), RUN_ON_GPU)
    P_amp = np.absolute(P)
    P_ph = np.angle(P)
        
    plt.figure(), plt.imshow(np.absolute(P_pred), cmap='gray', vmin=P_amp.min(), vmax=P_amp.max()), plt.title('Amplitude of pupil')
    plt.axis('off')
    plt.colorbar()
    plt.savefig('{}/pred_'.format(dirres) + name + "_pupil_amplitude.png") 
    plt.show()
        
    plt.figure(), plt.imshow(np.angle(P_pred), cmap='gray', vmin=P_ph.min(), vmax=P_ph.max()), plt.title('Phase of pupil')
    plt.axis('off')
    plt.colorbar()
    plt.savefig('{}/pred_'.format(dirres) + name + "_pupil_phase.png") 
    plt.show()
    
    snr_pupil = snr(P, P_pred)
    print("SNR of %s pupil prediction is %.7f"%(name, snr_pupil))

    rsnr_pupil = rsnr(P, P_pred)
    print("RSNR of %s pupil prediction is %.7f"%(name, rsnr_pupil))

    return P_pred, snr_pupil, rsnr_pupil