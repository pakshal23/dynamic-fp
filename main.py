import os
import shutil
import argparse
import json
import time
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import skimage.metrics as metrics
from skimage.io import imsave

import fp_utils
import utils
import solvers
import models

# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2021
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(1)


def save_im_seq(im_seq_real, im_range, new_dir, name):
    num_frames = im_seq_real.shape[0]
    for i in range(num_frames):
        im = im_seq_real[i]
        im = utils.unnormalize_img(im, im_range)
        im = (255 * im).astype(np.uint8)
        imsave(new_dir + '/' + name + '_' + str(i) + '.png', im)


def make_dir_and_save_ph_am(am_seq, ph_seq, name, amp_range, phase_range, dirres = './results'):
    new_dir = dirres + '/' + name + '_output'

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
        os.mkdir(new_dir + '/amplitude')
        os.mkdir(new_dir + '/phase')
    
    save_im_seq(am_seq, amp_range, new_dir + '/amplitude', name)
    save_im_seq(ph_seq, phase_range, new_dir + '/phase', name)
    

def compute_psnr_and_save_results(am_frames_GT, phase_frames_GT, o_high_res, name, 
                                  amp_range, phase_range, dirres = './results'):
    np.save("{}/recovered_im_{}.npy".format(dirres,name), o_high_res)
    
    am_estimate = np.absolute(o_high_res)
    ph_estimate = np.angle(o_high_res)
    
    psnr_am = metrics.peak_signal_noise_ratio(am_frames_GT, am_estimate, data_range = amp_range[1] - amp_range[0])
    psnr_ph = metrics.peak_signal_noise_ratio(phase_frames_GT, ph_estimate, data_range = phase_range[1] - phase_range[0])    
    print(name + ' method achieved PSNR (amplitude and phase): %.2f and %.2f'%(psnr_am, psnr_ph))
        
    make_dir_and_save_ph_am(am_estimate, ph_estimate, name, amp_range, phase_range, dirres = dirres)


def get_argument():
    parser = argparse.ArgumentParser(description='Input arguments to FPM')
    parser.add_argument('--res_dir', metavar = 'N', type = str, default = '/home/bohra/dynamic-fp/results', help = 'Directory to save results')
    parser.add_argument('--run_name', metavar = 'N', type = str, default = 'full_run_1', help = 'Name of experiment')
    parser.add_argument('--dataset_dir', metavar = 'N', type = str, default = '/home/bohra/dynamic-fp/data', 
                        help = 'File directory for amplitude')
    parser.add_argument('--recover_pupil', metavar = 'N', type = int, default = 0, 
                        help = 'Flag to recover pupil')
    parser.add_argument('--run_DC', metavar = 'N', type = int, default = 1, 
                        help = 'Flag to test DC')
    parser.add_argument('--run_STV', metavar = 'N', type = int, default = 1, 
                        help = 'Flag to test STV')
    parser.add_argument('--run_STTV', metavar = 'N', type = int, default = 1, 
                        help = 'Flag to test STTV')
    parser.add_argument('--run_DSTP', metavar = 'N', type = int, default = 1, 
                        help = 'Flag to test DSTP')
    
    parser.add_argument('--err_tol', metavar = 'N', type = float, default = 1e-10, help = 'stopping criterion (error tolerance) for all algorithms')
    parser.add_argument('--use_prev_as_initial_guess', metavar = 'N', type = int, default = 1, 
                        help = 'Flag to use previous result as initial guess in GS and DC/STV methods')
    parser.add_argument('--check_period', metavar = 'N', type = int, default = 1000, 
                        help = 'Number of epochs for checking the result')
    parser.add_argument('--batch_size_for_frames', metavar = 'N', type = int, default = 10, 
                        help = 'Number of frames used during each iteration in the optimization')
    parser.add_argument('--batch_size_for_led', metavar = 'N', type = int, default = 1, 
                        help = 'Number of views (per frame) used during each iteration in the optimization')
    parser.add_argument('--do_bestRSNR', metavar = 'N', type = int, default = 0, 
                        help = 'Early stop')

    parser.add_argument('--lr', metavar = 'N', type = float, default = 1e-3, 
                        help = 'Learning rate for the DC, STV and STTV methods')
    parser.add_argument('--num_epoch', metavar = 'N', type = int, default = 10, 
                        help = 'Number of epochs for the DC, STV and STTV methods')
    parser.add_argument('--use_activation', metavar = 'N', type = int, default = 1, 
                        help = 'Flag to use activation function in the output of DC, STV and STTV methods') 

    parser.add_argument('--STV_weight_reg_spatial_amp', metavar = 'N', type = float, default = 1e-3, 
                        help = 'Weight for spatial TV penalty in the STV method (amplitude)') 
    parser.add_argument('--STV_weight_reg_spatial_phase', metavar = 'N', type = float, default = 1e-3, 
                        help = 'Weight for spatial TV penalty in the STV method (phase)')
    
    parser.add_argument('--STTV_weight_reg_spatial_amp', metavar = 'N', type = float, default = 1e-3, 
                        help = 'Weight for spatial TV penalty in the STTV method (amplitude)') 
    parser.add_argument('--STTV_weight_reg_spatial_phase', metavar = 'N', type = float, default = 1e-3, 
                        help = 'Weight for spatial TV penalty in the STTV method (phase)')
    parser.add_argument('--STTV_weight_reg_temporal_amp', metavar = 'N', type = float, default = 1e-3, 
                        help = 'Weight for temporal TV penalty in the STTV method (amplitude)') 
    parser.add_argument('--STTV_weight_reg_temporal_phase', metavar = 'N', type = float, default = 1e-3, 
                        help = 'Weight for temporal TV penalty in the STTV method (phase)') 
    
    parser.add_argument('--lr_DSTP', metavar = 'N', type = float, default = 5e-5, 
                        help = 'Learning rate for the DSTP method')
    parser.add_argument('--num_epoch_DSTP', metavar = 'N', type = int, default = 10, 
                        help = 'Number of epochs for the DSTP method')
    parser.add_argument('--use_batchnorm', metavar = 'N', type = int, default = 1, 
                        help = 'Flag to use BN normalization') 
    parser.add_argument('--initialize_net', metavar = 'N', type = int, default = 1, 
                        help = 'Flag to initialize the network parameters via fitting GS result')
    parser.add_argument('--freeze_bn', metavar = 'N', type = int, default = 1, 
                        help = 'Freeze BN after initialization')
    
    parser.add_argument('--cuda', metavar = 'N', type = int, default = 0, 
                        help = 'Cuda')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_argument()

    torch.cuda.set_device(args.cuda)
    amp_range = (0.5, 1.0)
    phase_range = (-np.pi / 2, np.pi / 2)

    dirres = args.res_dir + '/' + args.run_name + '/'
    if not os.path.exists(dirres):
        os.makedirs(dirres)

    with open('{}/config.txt'.format(dirres), 'w') as f:
       f.write(''.join(["%s = %s \n" % (k,v) for k,v in args.__dict__.items()]))


    ## Setup + load data
    NA_obj, wave_length, upsampling_ratio, z0, pixel_size_for_low_res, image_size_low_res, H, LED_d, theta, xint, yint = utils.config_parameters(fact=1.0)

    m1, n1 = image_size_low_res
    pixel_size_for_high_res = pixel_size_for_low_res / upsampling_ratio
    image_size_high_res = (upsampling_ratio * image_size_low_res[0], upsampling_ratio * image_size_low_res[1])

    array_size = 10
    xi, yi = fp_utils.spiral_LED(array_size)
    
    num_coef = 9
    zernike_coef = np.zeros(num_coef)
    z0 = 0
    zernike_coef[1 - 1] = 0. # Tilt Y, Z 1,-1
    zernike_coef[2 - 1] = 0.15 # Tilt X, Z 1,1
    zernike_coef[3 - 1] = 0.3 # defocus, Z 2,0 
    zernike_coef[4 - 1] = -0.1 # oblique, Z 2,-2
    zernike_coef[5 - 1] = 0.2 # vertical, Z 2,2

    kx_normal, ky_normal, NA_illum = fp_utils.compute_k_vector_normalized(xi, yi, H, LED_d, theta = theta, xint = xint, yint = yint)

    P0, shifted_center_x, shifted_center_y, freq_cutoff_x_in_pixel_unit, freq_cutoff_y_in_pixel_unit, omega_z = fp_utils.compute_pupil_and_shift(kx_normal, ky_normal, NA_obj, wave_length, pixel_size_for_low_res, 
                                                             image_size_low_res = image_size_low_res, upsampling_ratio = upsampling_ratio, 
                                                             z0 = 0, zernike_coef = None)
    modes = fp_utils.get_zernike_modes(m1, n1, freq_cutoff_x_in_pixel_unit, freq_cutoff_y_in_pixel_unit, num_coef)
    non_zero_mask = ~np.isnan(modes[0])
    omega_z[np.isnan(modes[0])] = 0
    pupil_parametric_model = dict()
    modes[np.isnan(modes)] = 0
    pupil_parametric_model['modes'] = modes
    pupil_parametric_model['non_zero_mask'] = non_zero_mask
    pupil_parametric_model['omega_z'] = omega_z

    FP = models.FP_model(P0, shifted_center_x, shifted_center_y, m1, n1, 
                            upsampling_ratio = upsampling_ratio, 
                            pupil_parametric_model = pupil_parametric_model, RUN_ON_GPU = RUN_ON_GPU)

    coef_GT = utils.np_to_var(zernike_coef, RUN_ON_GPU)
    P = utils.var_to_np(FP.get_pupil(coef_GT), RUN_ON_GPU)
    coef_GT = utils.var_to_np(coef_GT, RUN_ON_GPU)
    FP.P = utils.np_to_var(P, RUN_ON_GPU = RUN_ON_GPU)

    num_views_per_frame = 1
    am_frames = np.load(args.dataset_dir + '/gt_amp_frames.npy')
    phase_frames = np.load(args.dataset_dir + '/gt_phase_frames.npy')
    list_of_selected_LED_ids = np.load(args.dataset_dir + '/list_of_selected_LED_ids.npy')
    im_low_res_seq = np.load(args.dataset_dir + '/im_low_res_seq_noiseless.npy')
    #im_low_res_seq = np.load(args.dataset_dir + '/im_low_res_seq_noisy.npy')

    am_frames = am_frames[0:5]
    phase_frames = phase_frames[0:5]
    list_of_selected_LED_ids = list_of_selected_LED_ids[0:5]
    im_low_res_seq = im_low_res_seq[0:5]
    #print(im_low_res_seq.shape)

    num_frames = 5
    #num_frames = am_frames.shape[0]
    o_ground_truth_frames = am_frames * np.exp(1j * phase_frames)

    if not args.recover_pupil:
        pupil_parametric_model = None
        P0 = P.copy()

    ## Reconstructions

    ############################# GS
    if os.path.exists("{}/recovered_im_GS.npy".format(dirres)):
            o_high_res_GS = np.load("{}/recovered_im_GS.npy".format(dirres))
    else:

        t0 = time.time()
        o_high_res_GS = solvers.GS_dynamic_solver(im_low_res_seq, P0, shifted_center_x, shifted_center_y, 
                            list_of_selected_LED_ids,
                            upsampling_ratio = upsampling_ratio, 
                            epoch_num_per_frame = args.num_epoch,
                            check_period = args.check_period, use_prev_as_initial_guess = args.use_prev_as_initial_guess, err_tol = args.err_tol, o_GT = o_ground_truth_frames, do_bestRSNR=args.do_bestRSNR)
        
        tGS = time.time()
        print('GS takes %.2f seconds'%(tGS-t0))
        np.save("{}/recovered_im_GS.npy".format(dirres), o_high_res_GS)
        with open('{}/Time_GS.txt'.format(dirres), 'w') as f:
            f.write('%.4f'%(tGS))

        rsnr_GS, o_high_res_GS_scaled = utils.rsnr_for_dynamic(o_ground_truth_frames, o_high_res_GS)
        print('RSNR for GS method is ', rsnr_GS)

        compute_psnr_and_save_results(am_frames, phase_frames, o_high_res_GS_scaled, 'GS', amp_range, phase_range, dirres = dirres)
        snr_list_GS = utils.compute_separate_snr(o_high_res_GS_scaled, o_ground_truth_frames)
        print(snr_list_GS)
        with open('{}/results_{}.txt'.format(dirres,'GS'), 'w') as f:
            f.write('RSNR for %s method is %.2f'%('GS', rsnr_GS))
            f.write(str(snr_list_GS))


    ####################################### DC
    if args.run_DC:
        t0 = time.time()
        o_high_res_DC, coef_list_DC = solvers.STV_solver(im_low_res_seq, P0, shifted_center_x, shifted_center_y, 
                                                list_of_selected_LED_ids,
                                                upsampling_ratio = upsampling_ratio, epoch_num = args.num_epoch, 
                                                check_period = args.check_period,
                                                lr = args.lr, 
                                                batch_size_for_frames = args.batch_size_for_frames, batch_size_for_led = args.batch_size_for_led,
                                                weight_reg = 0.0,
                                                use_activation = args.use_activation,
                                                pupil_parametric_model = pupil_parametric_model,
                                                RUN_ON_GPU = RUN_ON_GPU, o_GT = o_ground_truth_frames,
                                                amp_range = amp_range, phase_range = phase_range,
                                                err_tol = args.err_tol, o_high_res0 = o_high_res_GS, do_bestRSNR=args.do_bestRSNR)

        tDC = time.time()
        print('DC takes %.2f seconds'%(tDC-t0))
        np.save("{}/recovered_im_DC.npy".format(dirres), o_high_res_DC)
        with open('{}/Time_DC.txt'.format(dirres), 'w') as f:
            f.write('%.4f'%(tDC))

        rsnr_DC, o_high_res_DC_scaled = utils.rsnr_for_dynamic(o_ground_truth_frames, o_high_res_DC)
        print('RSNR for DC method is ', rsnr_DC)

        compute_psnr_and_save_results(am_frames, phase_frames, o_high_res_DC_scaled, 'DC', amp_range, phase_range, dirres = dirres)
        snr_list_DC = utils.compute_separate_snr(o_high_res_DC_scaled, o_ground_truth_frames)
        print(snr_list_DC)
        with open('{}/results_{}.txt'.format(dirres,'DC'), 'w') as f:
            f.write('RSNR for %s method is %.2f'%('DC', rsnr_DC))
            f.write(str(snr_list_DC))

        if args.recover_pupil:
            np.save("{}/recovered_coefs_DC.npy".format(dirres), coef_list_DC)
            P_DC, snr_pupil_DC, rsnr_pupil_DC = utils.compare_pupils(FP, P, coef_list_DC, 'DC', RUN_ON_GPU, dirres = dirres)
            with open('{}/pupil_coeff_{}.txt'.format(dirres, 'DC'),'w') as f:
                f.write(str(coef_list_DC))
                f.write('\n SNR ')
                f.write(str(snr_pupil_DC))
                f.write('\n RSNR ')
                f.write(str(rsnr_pupil_DC))


    ####################################### STV
    if args.run_STV:
        weight_reg_STV = (args.STV_weight_reg_spatial_amp, args.STV_weight_reg_spatial_phase)
        t0 = time.time()
        o_high_res_STV, coef_list_STV = solvers.STV_solver(im_low_res_seq, P0, shifted_center_x, shifted_center_y, 
                                                list_of_selected_LED_ids,
                                                upsampling_ratio = upsampling_ratio, epoch_num = args.num_epoch, 
                                                check_period = args.check_period,
                                                lr = args.lr, 
                                                batch_size_for_frames = args.batch_size_for_frames, batch_size_for_led = args.batch_size_for_led,
                                                weight_reg = weight_reg_STV,
                                                use_activation = args.use_activation,
                                                pupil_parametric_model = pupil_parametric_model,
                                                RUN_ON_GPU = RUN_ON_GPU, o_GT = o_ground_truth_frames,
                                                amp_range = amp_range, phase_range = phase_range,
                                                err_tol = args.err_tol, o_high_res0 = o_high_res_GS, do_bestRSNR=args.do_bestRSNR)

        tSTV = time.time()
        print('STV takes %.2f seconds'%(tSTV-t0))
        np.save("{}/recovered_im_STV.npy".format(dirres), o_high_res_STV)
        with open('{}/Time_STV.txt'.format(dirres), 'w') as f:
            f.write('%.4f'%(tSTV))

        rsnr_STV, o_high_res_STV_scaled = utils.rsnr_for_dynamic(o_ground_truth_frames, o_high_res_STV)
        print('RSNR for DC method is ', rsnr_STV)

        compute_psnr_and_save_results(am_frames, phase_frames, o_high_res_STV_scaled, 'STV', amp_range, phase_range, dirres = dirres)
        snr_list_STV = utils.compute_separate_snr(o_high_res_STV_scaled, o_ground_truth_frames)
        print(snr_list_STV)
        with open('{}/results_{}.txt'.format(dirres,'STV'), 'w') as f:
            f.write('RSNR for %s method is %.2f'%('STV', rsnr_STV))
            f.write(str(snr_list_STV))

        if args.recover_pupil:
            np.save("{}/recovered_coefs_STV.npy".format(dirres), coef_list_STV)
            P_STV, snr_pupil_STV, rsnr_pupil_STV = utils.compare_pupils(FP, P, coef_list_STV, 'STV', RUN_ON_GPU, dirres = dirres)
            with open('{}/pupil_coeff_{}.txt'.format(dirres, 'STV'),'w') as f:
                f.write(str(coef_list_STV))
                f.write('\n SNR ')
                f.write(str(snr_pupil_STV))
                f.write('\n RSNR ')
                f.write(str(rsnr_pupil_STV))


    ####################################### STTV
    if args.run_STTV:
        weight_reg_STTV = (args.STTV_weight_reg_spatial_amp, args.STTV_weight_reg_spatial_phase, args.STTV_weight_reg_temporal_amp, args.STTV_weight_reg_temporal_phase)
        t0 = time.time()

        o_high_res_STTV, coef_list_STTV = solvers.STTV_solver(im_low_res_seq, P0, shifted_center_x, shifted_center_y, 
                                                              list_of_selected_LED_ids,
                                                              upsampling_ratio = upsampling_ratio, epoch_num = args.num_epoch, 
                                                              check_period = args.check_period,
                                                              lr = args.lr, 
                                                              batch_size_for_frames = args.batch_size_for_frames, batch_size_for_led = args.batch_size_for_led,
                                                              weight_reg = weight_reg_STTV,
                                                              use_activation = args.use_activation,
                                                              pupil_parametric_model = pupil_parametric_model,
                                                              RUN_ON_GPU = RUN_ON_GPU, o_GT = o_ground_truth_frames,
                                                              amp_range = amp_range, phase_range = phase_range,
                                                              err_tol = args.err_tol, o_high_res0 = o_high_res_GS, do_bestRSNR=args.do_bestRSNR)

        tSTTV = time.time()
        print('STTV takes %.2f seconds'%(tSTTV-t0))
        np.save("{}/recovered_im_STTV.npy".format(dirres), o_high_res_STTV)
        with open('{}/Time_STTV.txt'.format(dirres), 'w') as f:
            f.write('%.4f'%(tSTTV))

        rsnr_STTV, o_high_res_STTV_scaled = utils.rsnr_for_dynamic(o_ground_truth_frames, o_high_res_STTV)
        print('RSNR for DC method is ', rsnr_STTV)

        compute_psnr_and_save_results(am_frames, phase_frames, o_high_res_STTV_scaled, 'STTV', amp_range, phase_range, dirres = dirres)
        snr_list_STTV = utils.compute_separate_snr(o_high_res_STTV_scaled, o_ground_truth_frames)
        print(snr_list_STTV)
        with open('{}/results_{}.txt'.format(dirres,'STTV'), 'w') as f:
            f.write('RSNR for %s method is %.2f'%('STTV', rsnr_STTV))
            f.write(str(snr_list_STTV))

        if args.recover_pupil:
            np.save("{}/recovered_coefs_STTV.npy".format(dirres), coef_list_STTV)
            P_STTV, snr_pupil_STTV, rsnr_pupil_STTV = utils.compare_pupils(FP, P, coef_list_STTV, 'STTV', RUN_ON_GPU, dirres = dirres)
            with open('{}/pupil_coeff_{}.txt'.format(dirres, 'STTV'),'w') as f:
                f.write(str(coef_list_STTV))
                f.write('\n SNR ')
                f.write(str(snr_pupil_STTV))
                f.write('\n RSNR ')
                f.write(str(rsnr_pupil_STTV))



    ####################################### DSTP
    if args.run_DSTP:

        if not os.path.exists(dirres + 'parameters'):
            os.mkdir(dirres + 'parameters')

        nl_layer = nn.ReLU()
        dim = 64

        # latent vectors
        z_start = np.random.randn(dim)
        z_end = np.random.randn(dim)
        t = np.linspace(0, 1, num = num_frames)
        z_fixed = []
        for i in range(num_frames):
            z_fixed.append((1 - t[i]) * z_start + t[i] * z_end)
        z_fixed = np.stack([z_fixed])
        #z_fixed = np.stack([z_fixed, z_fixed])
        np.save('{}/z_fixed.npy'.format(dirres), z_fixed)

        constant_width = 128 
        interp_mode = 'nearest'

        t0 = time.time()
        o_high_res_DSTP, o_high_res_initial_DSTP, loss_track_DSTP, loss_track_initial_DSTP, rsnr_track_DSTP, coef_list_DSTP = solvers.DSTP_solver(im_low_res_seq, P0, shifted_center_x, shifted_center_y, 
        list_of_selected_LED_ids,
        z_fixed, initialize_net = args.initialize_net, upsampling_ratio = upsampling_ratio, epoch_num = args.num_epoch_DSTP, 
        check_period = args.check_period,
        lr = args.lr_DSTP, batch_size_for_frames = args.batch_size_for_frames, batch_size_for_led = args.batch_size_for_led,
        constant_width = constant_width, interp_mode = interp_mode,
        nl_layer = nl_layer, use_batchnorm = args.use_batchnorm,
        pupil_parametric_model = pupil_parametric_model,
        RUN_ON_GPU = RUN_ON_GPU, 
        o_GT = o_ground_truth_frames, 
        amp_range = amp_range, phase_range = phase_range, err_tol = args.err_tol,
        o_high_res0 = o_high_res_GS, do_bestRSNR = args.do_bestRSNR, res_dir=dirres, freeze_bn = args.freeze_bn, weight_decay=0.0)

        tDSTP = time.time()
        print('DSTP takes %.2f seconds'%(tDSTP-t0))
        np.save("{}/recovered_im_DSTP.npy".format(dirres), o_high_res_DSTP)
        with open('{}/Time_DSTP.txt'.format(dirres), 'w') as f:
            f.write('%.4f'%(tDSTP))

        rsnr_DSTP, o_high_res_DSTP_scaled = utils.rsnr_for_dynamic(o_ground_truth_frames, o_high_res_DSTP)
        print('RSNR for DSTP method is ', rsnr_DSTP)

        compute_psnr_and_save_results(am_frames, phase_frames, o_high_res_DSTP_scaled, 'DSTP', amp_range, phase_range, dirres = dirres)
        snr_list_DSTP = utils.compute_separate_snr(o_high_res_DSTP_scaled, o_ground_truth_frames)
        print(snr_list_DSTP)
        with open('{}/results_{}.txt'.format(dirres,'DSTP'), 'w') as f:
            f.write('RSNR for %s method is %.2f'%('DSTP', rsnr_DSTP))
            f.write(str(snr_list_DSTP))

        if args.recover_pupil:
            np.save("{}/recovered_coefs_DSTP.npy".format(dirres), coef_list_DSTP)
            P_DSTP, snr_pupil_DSTP, rsnr_pupil_DSTP = utils.compare_pupils(FP, P, coef_list_DSTP, 'DSTP', RUN_ON_GPU, dirres = dirres)
            with open('{}/pupil_coeff_{}.txt'.format(dirres, 'DSTP'),'w') as f:
                f.write(str(coef_list_DSTP))
                f.write('\n SNR ')
                f.write(str(snr_pupil_DSTP))
                f.write('\n RSNR ')
                f.write(str(rsnr_pupil_DSTP))