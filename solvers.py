from fp_operators import OperatorKit
import models
import utils
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter


## Gerchberg-Saxton
def GS_solver(im_low_res_seq, P, shifted_center_x, shifted_center_y, 
              upsampling_ratio = 4, epoch_num = 10, monitor_result = False, 
              check_period = 10, err_tol = 1e-4, o_high_res0 = None, o_GT = None, do_bestRSNR=True):
    
    num_im, m1, n1 = im_low_res_seq.shape
    
    operators = OperatorKit(P, m1, n1, upsampling_ratio = upsampling_ratio)
       
    m = m1 * upsampling_ratio
    n = n1 * upsampling_ratio
    
    if o_high_res0 is None:
        o_high_res = cv2.resize(im_low_res_seq[0], (n, m), interpolation=cv2.INTER_CUBIC)
    else:
        o_high_res = o_high_res0.copy()
    o_hat = operators.FFT(o_high_res)
    
    o_high_res_list = []
    if monitor_result:
        o_high_res_list.append(o_high_res)
    
    best_rsnr = -1000
    best_o_high_res = None
    for i in range(epoch_num):
        
        for l in range(num_im):
            xc = shifted_center_x[l]
            yc = shifted_center_y[l]
            y_l = operators.H_l_operator(o_hat, xc, yc)
            y_l_projected = operators.amplitude_projector(y_l, im_low_res_seq[l])
            o_hat = operators.H_l_inv_operator(y_l_projected, xc, yc, o_hat_init = o_hat)
            
        if (monitor_result and (i % check_period == 0)) or i == (epoch_num - 1):
            o_high_res_new = operators.iFFT(o_hat)
            o_high_res_list.append(o_high_res_new)
            
            if o_GT is not None:
                rsnr, o_high_res_scaled = utils.rsnr_for_dynamic(o_GT, o_high_res_new)
                
                if rsnr > best_rsnr:
                    best_rsnr = rsnr
                    best_o_high_res = o_high_res_new
            if utils.l2_norm_squared(o_high_res_new - o_high_res) < err_tol:
                print('No change in result, break on epoch %.0f'%i)
                o_high_res = o_high_res_new
                break
            o_high_res = o_high_res_new
    if do_bestRSNR and best_o_high_res is not None:
        o_high_res = best_o_high_res
    
    return o_high_res, o_high_res_list


## Gerchberg-Saxton for sequences (frame-by-frame)
def GS_dynamic_solver(im_low_res_seq, P, shifted_center_x, shifted_center_y, 
                      list_of_selected_LED_ids,
                      upsampling_ratio = 4, epoch_num_per_frame = 10,
                      check_period = 10, use_prev_as_initial_guess = True, err_tol = 1e-4, o_GT = None, do_bestRSNR=True):
    num_frames = im_low_res_seq.shape[0]
    o_high_res0 = None
    o_high_res_out = []
    for i in range(num_frames):
        print('Frame number :', i)
        im_low_res_sub_seq = im_low_res_seq[i].copy()
        selected_LED_ids = list_of_selected_LED_ids[i].copy()
        if o_GT is not None:
            co_GT = o_GT[i]
        else:
            co_GT = None
        o_high_res, o_high_res_list = GS_solver(im_low_res_sub_seq, P, 
                                    shifted_center_x[selected_LED_ids], shifted_center_y[selected_LED_ids], 
                                    upsampling_ratio = upsampling_ratio, epoch_num = epoch_num_per_frame, monitor_result = True, 
                                    check_period = check_period, err_tol = err_tol, o_high_res0 = o_high_res0, o_GT = co_GT,do_bestRSNR=do_bestRSNR)
        o_high_res_out.append(o_high_res)
        if use_prev_as_initial_guess:
            o_high_res0 = o_high_res.copy()
    return np.stack(o_high_res_out)


## (Spatial) TV solver for 1 frame (No pupil estimation)
def TV_solver(im_low_res_seq, P, shifted_center_x, shifted_center_y, 
              upsampling_ratio = 4, epoch_num = 10, monitor_result = False, 
              check_period = 10,
              lr = 1e-2, batch_size = 5, weight_reg = 1e-3, use_activation = True,
              RUN_ON_GPU = False, err_tol = 1e-4, o_am0 = None, o_ph0 = None):
    
    num_im, m1, n1 = im_low_res_seq.shape
    m = m1 * upsampling_ratio
    n = n1 * upsampling_ratio        

    FP_model = models.FP_model(P, shifted_center_x, shifted_center_y, m1, n1, 
                               upsampling_ratio = upsampling_ratio, 
                               pupil_parametric_model = None, 
                               RUN_ON_GPU = RUN_ON_GPU)
    FP_model = utils.move_model(FP_model, RUN_ON_GPU)
    
    coef = None
        
    DataLoss = models.AmpDataLoss()
    RegLoss = models.RegLoss(weight_reg)
    
    if o_am0 is None:
        o_am = cv2.resize(im_low_res_seq[0], (n, m), interpolation = cv2.INTER_CUBIC)
    else:
        o_am = o_am0.copy()
    
    if o_ph0 is None:
        o_ph = np.zeros((m, n))
    else:
        o_ph = o_ph0.copy()
    
    o_high_res = o_am * np.exp(1j * o_ph)
    o_high_res_list = []
    if monitor_result:
        o_high_res_list.append(o_high_res)       
    
    if use_activation:
        o_ph = np.arctanh(o_ph / np.pi) 
    
    o_am = utils.np_to_var(o_am, RUN_ON_GPU)
    o_ph = utils.np_to_var(o_ph, RUN_ON_GPU)
    
    o_am = torch.unsqueeze(o_am, dim = 0)
    o_ph = torch.unsqueeze(o_ph, dim = 0)
    
    o_am.requires_grad = True
    o_ph.requires_grad = True  
    optimizer = optim.Adam([o_am, o_ph], lr = lr, amsgrad = True)
    
    loss_track = []
    coef_list = []

    im_low_res_seq_intensity = im_low_res_seq ** 2
    
    for i in range(epoch_num):
        # sampling views
        LED_ids_all = np.arange(num_im)
        np.random.shuffle(LED_ids_all)
        list_of_selected_LED_ids = np.split(LED_ids_all, np.arange(batch_size, num_im, batch_size))

        for selected_LED_ids in list_of_selected_LED_ids:
            # load data
            target = im_low_res_seq_intensity[selected_LED_ids]
            target = utils.np_to_var(target, RUN_ON_GPU)
            target = torch.unsqueeze(target, dim = 0)
            
            optimizer.zero_grad()
            
            # forward
            if use_activation:
                pred = FP_model.forward(models.DReLU(o_am), np.pi * torch.tanh(o_ph), np.expand_dims(selected_LED_ids, axis = 0), coef = coef)
            else:
                pred = FP_model.forward(o_am, o_ph, np.expand_dims(selected_LED_ids, axis = 0), coef = coef)
            
            loss0 = DataLoss.forward(pred, target)
            if np.any(np.array(weight_reg) > 0):
                loss = loss0 + RegLoss.TV_separate(FP_model.o_am, FP_model.o_ph)
            else:
                loss = loss0
            loss.backward()
            optimizer.step()     
            
            with torch.no_grad():
                loss_track.append(loss0.data.item())
            
        with torch.no_grad():
            if (monitor_result and (i % check_period == 0)) or i == (epoch_num - 1):
                if use_activation:
                    o_high_res_new = models.DReLU(o_am) * torch.exp(1j * np.pi * torch.tanh(o_ph))
                else:
                    o_high_res_new = o_am * torch.exp(1j * o_ph)
                o_high_res_new = utils.var_to_np(o_high_res_new[0], RUN_ON_GPU)
                o_high_res_list.append(o_high_res_new)
                
                if utils.l2_norm_squared(o_high_res_new - o_high_res) < err_tol:
                    o_high_res = o_high_res_new
                    print('No change in result, break on epoch %.0f'%i)
                    break
                o_high_res = o_high_res_new
    
    return o_high_res, o_high_res_list, loss_track
    

## Spatially TV-regularized solver
def STV_solver(im_low_res_seq, P, shifted_center_x, shifted_center_y, 
                       list_of_selected_LED_ids,
                       upsampling_ratio = 4, epoch_num = 40, 
                       check_period = 10,
                       lr = 1e-3, 
                       batch_size_for_frames = 5, batch_size_for_led = 5,
                       weight_reg = 0.001,
                       use_activation = True,
                       pupil_parametric_model = None,
                       RUN_ON_GPU = False, o_GT = None,
                       amp_range = (0, 1), phase_range = (-np.pi / 2, np.pi / 2),
                       err_tol = 1e-4, o_high_res0 = None, do_bestRSNR=True):
        
    if (pupil_parametric_model is None):
        num_frames = im_low_res_seq.shape[0]
        o_am0 = None
        o_ph0 = None
        o_high_res_out = []
        for i in range(num_frames):
            print('Frame number :', i)
            im_low_res_sub_seq = im_low_res_seq[i].copy()
            selected_LED_ids = list_of_selected_LED_ids[i].copy()
        
            o_high_res, _ , _ = TV_solver(im_low_res_sub_seq, P, shifted_center_x[selected_LED_ids], 
                                                                shifted_center_y[selected_LED_ids], upsampling_ratio = upsampling_ratio, epoch_num = epoch_num, monitor_result = False, check_period = check_period, lr = lr, batch_size = batch_size_for_led, weight_reg = weight_reg, use_activation = use_activation, RUN_ON_GPU = RUN_ON_GPU, err_tol = err_tol, o_am0 = o_am0, o_ph0 = o_ph0)
        
            o_high_res_out.append(o_high_res)
            o_high_res0 = o_high_res.copy()
            o_am0 = np.absolute(o_high_res0)
            o_ph0 = np.angle(o_high_res0)

        o_high_res = np.stack(o_high_res_out)
        return o_high_res, None

    else:

        rsnr_track = []
        num_frames, num_views, m1, n1 = im_low_res_seq.shape
    
        batch_size_for_frames = min(num_frames, batch_size_for_frames)
        batch_size_for_led = min(num_views, batch_size_for_led)
    
        DataLoss = models.AmpDataLoss()
        RegLoss = models.RegLoss(weight_reg)
    
        FP_model = models.FP_model(P, shifted_center_x, shifted_center_y, m1, n1, 
                                  upsampling_ratio = upsampling_ratio,
                                  pupil_parametric_model = pupil_parametric_model,
                                  RUN_ON_GPU = RUN_ON_GPU)
    
        if o_high_res0 is None:
            o_high_res = GS_dynamic_solver(im_low_res_seq, P, shifted_center_x, shifted_center_y, 
                                           list_of_selected_LED_ids,
                                           upsampling_ratio = upsampling_ratio, epoch_num_per_frame = 100,
                                           check_period = 10, use_prev_as_initial_guess = True, err_tol = 1e-1)
        else:
            o_high_res = o_high_res0
        
        num_coef = pupil_parametric_model['modes'].shape[0]
        coef = utils.np_to_var(np.zeros(num_coef), RUN_ON_GPU)
        coef.requires_grad = True
        optimizer_coef = optim.Adam([coef], lr = lr, amsgrad = True)
        
        loss_track = []
    
        o_ph = np.angle(o_high_res)
        if use_activation:
            o_ph = np.arctanh(o_ph / np.pi) 
        o_am = utils.np_to_var(np.absolute(o_high_res), RUN_ON_GPU)
        o_am = torch.unsqueeze(o_am, dim = 1)
        o_ph = utils.np_to_var(o_ph, RUN_ON_GPU)        
        o_ph = torch.unsqueeze(o_ph, dim = 1)
    
        o_am.requires_grad = True
        o_ph.requires_grad = True
        optimizer = optim.Adam([o_am, o_ph], lr = lr, amsgrad = True)
    
        if o_GT is not None:
            rsnr, o_high_res_scaled = utils.rsnr_for_dynamic(o_GT, o_high_res)
            rsnr_track.append(rsnr)
        
            o_high_res_scaled = np.expand_dims(o_high_res_scaled, axis = 1)
        
            amp_estimate = np.absolute(o_high_res_scaled)
            phase_estimate = np.angle(o_high_res_scaled)
            amp_estimate = utils.unnormalize_img(amp_estimate, amp_range)
            phase_estimate = utils.unnormalize_img(phase_estimate, phase_range)
        
        im_low_res_seq_intensity = im_low_res_seq ** 2
        
        coef_list = [] 
    
        loss_min = 1000
        best_rsnr = -100
        best_o_high_res = None
        for i in range(epoch_num):
            # list_of_selected_LED_ids
            sampling_matrix = [np.random.permutation(np.arange(num_views)) for j in range(num_frames)]
            sampling_matrix = np.stack(sampling_matrix)

            epoch_loss = 0
            for view_start_id in range(0, num_views, batch_size_for_led):
                view_end_id = min(view_start_id + batch_size_for_led, num_views)
            
                frame_ids_all = np.arange(num_frames)
                np.random.shuffle(frame_ids_all)
                list_of_selected_frame_ids = np.split(frame_ids_all, np.arange(batch_size_for_frames, num_frames, batch_size_for_frames))
                for selected_frame_ids in list_of_selected_frame_ids:
                    sampling_ids = sampling_matrix[selected_frame_ids, :][:, view_start_id:view_end_id].copy()
                    selected_LED_ids = [list_of_selected_LED_ids[frame_id, sampling_ids[j]] for j, frame_id in enumerate(selected_frame_ids)]
                    selected_LED_ids = np.stack(selected_LED_ids)
                
                    target = [im_low_res_seq_intensity[frame_id][sampling_ids[j]] for j, frame_id in enumerate(selected_frame_ids)]
                    target = np.stack(target)
                    target = utils.np_to_var(target, RUN_ON_GPU)
                
                    optimizer.zero_grad()
                    optimizer_coef.zero_grad()
                
                    amp = o_am[selected_frame_ids]
                    phase = o_ph[selected_frame_ids]

                    if use_activation:
                        pred = FP_model.forward(models.DReLU(amp[:,0]), np.pi * torch.tanh(phase[:,0]), 
                                                selected_LED_ids, indices_frame = np.arange(selected_frame_ids.shape[-1]),
                                                coef = coef)
                    else:
                        pred = FP_model.forward(amp[:,0],phase[:,0], 
                                                selected_LED_ids, indices_frame = np.arange(selected_frame_ids.shape[-1]),
                                                coef = coef) 
                    
                    loss0 = DataLoss.forward(pred, target)
                    if np.any(np.array(weight_reg) > 0):
                        loss = loss0 + RegLoss.TV_separate(FP_model.o_am, FP_model.o_ph)
                    else:
                        loss = loss0
                
                    loss.backward()
                
                    optimizer.step()
                    optimizer_coef.step()
                
                    with torch.no_grad():
                        epoch_loss += loss.data.item() * selected_frame_ids.shape[-1] * (view_end_id - view_start_id)
                    
            with torch.no_grad():
                epoch_loss /= (num_views * num_frames)
                loss_track.append(epoch_loss)
                #writer.add_scalar('Average epoch loss', epoch_loss, global_step = i + 1)
                if o_GT is not None:
                    print('Epoch [{:6d}/{:6d}] | loss: {:e} | SNR: {:e}'.format(i + 1, epoch_num, epoch_loss,rsnr))
                else:
                    print('Epoch [{:6d}/{:6d}] | loss: {:e}'.format(i + 1, epoch_num, epoch_loss))
                
                if epoch_loss < loss_min:
                    loss_min = epoch_loss
                
                if (i % check_period == 0) or i == (epoch_num - 1):
                    if use_activation:
                        o_high_res_new = models.DReLU(o_am) * torch.exp(1j * np.pi * torch.tanh(o_ph))
                    else:
                        o_high_res_new = o_am * torch.exp(1j * o_ph)   
                    o_high_res_new = o_high_res_new[:, 0, :, :]
                    o_high_res_new = utils.var_to_np(o_high_res_new, RUN_ON_GPU)
                    if o_GT is not None:
                        rsnr, o_high_res_scaled = utils.rsnr_for_dynamic(o_GT, o_high_res_new)
                        rsnr_track.append(rsnr)
                        if rsnr > best_rsnr:
                            best_rsnr = rsnr
                            best_o_high_res = o_high_res_new
                    
                        o_high_res_scaled = np.expand_dims(o_high_res_scaled, axis = 1)
                        amp_estimate = np.absolute(o_high_res_scaled)
                        phase_estimate = np.angle(o_high_res_scaled)
                        amp_estimate = utils.unnormalize_img(amp_estimate, amp_range)
                        phase_estimate = utils.unnormalize_img(phase_estimate, phase_range)
                    
                    coef_list.append(utils.var_to_np(coef, RUN_ON_GPU))

                    if utils.l2_norm_squared(o_high_res_new - o_high_res) < err_tol or epoch_loss < 1e-10:
                        o_high_res = o_high_res_new.copy()
                        break
                    o_high_res = o_high_res_new.copy()
                
        with torch.no_grad():
            if use_activation:
                o_high_res = models.DReLU(o_am) * torch.exp(1j * np.pi * torch.tanh(o_ph))
            else:
                o_high_res = o_am * torch.exp(1j * o_ph)
            o_high_res = o_high_res[:, 0, :, :]
            o_high_res = utils.var_to_np(o_high_res, RUN_ON_GPU)
        
        if do_bestRSNR and best_o_high_res is not None:
            o_high_res = best_o_high_res
        
        return o_high_res, np.stack(coef_list)
        

# Spatiotemporally TV-regularized solver (use only with full batch size for frames)
def STTV_solver(im_low_res_seq, P, shifted_center_x, shifted_center_y, 
                          list_of_selected_LED_ids,
                          upsampling_ratio = 4, epoch_num = 40, 
                          check_period = 10,
                          lr = 1e-3, 
                          batch_size_for_frames = 5, batch_size_for_led = 5,
                          weight_reg = 0.001,
                          use_activation = True,
                          pupil_parametric_model = None,
                          RUN_ON_GPU = False, o_GT = None,
                          amp_range = (0, 1), phase_range = (-np.pi / 2, np.pi / 2),
                          err_tol = 1e-4, o_high_res0 = None, do_bestRSNR=True):
    
    rsnr_track = []
    num_frames, num_views, m1, n1 = im_low_res_seq.shape
    
    batch_size_for_frames = min(num_frames, batch_size_for_frames)
    batch_size_for_led = min(num_views, batch_size_for_led)
    
    DataLoss = models.AmpDataLoss()
    RegLoss = models.RegLoss(weight_reg, TVtime = True)
    
    FP_model = models.FP_model(P, shifted_center_x, shifted_center_y, m1, n1, 
                                  upsampling_ratio = upsampling_ratio, 
                                  pupil_parametric_model = pupil_parametric_model,
                                  RUN_ON_GPU = RUN_ON_GPU)
    
    coef0 = None

    if o_high_res0 is None:
        o_high_res = GS_dynamic_solver(im_low_res_seq, P, shifted_center_x, shifted_center_y, 
                                       list_of_selected_LED_ids,
                                       upsampling_ratio = upsampling_ratio, epoch_num_per_frame = 100,
                                       check_period = 10, use_prev_as_initial_guess = True, err_tol = 1e-1)
    else:
        o_high_res = o_high_res0
        
    if pupil_parametric_model is None:
        coef = None
        recover_pupil = 0
    else:
        num_coef = pupil_parametric_model['modes'].shape[0]

        if coef0 is None:
            coef = utils.np_to_var(np.zeros(num_coef), RUN_ON_GPU)
        else:
            coef = utils.np_to_var(coef0, RUN_ON_GPU)

        coef.requires_grad = True
        optimizer_coef = optim.Adam([coef], lr = lr, amsgrad = True)
        recover_pupil = 1    
    
    loss_track = []
    
    o_ph = np.angle(o_high_res)
    if use_activation:
        o_ph = np.arctanh(o_ph / np.pi) 
    o_am = utils.np_to_var(np.absolute(o_high_res), RUN_ON_GPU)
    o_am = torch.unsqueeze(o_am, dim = 1)
    o_ph = utils.np_to_var(o_ph, RUN_ON_GPU)        
    o_ph = torch.unsqueeze(o_ph, dim = 1)
    
    o_am.requires_grad = True
    o_ph.requires_grad = True
    
    optimizer = optim.Adam([o_am, o_ph], lr = lr, amsgrad = True)
    
    if o_GT is not None:
        rsnr, o_high_res_scaled = utils.rsnr_for_dynamic(o_GT, o_high_res)
        rsnr_track.append(rsnr)
        
        o_high_res_scaled = np.expand_dims(o_high_res_scaled, axis = 1)
        
        amp_estimate = np.absolute(o_high_res_scaled)
        phase_estimate = np.angle(o_high_res_scaled)
        amp_estimate = utils.unnormalize_img(amp_estimate, amp_range)
        phase_estimate = utils.unnormalize_img(phase_estimate, phase_range)
        
    im_low_res_seq_intensity = im_low_res_seq ** 2
    
    coef_list = [] 
    
    loss_min = 1000
    best_rsnr = -1000
    best_o_high_res = None
    for i in range(epoch_num):
        # list_of_selected_LED_ids
        sampling_matrix = [np.random.permutation(np.arange(num_views)) for j in range(num_frames)]
        sampling_matrix = np.stack(sampling_matrix)

        epoch_loss = 0
        for view_start_id in range(0, num_views, batch_size_for_led):
            view_end_id = min(view_start_id + batch_size_for_led, num_views)
            
            frame_ids_all = np.arange(num_frames)
            np.random.shuffle(frame_ids_all)
            list_of_selected_frame_ids = np.split(frame_ids_all, np.arange(batch_size_for_frames, num_frames, batch_size_for_frames))
            for selected_frame_ids in list_of_selected_frame_ids:
                sampling_ids = sampling_matrix[selected_frame_ids, :][:, view_start_id:view_end_id].copy()
                selected_LED_ids = [list_of_selected_LED_ids[frame_id, sampling_ids[j]] for j, frame_id in enumerate(selected_frame_ids)]
                selected_LED_ids = np.stack(selected_LED_ids)
                
                target = [im_low_res_seq_intensity[frame_id][sampling_ids[j]] for j, frame_id in enumerate(selected_frame_ids)]
                target = np.stack(target)
                target = utils.np_to_var(target, RUN_ON_GPU)
                
                optimizer.zero_grad()
                if recover_pupil:
                    optimizer_coef.zero_grad()
                
                amp = o_am[selected_frame_ids]
                phase = o_ph[selected_frame_ids]
                
                if use_activation:
                    pred = FP_model.forward(models.DReLU(amp[:,0]), np.pi * torch.tanh(phase[:,0]), 
                                            selected_LED_ids, indices_frame = np.arange(selected_frame_ids.shape[-1]),
                                            coef = coef) 
                else:
                    pred = FP_model.forward(amp[:,0],phase[:,0], 
                                            selected_LED_ids, indices_frame = np.arange(selected_frame_ids.shape[-1]),
                                            coef = coef)  
                
                loss0 = DataLoss.forward(pred, target)
                if np.any(np.array(weight_reg) > 0):
                    loss = loss0 + RegLoss.TV_separate(FP_model.o_am, FP_model.o_ph)
                else:
                    loss = loss0
                
                loss.backward()
                
                optimizer.step()
                
                if recover_pupil:
                    optimizer_coef.step()
                
                with torch.no_grad():
                    epoch_loss += loss.data.item() * selected_frame_ids.shape[-1] * (view_end_id - view_start_id)
                       
        with torch.no_grad():
            epoch_loss /= (num_views * num_frames)
            loss_track.append(epoch_loss)
            if o_GT is not None:
                print('Epoch [{:6d}/{:6d}] | loss: {:e} | RSNR: {:e}'.format(i + 1, epoch_num, epoch_loss,rsnr))
            else:    
                print('Epoch [{:6d}/{:6d}] | loss: {:e}'.format(i + 1, epoch_num, epoch_loss))
            
            if epoch_loss < loss_min:
                loss_min = epoch_loss
                
            if (i % check_period == 0) or i == (epoch_num - 1):
                if use_activation:
                    o_high_res_new = models.DReLU(o_am) * torch.exp(1j * np.pi * torch.tanh(o_ph))
                else:
                    o_high_res_new = o_am * torch.exp(1j * o_ph)
                o_high_res_new = o_high_res_new[:, 0, :, :]
                o_high_res_new = utils.var_to_np(o_high_res_new, RUN_ON_GPU)
                
                if o_GT is not None:
                    rsnr, o_high_res_scaled = utils.rsnr_for_dynamic(o_GT, o_high_res_new)
                    rsnr_track.append(rsnr)
                    if rsnr > best_rsnr:
                        best_rsnr = rsnr
                        best_o_high_res = o_high_res_new
                    
                    o_high_res_scaled = np.expand_dims(o_high_res_scaled, axis = 1)
                    amp_estimate = np.absolute(o_high_res_scaled)
                    phase_estimate = np.angle(o_high_res_scaled)
                    amp_estimate = utils.unnormalize_img(amp_estimate, amp_range)
                    phase_estimate = utils.unnormalize_img(phase_estimate, phase_range)
                    
                if recover_pupil:
                    coef_list.append(utils.var_to_np(coef, RUN_ON_GPU))

                if utils.l2_norm_squared(o_high_res_new - o_high_res) < err_tol or epoch_loss < 1e-10:
                    o_high_res = o_high_res_new.copy()
                    break
                o_high_res = o_high_res_new.copy()
                
    with torch.no_grad():
        if use_activation:
            o_high_res = models.DReLU(o_am) * torch.exp(1j * np.pi * torch.tanh(o_ph))
        else:
            o_high_res = o_am * torch.exp(1j * o_ph)
        o_high_res = o_high_res[:, 0, :, :]
        o_high_res = utils.var_to_np(o_high_res, RUN_ON_GPU)
    if do_bestRSNR and best_o_high_res is not None:
        o_high_res = best_o_high_res
    if len(coef_list) > 0:
        return o_high_res, np.stack(coef_list)
    else:
        return o_high_res, None
    

## Deep Spatiotemporal Prior (DSTP) solver
def DSTP_solver(im_low_res_seq, P, shifted_center_x, shifted_center_y, 
                          list_of_selected_LED_ids,
                          z_fixed, initialize_net = False,
                          upsampling_ratio = 4, epoch_num = 40, 
                          check_period = 10,
                          lr = 1e-3, 
                          batch_size_for_frames = 5, batch_size_for_led = 5,
                          constant_width = 64, interp_mode = 'nearest',
                          nl_layer = nn.LeakyReLU(), use_batchnorm = True,
                          pupil_parametric_model = None,
                          RUN_ON_GPU = False, o_GT = None,
                          amp_range = (0, 1), phase_range = (-np.pi / 2, np.pi / 2),
                          err_tol = 1e-4, o_high_res0=None, do_bestRSNR=True, 
                          res_dir='/home/bohra/data/tmp', freeze_bn=1, weight_decay = 0.0):
    
    weights_name = res_dir + 'parameters/DSTP_model_best_dynamic'
    weights_name_rsnr = res_dir + 'parameters/DSTP_model_best_dynamic_rnsr'    
    
    monitor_dir = res_dir + 'runs'
    writer = SummaryWriter(monitor_dir)
    
    rsnr_track = []
    num_frames, num_views, m1, n1 = im_low_res_seq.shape
    
    batch_size_for_frames = min(num_frames, batch_size_for_frames)
    batch_size_for_led = min(num_views, batch_size_for_led)
    
    batch_size = batch_size_for_frames * batch_size_for_led
    
    m = m1 * upsampling_ratio
    n = n1 * upsampling_ratio        
    image_size_high_res = (m, n)

    DataLoss = models.AmpDataLoss()
    
    FP_model = models.FP_model(P, shifted_center_x, shifted_center_y, m1, n1, 
                               upsampling_ratio = upsampling_ratio, 
                               pupil_parametric_model = pupil_parametric_model,
                               RUN_ON_GPU = RUN_ON_GPU)
    
    net = models.ImageGenerator(image_size_high_res, constant_width = constant_width, 
                                interp_mode = interp_mode, nl_layer = nl_layer, use_batchnorm = use_batchnorm, 
                                block_num = 2)

    num_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of learnable parameters is %.0f'%num_param)

    net = utils.move_model(net, RUN_ON_GPU)
    net.train()

    coef0 = None

    loss_track_initial = []

    if initialize_net > 0:
        print('Start initializing the network parameters...')
        
        if o_high_res0 is None:
            o_high_res0 = GS_dynamic_solver(im_low_res_seq, P, shifted_center_x, shifted_center_y, 
                                            list_of_selected_LED_ids,
                                            upsampling_ratio = upsampling_ratio, epoch_num_per_frame = 1000,
                                            check_period = 10, use_prev_as_initial_guess = True, err_tol = 1e-6)
                
        err_tol_init = 1e-1
        num_iter_for_init = 1000
        lr_init = 1e-3

        amp0 = utils.np_to_var(np.absolute(o_high_res0), RUN_ON_GPU)
        amp0 = torch.unsqueeze(amp0, dim = 1)
        phase0 = utils.np_to_var(np.angle(o_high_res0), RUN_ON_GPU)        
        phase0 = torch.unsqueeze(phase0, dim = 1)

        optimizer = optim.Adam(net.parameters(), lr = lr_init, weight_decay = 0, amsgrad = True)
        
        loss_initial = 100
        i = 0
        while loss_initial > err_tol_init:

            sampling_indices = np.arange(num_frames)
            np.random.shuffle(sampling_indices)
            sampling_indices = sampling_indices[:batch_size]
            
            latent_z = z_fixed[:, sampling_indices].copy() 
            latent_z = utils.np_to_var(latent_z, RUN_ON_GPU)            

            optimizer.zero_grad()

            amp, phase = net.forward(latent_z)
            
            loss = (amp - amp0[sampling_indices]).abs().mean() + (phase - phase0[sampling_indices]).abs().mean()
            
            loss.backward()
            optimizer.step()
            
            loss_initial = loss.data.item()
            i += 1
            print('During initialization, the fitting loss is %.6f'%(loss_initial))
            loss_track_initial.append(loss_initial)
            
            if i > num_iter_for_init:
                print('Exceeds %s iterations, break the initialization...'%(num_iter_for_init))
                break
        
    torch.save(net.state_dict(), weights_name)
    
    if pupil_parametric_model is None:
        coef = None
        recover_pupil = 0
    else:
        num_coef = pupil_parametric_model['modes'].shape[0]

        if coef0 is None:
            coef = utils.np_to_var(np.zeros(num_coef), RUN_ON_GPU)
        else:
            coef = utils.np_to_var(coef0, RUN_ON_GPU)

        coef.requires_grad = True
        optimizer_coef = optim.Adam([coef], lr = lr, amsgrad = True)
        recover_pupil = 1    
    
    optimizer = optim.Adam(net.parameters(), lr = lr, weight_decay = weight_decay, amsgrad = True)

    loss_track = []
    
    with torch.no_grad():
        o_high_res = get_sequence_from_network(net, z_fixed, RUN_ON_GPU = RUN_ON_GPU)
        
    if freeze_bn == 1:
        if (initialize_net > 0):
            net.eval()
   
    o_high_res_initial = o_high_res.copy()
    
    if o_GT is not None:
        rsnr, o_high_res_scaled = utils.rsnr_for_dynamic(o_GT, o_high_res)
        rsnr_track.append(rsnr)
        
        writer.add_scalar('RSNR', rsnr, global_step = 0)
        
        o_high_res_scaled = np.expand_dims(o_high_res_scaled, axis = 1)
        
        amp_estimate = np.absolute(o_high_res_scaled)
        phase_estimate = np.angle(o_high_res_scaled)
        amp_estimate = utils.unnormalize_img(amp_estimate, amp_range)
        phase_estimate = utils.unnormalize_img(phase_estimate, phase_range)
        writer.add_images('Amplitude', amp_estimate, global_step = 0)
        writer.add_images('Phase', phase_estimate, global_step = 0)
        writer.close()
        
    im_low_res_seq_intensity = im_low_res_seq ** 2
    
    count = 0
    
    coef_list = [] 
    
    loss_min = 1000
    best_rsnr = -100
    for i in range(epoch_num):

        sampling_matrix = [np.random.permutation(np.arange(num_views)) for j in range(num_frames)]
        sampling_matrix = np.stack(sampling_matrix)

        epoch_loss = 0
        for view_start_id in range(0, num_views, batch_size_for_led):
            view_end_id = min(view_start_id + batch_size_for_led, num_views)
            
            frame_ids_all = np.arange(num_frames)
            np.random.shuffle(frame_ids_all)
            list_of_selected_frame_ids = np.split(frame_ids_all, np.arange(batch_size_for_frames, num_frames, batch_size_for_frames))
            for selected_frame_ids in list_of_selected_frame_ids:
                sampling_ids = sampling_matrix[selected_frame_ids, :][:, view_start_id:view_end_id].copy()
                selected_LED_ids = [list_of_selected_LED_ids[frame_id, sampling_ids[j]] for j, frame_id in enumerate(selected_frame_ids)]
                selected_LED_ids = np.stack(selected_LED_ids)
                
                # load data
                latent_z = z_fixed[:, selected_frame_ids].copy() 
                latent_z = utils.np_to_var(latent_z, RUN_ON_GPU)            
                
                target = [im_low_res_seq_intensity[frame_id][sampling_ids[j]] for j, frame_id in enumerate(selected_frame_ids)]
                target = np.stack(target)
                target = utils.np_to_var(target, RUN_ON_GPU)
                
                optimizer.zero_grad()
                if recover_pupil:
                    optimizer_coef.zero_grad()
                    
                amp, phase = net.forward(latent_z)
                    
                pred = FP_model.forward(amp[:, 0, :, :], phase[:, 0, :, :], 
                                          selected_LED_ids, indices_frame = np.arange(selected_frame_ids.shape[-1]),
                                          coef = coef)
                
                loss = DataLoss.forward(pred, target)
                
                loss.backward()
                
                optimizer.step()
            
                if recover_pupil:
                    optimizer_coef.step()
                
                with torch.no_grad():
                    epoch_loss += loss.data.item() * selected_frame_ids.shape[-1] * (view_end_id - view_start_id)
                    
        with torch.no_grad():
            epoch_loss /= (num_views * num_frames)
            loss_track.append(epoch_loss)
            writer.add_scalar('Average epoch loss', epoch_loss, global_step = i + 1)
            print('Epoch [{:6d}/{:6d}] | loss: {:.6f}'.format(i + 1, epoch_num, epoch_loss))
            if epoch_loss < loss_min:
                loss_min = epoch_loss
                
                # save model
                torch.save(net.state_dict(), weights_name)

                o_high_res_new = get_sequence_from_network(net, z_fixed, RUN_ON_GPU = RUN_ON_GPU)
    
                if o_GT is not None:
                    rsnr, o_high_res_scaled = utils.rsnr_for_dynamic(o_GT, o_high_res_new)
                    rsnr_track.append(rsnr)
                    if rsnr > best_rsnr:
                        best_rsnr = rsnr
                        torch.save(net.state_dict(), weights_name_rsnr)
                            
                    writer.add_scalar('RSNR', rsnr, global_step = i + 1)
                    writer.close()
                        
                    o_high_res_scaled = np.expand_dims(o_high_res_scaled, axis = 1)
                    amp_estimate = np.absolute(o_high_res_scaled)
                    phase_estimate = np.angle(o_high_res_scaled)
                    amp_estimate = utils.unnormalize_img(amp_estimate, amp_range)
                    phase_estimate = utils.unnormalize_img(phase_estimate, phase_range)
                    writer.add_images('Amplitude', amp_estimate, global_step = i + 1)
                    writer.add_images('Phase', phase_estimate, global_step = i + 1)
                    writer.close()

                if recover_pupil:
                    coef_list.append(utils.var_to_np(coef, RUN_ON_GPU))

                if freeze_bn == 1:
                    if (initialize_net > 0):
                        net.eval()
                
                if utils.l2_norm_squared(o_high_res_new - o_high_res) < err_tol or epoch_loss < 1e-10:
                    o_high_res = o_high_res_new.copy()
                    break
                o_high_res = o_high_res_new.copy()

            elif (i % check_period == 0) or i == (epoch_num - 1):
                o_high_res_new = get_sequence_from_network(net, z_fixed, RUN_ON_GPU = RUN_ON_GPU)
                
                if o_GT is not None:
                    rsnr, o_high_res_scaled = utils.rsnr_for_dynamic(o_GT, o_high_res_new)
                    rsnr_track.append(rsnr)
                    if rsnr > best_rsnr:
                        best_rsnr = rsnr
                        torch.save(net.state_dict(), weights_name_rsnr)
                        
                    writer.add_scalar('RSNR', rsnr, global_step = i + 1)
                    writer.close()
                    
                    o_high_res_scaled = np.expand_dims(o_high_res_scaled, axis = 1)
                    amp_estimate = np.absolute(o_high_res_scaled)
                    phase_estimate = np.angle(o_high_res_scaled)
                    amp_estimate = utils.unnormalize_img(amp_estimate, amp_range)
                    phase_estimate = utils.unnormalize_img(phase_estimate, phase_range)
                    writer.add_images('Amplitude', amp_estimate, global_step = i + 1)
                    writer.add_images('Phase', phase_estimate, global_step = i + 1)
                    writer.close()

                if recover_pupil:
                    coef_list.append(utils.var_to_np(coef, RUN_ON_GPU))

                if freeze_bn == 1:
                    if (initialize_net > 0):
                        net.eval()
                
                if utils.l2_norm_squared(o_high_res_new - o_high_res) < err_tol or epoch_loss < 1e-10:
                    o_high_res = o_high_res_new.copy()
                    break
                o_high_res = o_high_res_new.copy()         
                
    if do_bestRSNR and o_GT is not None:
        weights_name_sol = weights_name_rsnr
    else:
        weights_name_sol = weights_name
    if RUN_ON_GPU:
        net.load_state_dict(torch.load(weights_name_sol))
    else:
        net.load_state_dict(torch.load(weights_name_sol, map_location = lambda storage, loc: storage))
    
    with torch.no_grad():
        o_high_res = get_sequence_from_network(net, z_fixed, RUN_ON_GPU = RUN_ON_GPU)
    
    if len(coef_list) > 0:
        return o_high_res, o_high_res_initial, np.array(loss_track), loss_track_initial, rsnr_track, np.stack(coef_list)
    else:
        return o_high_res, o_high_res_initial, np.array(loss_track), loss_track_initial, rsnr_track, None
    

def get_sequence_from_network(net, z_fixed, RUN_ON_GPU = False):
    net.eval()

    z_tensor = utils.np_to_var(z_fixed, RUN_ON_GPU)
    amp_tensor = []
    phase_tensor = []
    for i in range(z_tensor.shape[1]):
        amp, phase = net.forward(z_tensor[:, [i]])
        amp_tensor.append(amp)
        phase_tensor.append(phase)
    
    amp = torch.cat(amp_tensor, dim = 0)
    phase = torch.cat(phase_tensor, dim = 0)

    amp = amp[:, 0, :, :]
    phase = phase[:, 0, :, :]
                
    o_complex = amp * torch.exp(1j * phase)   
    o_high_res = utils.var_to_np(o_complex, RUN_ON_GPU)
    
    net.train()
    return o_high_res