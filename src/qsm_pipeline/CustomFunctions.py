# -*- coding: utf-8 -*-
"""
Created on Mon Dec 3 12:27:00 2018

@author: shahidm 

Copyright 2023 Population Health Sciences, German Center for Neurodegenerative Diseases (DZNE)
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0 
Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

"""

from __future__ import print_function
import numpy as np
import numpy.fft as ft
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

def hip(euler1, euler2):
    hip = euler1 * np.conj(euler2)
    return hip

def hip_phase(phase1, phase2):
    h = hip(np.exp(1.0j*phase1), np.exp(1.0j * phase2))
    return np.angle(h)

def get_comp_ro_kspace(pc_AP_magn_file, pc_PA_magn_file, pc_AP_phas_file, pc_PA_phas_file):
    
    magn = nib.load(pc_AP_magn_file)
    phas = nib.load(pc_AP_phas_file)
    comp_ro = magn.get_data() * np.exp(1.0j * phas.get_data())
    
    magn = nib.load(pc_PA_magn_file)
    phas = nib.load(pc_PA_phas_file)
    #print(comp_ro.shape)
    #print(magn.shape)
    tmp = magn.get_data() * np.exp(1.0j * phas.get_data())
    
    comp_ro = np.append(comp_ro[...,np.newaxis], tmp[...,np.newaxis], axis=-1)
    affine = magn.affine
    #print(comp_ro.shape)
    
    
    comp_ro = np.mean(comp_ro, axis=-2)
    #print(comp_ro.shape)
    return affine, comp_ro

def get_ka_kd_kspace(comp_ro):

    k_a = ft.fftshift(ft.fftn(comp_ro[...,0], axes=[0,1,2]), axes=[0,1,2])
    k_d = ft.fftshift(ft.fftn(comp_ro[...,1], axes=[0,1,2]), axes=[0,1,2])
    return k_a, k_d

def get_mean_ro_pe_kspace(shape,comp_ro):
    
    k_a,k_d = get_ka_kd_kspace(comp_ro)
    RPE=3
    bPFfilter = False
    
    if bPFfilter:
        k_a[:,:2*shape[1]//8+RPE,...] = 0.0
        k_d[:,6*shape[1]//8-RPE:,...] = 0.0
        
    k = k_a + k_d
    mean_ro_pe = ft.ifftn(ft.fftshift(k/2.0, axes=[0,1,2]), axes=[0,1,2])
    #print(shape)
    #print(k.shape)
    return k_a, k_d, mean_ro_pe
    
    
def kspace_combine(pc_AP_magn_file, pc_PA_magn_file, pc_AP_phas_file, pc_PA_phas_file):
    
    import nibabel as nib
    import numpy as np
    import numpy.fft as ft
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from qsm_pipeline.CustomFunctions import hip, get_comp_ro_kspace,get_mean_ro_pe_kspace
    import os
    
    output_files=[]
    
    affine, comp_ro = get_comp_ro_kspace(pc_AP_magn_file, pc_PA_magn_file, 
                                         pc_AP_phas_file, pc_PA_phas_file)
    
    
    shape = comp_ro.shape
    
    # print(np.ndim(comp_ro))
    #phas_diff = 0
    bPFfilter = False
    
    k_a, k_d, mean_ro_pe = get_mean_ro_pe_kspace(shape,comp_ro)
    k = k_a + k_d

    #nib.save(nib.Nifti1Image(np.abs(mean_ro_pe), affine = affine), path.join(sub_dir, 'interim','mean_magn.nii.gz'))
    #nib.save(nib.Nifti1Image(np.angle(mean_ro_pe), affine = affine), path.join(sub_dir, 'interim','mean_phas.nii.gz'))
    
    A = np.abs(k_a)
    D = np.abs(k_d)
    print(A.shape)
    A = np.sqrt(np.mean(A**2, axis=(0,2,3)))
    D = np.sqrt(np.mean(D**2, axis=(0,2,3)))
    
    P = 1
    f_a = A**P/(A**P+D**P)
    
    f_d = D**P/(A**P+D**P)
    
    plt.figure()
    plt.plot(f_a)
    plt.plot(f_d)
    #plt.show()
    A_D_png = os.path.join(os.getcwd(),'A_D.png')
    
    plt.savefig(A_D_png, dpi=300)
    del A, D
    
    output_files.append(os.path.abspath(A_D_png))
    
    
    k_filt  = np.rollaxis(k_a, axis=1, start=k_a.ndim) * f_a
    k_filt += np.rollaxis(k_d, axis=1, start=k_d.ndim) * f_d
    k_filt  = np.rollaxis(k_filt, axis=-1, start=1)
    
    mean_ro_pe_filt = ft.ifftn(ft.fftshift(k_filt, axes=[0,1,2]), axes=[0,1,2])
    
    mean_magn_filt_file = os.path.join(os.getcwd(),'mean_magn_filt%d.nii.gz'%(int(bPFfilter)))
    mean_phas_filt_file = os.path.join(os.getcwd(),'mean_phas_filt%d.nii.gz'%(int(bPFfilter)))
    
    nib.save(nib.Nifti1Image(np.abs(mean_ro_pe_filt), affine = affine), mean_magn_filt_file)
    nib.save(nib.Nifti1Image(np.angle(mean_ro_pe_filt), affine = affine), mean_phas_filt_file)
        
    f,a = plt.subplots(3,4,figsize=[16,8])
    t=-1
    sl = 110#62 #shape[2]//2
    x = shape[1]
    y = shape[2]
    x0 = 0#x//2#x//4
    x1 = x#3*x//4
    y0 = 0#y//4
    y1 = y#3*y//4
    
    a[0][0].imshow(np.rot90(np.abs(k_a[shape[0]//2,:,:,t])**0.2), interpolation='none')
    a[0][1].imshow(np.rot90(np.abs(k_d[shape[0]//2,:,:,t])**0.2), interpolation='none')
    a[0][2].imshow(np.rot90(np.abs(k[shape[0]//2,:,:,t])**0.2), interpolation='none')
    a[0][3].imshow(np.rot90(np.abs(k_filt[shape[0]//2,:,:,t])**0.2), interpolation='none')
    
    a[1][0].imshow(np.rot90(np.abs(comp_ro[sl,x0:x1,y0:y1,t,0])), interpolation='spline16', cmap='gray')
    a[1][1].imshow(np.rot90(np.abs(comp_ro[sl,x0:x1,y0:y1,t,1])), interpolation='spline16', cmap='gray')
    a[1][2].imshow(np.rot90(np.abs(mean_ro_pe[sl,x0:x1,y0:y1,t])), interpolation='spline16', cmap='gray')
    a[1][3].imshow(np.rot90(np.abs(mean_ro_pe_filt[sl,x0:x1,y0:y1,t])), interpolation='spline16', cmap='gray')
    
    a[2][0].imshow(np.rot90(np.angle(comp_ro[sl,x0:x1,y0:y1,t,0])), interpolation='spline16', cmap='gray')
    a[2][1].imshow(np.rot90(np.angle(comp_ro[sl,x0:x1,y0:y1,t,1])), interpolation='spline16', cmap='gray')
    a[2][2].imshow(np.rot90(np.angle(mean_ro_pe[sl,x0:x1,y0:y1,t])), interpolation='spline16', cmap='gray')
    a[2][3].imshow(np.rot90(np.angle(mean_ro_pe_filt[sl,x0:x1,y0:y1,t])), interpolation='spline16', cmap='gray')
    
    for ax in a.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    PF_recovery_sag_png = os.path.join(os.getcwd(),'PF_recovery_sag.png')
    
    plt.savefig(PF_recovery_sag_png, dpi=300, bbox_inches='tight')
    #plt.show()
    
    output_files.append(os.path.abspath(PF_recovery_sag_png))
    
    pe = np.linspace(-0.5,0.5,f_a.shape[0])
    plt.figure(figsize=(3.2,3))
    plt.plot(pe, 0.5*np.ones_like(f_a))
    plt.plot(pe, 0.5*np.ones_like(f_a), '--')
    plt.tight_layout()
    plt.xticks([-0.5,0,0.5])
    plt.xlim([-0.5,0.5])
    plt.ylim([0, 1])
    plt.ylabel('weight')
    plt.gca().set_xticklabels([r'$-k_{max}$', '0', r'$+k_{max}$'])
    plt.gca().set_aspect(float(y)/float(x))
    
    weights_uniform_png = os.path.join(os.getcwd(),'weights_uniform.png')
    plt.savefig(weights_uniform_png, dpi=300, bbox_inches='tight')
    plt.tight_layout()

    #plt.show()
    
    output_files.append(os.path.abspath(weights_uniform_png))
    
    pe = np.linspace(-0.5,0.5,f_a.shape[0])
    plt.figure(figsize=(3.2,3))
    
    plt.plot(f_a, pe)
    plt.plot(f_d, pe)
    plt.tight_layout()
    plt.yticks([-0.5,0,0.5])
    plt.ylim([-0.5,0.5])
    plt.xlim([0, 1])
    plt.xlabel('weight')
    
    plt.gca().set_yticklabels([r'$-k_{max}$', '0', r'$+k_{max}$'])
    weights_png=os.path.join(os.getcwd(),'weights.png')
    plt.savefig(weights_png, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    #plt.show()
    
    output_files.append(weights_png)
    
    pe = np.linspace(-0.5,0.5,f_a.shape[0])
    plt.figure(figsize=(3.2,3))
    
    plt.plot(pe, f_a)
    plt.plot(pe, f_d)
    plt.tight_layout()
    plt.xticks([-0.5,0,0.5])
    plt.xlim([-0.5,0.5])
    plt.ylim([0, 1])
    plt.ylabel('weight')
    
    plt.gca().set_xticklabels([r'$-k_{max}$', '0', r'$+k_{max}$'])
    plt.gca().set_aspect(float(y)/float(x))
    weights_png2 = os.path.join(os.getcwd(),'weights2.png')
    plt.savefig(weights_png2, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    #plt.show()
    output_files.append(weights_png2)
    
    """
    rmse_magn_filt = np.sqrt(np.mean(np.abs(mean_ro_pe_filt)**2, axis=-1))
    
    rms_magn_filt_file = os.path.join(os.getcwd(),'rms_magn_filt%d.nii.gz'%(int(bPFfilter)))
    nib.save(nib.Nifti1Image(rmse_magn_filt, affine = affine), rms_magn_filt_file)
    
    output_files.append(rms_magn_filt_file)
    
    rmse_magn = np.sqrt(np.mean(np.abs(mean_ro_pe)**2, axis=-1))
    rms_magn_file = os.path.join(os.getcwd(),'rms_magn%d.nii.gz'%(int(bPFfilter)))
    nib.save(nib.Nifti1Image(rmse_magn, affine = affine), rms_magn_file)
    
    output_files.append(rms_magn_file)
    
    bPFfilter = False
    
    mean_ro_pe_filt = nib.load(mean_magn_filt_file)
    
    affine = mean_ro_pe_filt.affine
    mean_ro_pe_filt = mean_ro_pe_filt.get_data() * np.exp(1.0j * nib.load(mean_phas_filt_file).get_data())
    pdiff_filt = np.angle(np.mean(hip(mean_ro_pe_filt[...,1:], mean_ro_pe_filt[...,:-1]), axis=-1))
    pdiff_phas_filt_file = os.path.join(os.getcwd(),'pdiff_phas_filt%d.nii.gz'%(int(bPFfilter)))
    nib.save(nib.Nifti1Image(pdiff_filt, affine = affine), pdiff_phas_filt_file)
    output_files.append(pdiff_phas_filt_file)
    
    
    gre_magn = nib.load(path.join(sub_dir, 'raw', '033-QSM.nii.gz'))
    gre_affine = gre_magn.affine
    gre_rms = np.sqrt(np.mean(gre_magn.get_data().astype(float)**2, axis=-1))
    nib.save(nib.Nifti1Image(gre_rms, gre_affine),path.join(sub_dir, 'interim', 'gre_rms.nii.gz'))
    
    raw_magn = nib.load(path.join(sub_dir, 'interim', 'mean_magn_raw.nii.gz'))
    raw_affine = raw_magn.affine
    raw_rms = np.sqrt(np.mean(raw_magn.get_data().astype(float)**2, axis=-1))
    nib.save(nib.Nifti1Image(raw_rms, raw_affine), path.join(sub_dir, 'interim', 'raw_rms.nii.gz'))
    """
    
    return mean_magn_filt_file, mean_phas_filt_file, output_files

def get_comp_phase_correction(applytopup_corrected_files):
    
    applytopup_real_ap = ''
    applytopup_real_pa = ''
    applytopup_imag_ap = ''
    applytopup_imag_pa = ''
    
    for ac in applytopup_corrected_files:
        if 'real_mc_AP' in ac: applytopup_real_ap = ac
        if 'real_mc_PA' in ac: applytopup_real_pa = ac
        if 'imag_mc_AP' in ac: applytopup_imag_ap = ac
        if 'imag_mc_PA' in ac: applytopup_imag_pa = ac
    
    applytopup_real = nib.load(applytopup_real_ap)
    affine = applytopup_real.affine
    
    applytopup_real = applytopup_real.get_data()
    applytopup_real = np.append(applytopup_real, nib.load(applytopup_real_pa).get_data(), axis=-1)
    
    applytopup_imag = nib.load(applytopup_imag_ap).get_data()
    applytopup_imag = np.append(applytopup_imag, nib.load(applytopup_imag_pa).get_data(), axis=-1)
    
    comp = applytopup_real + 1.0j * applytopup_imag
    return affine, comp

def get_comp_corr_phase_correction(comp):
    from scipy.ndimage.filters import gaussian_filter
    #average_comp = np.mean(comp, axis=-1)
    #average_comp == mean_tu == np.mean(comp, axis=-1)
    #print(comp.shape)
    comp_roll = np.rollaxis(comp, axis=comp.shape[-1], start=0)
    #print(comp_roll.shape)
    
    # phase difference
    #hip_roll = hip(average_comp, comp_roll)
    hip_roll = hip(np.mean(comp, axis=-1), comp_roll)
    #print(hip_roll.shape)
    hip_ = np.rollaxis(hip_roll, axis=0, start=comp.shape[-1]+1)
    #print(hip_.shape)
   
 
    gauss_sigma = np.array([2, 2, 2, 0, 0])
    hip_real = gaussian_filter(np.real(hip_), gauss_sigma)
    hip_imag = gaussian_filter(np.imag(hip_), gauss_sigma)
    
    # hip_smooth = hip_real + 1.0j * hip_imag
    # we don't need the magnitude
    hip_smooth = np.exp(1.0j * np.arctan2(hip_imag, hip_real))
    
    comp_corr = comp.copy() * hip_smooth
    return comp_corr

def get_comp_plots_phase_correction(comp,comp_corr):
    
    import matplotlib.pyplot as plt
    sl = comp.shape[2]/2
    #print('Generating plots...')
    f,a = plt.subplots(comp.shape[-1]+1, comp.shape[-2], figsize=(10,8))
    for te in range(comp.shape[-2]):
        for meas in range(comp.shape[-1]):
            a[meas][te].imshow(np.angle(comp[...,sl,te,meas]))
        a[-1][te].imshow(np.abs(np.mean(comp[...,sl,te,:], axis=-1)), clim=[0,3000], cmap='gray')
    
    for ax in a.flatten():
        ax.axis('off')
        comp1_png = os.path.join(os.getcwd(), 'comp1.png')
        plt.savefig(comp1_png, dpi=300)
        
    #print('Generating plots...')    
    f,a = plt.subplots(comp.shape[-1]+1, comp.shape[-2], figsize=(10,8))
    for te in range(comp.shape[-2]):
        for meas in range(comp.shape[-1]):
            a[meas][te].imshow(np.angle(comp_corr[...,sl,te,meas]))
        a[-1][te].imshow(np.abs(np.mean(comp_corr[...,sl,te,:], axis=-1)), clim=[0,3000], cmap='gray')
    for ax in a.flatten():
        ax.axis('off')
        comp2_png = os.path.join(os.getcwd(), 'comp2.png')
        plt.savefig(comp2_png,dpi=300)
    
    #print('Generating plots...')
    f,a = plt.subplots(comp.shape[-1], comp.shape[-2], figsize=(10,8))
    for meas in range(comp.shape[-1]):
        for te in range(comp.shape[-2]):
            a[meas][te].imshow(np.angle(comp_corr[...,sl,te,meas]*np.conj(comp[...,sl,te,meas])), clim=[-np.pi, np.pi])
        a[meas][te].axis('off')
        comp3_png = os.path.join(os.getcwd(), 'comp3.png')
        plt.savefig(comp3_png,dpi=300)
    
    return comp1_png,comp2_png, comp3_png

def phase_correction(applytopup_corrected):
    
    import nibabel as nib
    import numpy as np
    import os
    from qsm_pipeline.CustomFunctions import (get_comp_phase_correction,
                                              get_comp_corr_phase_correction)
        
    affine, comp = get_comp_phase_correction(applytopup_corrected)
    
    #fix for inaccuracy in phasecorr step: see Ruediger 21.1.2019 email
    #####
    N = comp.shape[-1]
    # CAUTION: Assumes that in the phase difference between AP and PA measurements,
    # there exists a linear gradient along the readout direction, which may be explained
    # by the fact that for AP the whole k-space trajectory is rotated by 180 deg
    # compared to PA (not only PE inversion).
    readout_pc = comp[...,N//2:]*np.conj(comp[...,:N//2])
    if N>2:
        # average along multiple measurements per PE direction (if available)
        readout_pc = np.mean(readout_pc, axis=-1)
    
    # Ahn and Cho approach to estimating linear phase slopw without unwrapping
    readout_pc = readout_pc[1:,:,:,:] * np.conj(readout_pc[:-1,:,:,:])
    readout_pc = np.mean(readout_pc)
    X = np.arange(comp.shape[0])-comp.shape[0]*0.5
    readout_pc = np.exp(0.5j*X*np.angle(readout_pc))
    
    # Ahn and Cho phase correction approach (not to remove N/2 ghost,
    # but to remove linear phase gradient between AP and PA measurement first
    # to avoid phase wraps in subsequent estimation of phase drifts caused by
    # subject motion or scanner heating, etc.
    comp = np.rollaxis(comp, axis=0, start=comp.ndim)
    comp[...,:N//2,:] *= readout_pc
    comp[...,N//2:,:] *= np.conj(readout_pc)
    comp = np.rollaxis(comp, axis=comp.ndim-1, start=0)
    #####
    
   
 
    #mean_tu = np.mean(comp, axis=-1)
    
    #nn# mean_magn_tu_file = os.path.join(os.getcwd(), 'mean_magn_tu.nii.gz')
    #nn# mean_phas_tu_file = os.path.join(os.getcwd(), 'mean_phas_tu.nii.gz')

    #nn# nib.save(nib.Nifti1Image(np.abs(np.mean(comp, axis=-1)), affine), mean_magn_tu_file)
    #nn# nib.save(nib.Nifti1Image(np.angle(np.mean(comp, axis=-1)), affine), mean_phas_tu_file)
    
    comp_corr = get_comp_corr_phase_correction(comp)
    
 
    #mean_pc = np.mean(comp_corr, axis=-1)
    
    #nn# mean_magn_pc_file = os.path.join(os.getcwd(), 'mean_magn_pc.nii.gz')
    #nn# mean_phas_pc_file = os.path.join(os.getcwd(), 'mean_phas_pc.nii.gz')
    
    #as mean_pc = np.mean(comp_corr, axis=-1)
    #nib.save(nib.Nifti1Image(np.abs(mean_pc), affine), mean_magn_pc_file)
    #nib.save(nib.Nifti1Image(np.angle(mean_pc), affine), mean_phas_pc_file)
    #nn# nib.save(nib.Nifti1Image(np.abs(np.mean(comp_corr, axis=-1)), affine), mean_magn_pc_file)
    #nn# nib.save(nib.Nifti1Image(np.angle(np.mean(comp_corr, axis=-1)), affine), mean_phas_pc_file)

    """
    get_comp_plots_phase_correction(comp,comp_corr)
    """
    del comp
    
    
    Nmeas = comp_corr.shape[-1]
    
    
    pc_AP_magn_file = os.path.join(os.getcwd(), 'pc_AP_magn.nii.gz')
    pc_PA_magn_file = os.path.join(os.getcwd(), 'pc_PA_magn.nii.gz')
    pc_AP_phas_file = os.path.join(os.getcwd(), 'pc_AP_phas.nii.gz')
    pc_PA_phas_file = os.path.join(os.getcwd(), 'pc_PA_phas.nii.gz')
    
    nib.save(nib.Nifti1Image(np.abs(comp_corr[...,:Nmeas//2]), affine), pc_AP_magn_file)
    nib.save(nib.Nifti1Image(np.abs(comp_corr[...,Nmeas//2:]), affine), pc_PA_magn_file)
    nib.save(nib.Nifti1Image(np.angle(comp_corr[...,:Nmeas//2]), affine), pc_AP_phas_file)
    nib.save(nib.Nifti1Image(np.angle(comp_corr[...,Nmeas//2:]), affine), pc_PA_phas_file)
    
    return os.path.abspath(pc_AP_magn_file), \
           os.path.abspath(pc_PA_magn_file), \
           os.path.abspath(pc_AP_phas_file), \
           os.path.abspath(pc_PA_phas_file)
           
    
    
def create_topup_datain_file(in_filename):
    """
    create acqparams file for topup
    """
    
    import os
    
    #in_filename is just for the workflow connection, not used here
    
    AP_string = "0 1 0 1\n"
    PA_string = "0 -1 0 1\n"
    out_filename = 'datain'
    
    with open(out_filename,'w') as ofile:
        ofile.write(AP_string)
        ofile.write(PA_string)
        
    return os.path.abspath(out_filename)


def comp_mag_phase(mag_ap, phase_ap, mag_pa, phase_pa):
    
    import nibabel as nib
    import numpy as np
    import os
    from qsm_pipeline.CustomFunctions import hip

    magn = nib.load(mag_ap)
    affine = magn.affine
    
    magn = magn.get_data()
    magn = np.append(magn, nib.load(mag_pa).get_data(), axis=-1)
    shape = magn.shape
    
    
    phas = nib.load(phase_ap).get_data()*np.pi/4096.0
    phas = np.append(phas, nib.load(phase_pa).get_data()*np.pi/4096.0, axis=-1)
    comp = magn * np.exp(1.0j * phas)
    
    tseg = 2
    Nmeas = 4
    newshape = np.array(shape)
    newshape[-1] /= tseg*Nmeas
    newshape=np.append(newshape,-1)
    comp = np.reshape(comp, newshape,order='F')
    newshape = comp.shape
    
    Nvols = newshape[-1]
    Ntes  = newshape[-2]
    
    rmste = np.sqrt(np.mean(np.abs(comp)**2, axis=-2))
    
    rmste_file = os.path.join(os.getcwd(), 'rmste_magn.nii.gz')
    rms_file   = os.path.join(os.getcwd(), 'rms_magn.nii.gz')
    
    nib.save(nib.Nifti1Image(rmste, affine), rmste_file)
    
    rms = np.sqrt(np.mean(rmste**2, axis=-1))
    nib.save(nib.Nifti1Image(rms, affine), rms_file)
    
    real_te_files=[]
    imag_te_files=[]

    for te in range(Ntes):
        real_te_file = os.path.join(os.getcwd(), 'real_te%d.nii.gz'%(te))
        imag_te_file = os.path.join(os.getcwd(), 'imag_te%d.nii.gz'%(te))
        nib.save(nib.Nifti1Image(np.real(comp[...,te,:]), affine), real_te_file)
        nib.save(nib.Nifti1Image(np.imag(comp[...,te,:]), affine), imag_te_file)
        real_te_files.append(real_te_file)
        imag_te_files.append(imag_te_file)
    
    #mean raw
    #nn# mean_raw = np.mean(comp, axis=-1)
    #nn# mean_magn_raw_file = os.path.join(os.getcwd(), 'mean_magn_raw.nii.gz')
    #nn# mean_phas_raw_file = os.path.join(os.getcwd(), 'mean_phas_raw.nii.gz')
    #nn# nib.save(nib.Nifti1Image(np.abs(mean_raw),   affine), mean_magn_raw_file)
    #nn# nib.save(nib.Nifti1Image(np.angle(mean_raw), affine), mean_phas_raw_file)
    
    #rms raw
    #nn# rms_raw = np.sqrt(np.mean(np.abs(mean_raw)**2, axis=-1))
    #nn# rms_raw_file = os.path.join(os.getcwd(), 'rms_magn_raw.nii.gz')
    #nn# nib.save(nib.Nifti1Image(rms_raw, affine = affine), rms_raw_file)
    
    #pdiff raw
    #nn# pdiff_raw = np.angle(np.mean(hip(comp[...,1:,:], comp[...,:-1,:]), axis=(-2,-1)))
    #nn# pdiff_raw_file = os.path.join(os.getcwd(), 'pdiff_phas_raw.nii.gz')
    #nn# nib.save(nib.Nifti1Image(pdiff_raw, affine = affine), pdiff_raw_file)
    
    #save comp and comp_reshaped to pass them to reshape
    #nn# comp_file=os.path.join(os.getcwd(), 'comp.nii.gz')
    #nn# comp_reshaped_file = os.path.join(os.getcwd(), 'comp_reshaped.nii.gz')
    #nn# nib.save(nib.Nifti1Image(comp, affine=affine), comp_file)
    #nn# nib.save(nib.Nifti1Image(comp_reshaped, affine=affine), comp_reshaped_file)

    return Ntes, Nvols, newshape, rmste_file, rms_file,real_te_files, imag_te_files

def get_plots_reshape_mc(Nvols,Ntes,reshape,tseg,magn_reshaped_mc,phas_reshaped_mc,rmste_mc,pdiffte_mc):
    
    newshape = reshape
    Nslc  = newshape[2]
    Nro = int(newshape[1])
    Npe = int(newshape[0])
    img_m = np.zeros([int(Nro*(Ntes+2+float(tseg-1)/tseg)), Npe*Nvols])
    img_p = np.zeros([int(Nro*(Ntes+2+float(tseg-1)/tseg)), Npe*Nvols])
    
    for vol in range(Nvols):
        for te in range(Ntes+2):
            if te<Ntes:
                off = Nro*np.remainder(vol,tseg)//tseg
                arr_m = magn_reshaped_mc[:,:,Nslc//2,te,vol].copy()
                arr_p = phas_reshaped_mc[:,:,Nslc//2,te,vol].copy()
            else:
                off = Nro*np.remainder(Nvols-1,tseg)//tseg
                if te==Ntes:
                    arr_m = rmste_mc[:,:,Nslc//2,vol].copy()
                    arr_p = pdiffte_mc[:,:,Nslc//2,vol].copy()
                else:
                    arr_m = np.abs(rmste_mc[:,:,Nslc//2,vol] - rmste_mc[:,:,Nslc//2,0])
                    arr_p = hip_phase(pdiffte_mc[:,:,Nslc//2,vol], pdiffte_mc[:,:,Nslc//2, 0])
                arr_m[:,-4:] = np.nan
                arr_p[:,-4:] = np.nan
            img_m[te*Nro+off:(te+1)*Nro+off,vol*Npe:(vol+1)*Npe] = np.rot90(arr_m)
            img_p[te*Nro+off:(te+1)*Nro+off,vol*Npe:(vol+1)*Npe] = np.rot90(arr_p)
    
    
    plt.figure(figsize=(Nvols*1.5, (Ntes+2+float(tseg-1)/tseg)*1.5))
    #plt.imshow(img_m, cmap='gray', clim=[0,3000])
    plt.axis('off')
    rmste_magn_mc_png = os.path.join(os.getcwd(), 'rmste_magn_mc.png')
    plt.savefig(rmste_magn_mc_png, dpi=300)
    
    plt.figure(figsize=(Nvols*1.5, (Ntes+2+float(tseg-1)/tseg)*1.5))
    #plt.imshow(img_p, cmap='gray', clim=[-np.pi, np.pi])
    plt.axis('off')
    pdiff_phas_mc_png = os.path.join(os.getcwd(), 'pdiff_phas_mc.png')
    plt.savefig(pdiff_phas_mc_png, dpi=300)
    

    
def do_reshape_mc(Ntes, Nvols, shape, mc_real_te, mc_imag_te):
    import os
    import nibabel as nib
    import numpy as np
        
    tseg=2

    #nn# comp=nib.load(comp_file).get_data()
    #nn# comp_reshaped=nib.load(comp_reshaped_file).get_data()
    
    
    
    reshape = np.array(shape)
    #reshape[-1] *= tseg
    #reshape[-2] /= tseg
    
    shape = np.array(shape)
    shape[-1]/=tseg #from shape
    shape[-2]*=tseg #from shape
    
    real_mc = np.zeros(shape) #from shape
    imag_mc = np.zeros(shape) #from shape
    
    magn_reshaped_mc = np.zeros(reshape)
    phas_reshaped_mc = np.zeros(reshape)
    
    newshape = shape.copy() #from shape
    newshape[-1] = -1
    newshape[-2] = tseg
    
    affine = nib.load(mc_real_te[0]).affine
    
    for te in range(Ntes):
        real = nib.load(mc_real_te[te]).get_data()
        imag = nib.load(mc_imag_te[te]).get_data()
        print('real shape')
        print(real.shape)
        print(imag.shape)
        real_mc[...,te*tseg:(te+1)*tseg,:] = np.reshape(real, newshape, order='F')
        imag_mc[...,te*tseg:(te+1)*tseg,:] = np.reshape(imag, newshape, order='F')
        magn_reshaped_mc[...,te,:] = np.sqrt(real**2 + imag**2)
        phas_reshaped_mc[...,te,:] = np.arctan2(imag, real)
    
    Nmeas = shape[-1] #from shape
    #print(shape)
    #print(real_mc.shape)
    #print(Nmeas)
    real_mc_AP = os.path.join(os.getcwd(), 'real_mc_AP.nii.gz') 
    real_mc_PA = os.path.join(os.getcwd(), 'real_mc_PA.nii.gz')
    imag_mc_AP = os.path.join(os.getcwd(), 'imag_mc_AP.nii.gz')
    imag_mc_PA = os.path.join(os.getcwd(), 'imag_mc_PA.nii.gz')
    nib.save(nib.Nifti1Image(real_mc[...,:Nmeas//2], affine), real_mc_AP)
    nib.save(nib.Nifti1Image(real_mc[...,Nmeas//2:], affine), real_mc_PA)
    nib.save(nib.Nifti1Image(imag_mc[...,:Nmeas//2], affine), imag_mc_AP)
    nib.save(nib.Nifti1Image(imag_mc[...,Nmeas//2:], affine), imag_mc_PA)

    rmste_mc = np.sqrt(np.mean(magn_reshaped_mc**2, axis=-2))
    #nn# rmste_magn_mc = os.path.join(os.getcwd(), 'rmste_magn_mc.nii.gz')
    #nn# nib.save(nib.Nifti1Image(rmste_mc, affine), rmste_magn_mc)
    
    # ms excl # comp_reshaped_mc = magn_reshaped_mc * np.exp(1.0j * phas_reshaped_mc)
    # ms excl # pdiffte_mc = np.angle(np.mean(hip(comp_reshaped_mc[...,1:,:], comp_reshaped_mc[...,:-1,:]), axis=-2))
    
    #nn# comp_mc = np.zeros_like(comp)
    #nn# print (comp_reshaped.shape)
    #nn# print(comp_mc.shape)
    
    #nn# for t in range(tseg):
    #nn#     comp_mc[...,t::tseg,:] = comp_reshaped[...,t::tseg]
    #nn# mean_mc = np.mean(comp_mc, axis=-1)
    #nn# mean_magn_mc = os.path.join(os.getcwd(), 'mean_magn_mc.nii.gz')
    #nn# mean_phas_mc = os.path.join(os.getcwd(), 'mean_phas_mc.nii.gz')
    #nn# nib.save(nib.Nifti1Image(np.abs(mean_mc), affine), mean_magn_mc)
    #nn# nib.save(nib.Nifti1Image(np.angle(mean_mc), affine), mean_phas_mc)
    
    #nn# rms_mc = np.sqrt(np.mean(rmste_mc**2, axis=-1))
    #nn# rms_magn_mc = os.path.join(os.getcwd(), 'rms_magn_mc.nii.gz')
    #nn# nib.save(nib.Nifti1Image(rms_mc, affine), rms_magn_mc)
    
    AP = np.sqrt(np.mean(rmste_mc[...,:Nvols//2]**2, axis=-1))
    PA = np.sqrt(np.mean(rmste_mc[...,Nvols//2-1:]**2, axis=-1))
    APPA_mc = np.append(AP[...,np.newaxis], PA[...,np.newaxis], axis=-1)
    appa_mc_file = os.path.join(os.getcwd(), 'APPA_mc.nii.gz')
    nib.save(nib.Nifti1Image(APPA_mc, affine), appa_mc_file)
    
    #do we need the plotting here?
    """
    get_plots_reshape_mc(Nvols,Ntes,reshape,tseg,magn_reshaped_mc,phas_reshaped_mc,rmste_mc,pdiffte_mc)

    """
    appa_mc_file = os.path.abspath(appa_mc_file)
    real_mc_AP = os.path.abspath(real_mc_AP)
    real_mc_PA = os.path.abspath(real_mc_PA)
    imag_mc_AP = os.path.abspath(imag_mc_AP)
    imag_mc_PA = os.path.abspath(imag_mc_PA) 
    
    return appa_mc_file, real_mc_AP,real_mc_PA,imag_mc_AP,imag_mc_PA

    
    
def moco_iterator(rms_magn, rmste_magn, it):
    import os
    import nibabel as nib
    import numpy as np
    from nipype.interfaces import fsl
    from nipype.utils.filemanip import copyfile 

    fn, ext = os.path.splitext(rms_magn)
    ref_file=fn.replace('.nii','')+'_mcflirtref0.nii'+ext

    
    copyfile(rms_magn,ref_file,False,use_hardlink=True)
    
    #out_files=[]
    ref_files=[]
    out_par_file="" #we only want to keep the last iteration
    out_rms_files=[] #we only want to keep the last iteration rms
    

    out_mat_dirs=[]
    #last out_mat_dir in (0,1,2) is 2 as in applyxfm4D code mcflirt_ref%d.nii.gz (%mcf_iterations-1) 
    #is used,and that means the last transform .mat dir is having no.2 in its name
    last_mat_dir=''

    out_file=''

    for i in range(it):
        
        out_file = 'rmste_magn_mcflirt%d.nii.gz'%(i)
    
        mcflirt=fsl.MCFLIRT()
        mcflirt.inputs.in_file = rmste_magn
        mcflirt.inputs.ref_file = ref_file #actually the input rms_magn the first time
        mcflirt.inputs.out_file = out_file
        mcflirt.inputs.save_rms = True
        mcflirt.inputs.save_plots = True
        mcflirt.inputs.save_mats=True
        mcflirt.inputs.interpolation='spline' #-spline_final
        mcflirt.terminal_output='file'
        
        print(mcflirt.cmdline)
        
        runtime=mcflirt.run()
        outputs = runtime.outputs.get()
        out_mcf_file = outputs['out_file']
        
        out_par_file = outputs['par_file']
        out_rms_file = outputs['rms_files']
        
        out_mat_dir = out_file+'.mat'
        last_mat_dir=os.path.abspath(out_mat_dir) #overwrite until last (2) 
        out_mcf_img = nib.load(out_mcf_file)
        affine=out_mcf_img.affine
        out_mcf_img = out_mcf_img.get_data()
        ref_file = 'mcflirt_ref%d.nii.gz'%(i+1)
        nib.save(nib.Nifti1Image(np.sqrt(np.mean(out_mcf_img**2, axis=-1)), affine), ref_file)
        
        #out_files.append(os.path.abspath(out_file))
        
        ref_files.append(os.path.abspath(ref_file))
        out_mat_dirs.append(os.path.abspath(out_mat_dir))
        
        out_file = out_mcf_file
        out_rms_files=[]
        for rms_file in out_rms_file:
            out_rms_files.append(rms_file)
        
    
    return out_file, ref_files, out_mat_dirs, last_mat_dir, out_par_file, out_rms_files


