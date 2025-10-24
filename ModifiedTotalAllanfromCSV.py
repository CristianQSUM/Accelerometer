#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:41:22 2025

@author: Cristian Ramirez Rodriguez
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import allantoolkit as atk
import allantoolkit.devs as devs

def se2dcsvfilestoarrays(filename):
  samplerate_df = pd.read_csv(filename, usecols=[" Rate (S/s)"], nrows=1, dtype={" Rate (S/s)":"int"})
  true_rate = float(samplerate_df[" Rate (S/s)"].iat[0])

  sequence_number_list = []
  array_ch_0_list = []
  array_ch_1_list = []

  chunksize = 10000  # You can adjust the chunk size as needed
  for chunk in pd.read_csv(filename, usecols=["X", "Channel 0", "Channel 1"], skiprows=[1], chunksize=chunksize):
    sequence_number_list.append(chunk["X"].values.astype(np.float64))
    array_ch_0_list.append(chunk["Channel 0"].values.astype(np.float64))
    array_ch_1_list.append(chunk["Channel 1"].values.astype(np.float64))

  sequence_number = np.concatenate(sequence_number_list)
  array_ch_0 = np.concatenate(array_ch_0_list)[:-1]
  array_ch_1 = np.concatenate(array_ch_1_list)[:-1]

  timelist = sequence_number[:-1] / true_rate
  halfsumarray = .5*(array_ch_0 + array_ch_1)
  halfdifferencearray = .5*(array_ch_0 - array_ch_1)
  return timelist, array_ch_0, array_ch_1, halfsumarray, halfdifferencearray, true_rate


def mtotdevtocsv(data, rate, outname="mtotdev_results.csv",
                 data_type='phase', taus='decade', max_af=None, alpha=None):
    """
    Compute modified total deviation (MTOTDEV), save to CSV,
    and returns result as panda df.

    Parameters
    data : ndarray Input data array (phase or frequency).
    rate : float Sampling rate in Hz.
    outname : str, optional Output CSV filename.
    data_type : str, 'phase' or 'freq', default phase
    taus : str, float, list, or ndarray Averaging times or keyword ('decade', 'octave', e'all')
    max_af : int, optional Maximum averaging factor.
    alpha : int, optional Optional alpha override.

    Returns
    df: Pandas df
    """

    res = devs.mtotdev(data, rate, data_type, taus, max_af, alpha)

    df = pd.DataFrame({
        'tau_s': res.taus,
        'devs_lo': res.devs_lo,
        'devs': res.devs,
        'devs_hi': res.devs_hi,
        'alpha': res.alphas,
        'n_points': res.ns,
        'avg_factor': res.afs
    })

    df.to_csv(outname, index=False)
    print(f"MTOTDEV results saved to: {os.path.abspath(outname)}")
    return df


def plottingmtotdev(filename, outname="mtotdev_results.csv", fig_name="mtotdev_plot.png"):
    timelist, array_ch_0, array_ch_1, halfsumarray, halfdifferencearray, true_rate = se2dcsvfilestoarrays(filename)

    df = mtotdevtocsv(halfdifferencearray, true_rate, outname=outname)

    tau = df['tau_s']
    devs = df['devs']
    alphas = df['alpha']

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    
    plt.suptitle('Half Difference Allan Deviation')

    axs[0].loglog(tau, devs, 'k-o', label='MTOTDEV')
    axs[0].set_xlabel(r'$\tau$ (s)')
    axs[0].set_ylabel('Modified Total Deviation')
    axs[0].set_title('Modified Total Deviation vs Averaging Time')
    axs[0].grid(True, which='both', ls='--', lw=0.5)
    axs[0].legend()

    axs[1].semilogx(tau, alphas, 'r-^', label=r'$\alpha$')
    axs[1].set_xlabel(r'$\tau$ (s)')
    axs[1].set_ylabel(r'$\alpha$ (slope)')
    axs[1].set_title('Noise Type Slope (α) vs Averaging Time')
    axs[1].grid(True, which='both', ls='--', lw=0.5)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    print(f"Figure saved to: {os.path.abspath(fig_name)}")
    plt.show()
    return df
