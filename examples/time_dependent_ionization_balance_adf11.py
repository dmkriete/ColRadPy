#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:45:42 2020

@author: Matt Kriete
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import colradpy


#%% Specify element to analyze and path to ACD and SCD adf11 files
element_name = 'carbon'
element_symbol = 'C'
num_charge_states = 7
year = '96'
metastable_resolved = True
atomic_data_path = ''


#%% Build list of atomic data file paths
if metastable_resolved:
    r = 'r'
else:
    r = ''
files = [
    os.path.join(atomic_data_path, f'scd{year}{r}_{element_symbol.lower()}.dat'),
    os.path.join(atomic_data_path, f'acd{year}{r}_{element_symbol.lower()}.dat'),
    # os.path.join(atomic_data_path, f'ccd{year}{r}_{element_symbol.lower()}.dat'),
]
if metastable_resolved:
    files.extend([
        os.path.join(atomic_data_path, f'qcd{year}{r}_{element_symbol.lower()}.dat'),
        os.path.join(atomic_data_path, f'xcd{year}{r}_{element_symbol.lower()}.dat'),
    ])


#%% Run time-dependent ionization balance

# Set up ionization balance
ion_td = colradpy.ionization_balance(
    files,
    temp_grid=np.array([10]),  # Electron temperature in eV
    dens_grid=np.array([1e13]),  # Electron density in cm^-3
    htemp_grid=np.array([5]),  # Neutral hydrogen temperature in eV
    hdens_grid=np.array([5e11]),  # Neutral hydrogen density in cm^-3
    use_cx=False,
    input_files_type="adf11",
)

# Specify the transport and source (need to be balanced for total number of 
# particles to be conserved)
ne_tau = np.inf  # cm^-3 s
# ne_tau = 1e11  # cm^-3 s
num_metas = sum(ion_td.data["input_file"]["scd"]["metas"])
source = np.zeros((num_metas, len(ion_td.data["user"]["temp_grid"]), len(ion_td.data["user"]["dens_grid"])))
source[0] = 1  # Pure neutral source: models transport from lower Te region
# source[-1] = 1  # Pure fully ionized source: models transport from higher Te region

# Set up ionization matrix (with transport if desired)
# TODO: fix bugs with time-dependent calculation when transport/source is included
# ion_td.populate_ion_matrix(ne_tau, source)
ion_td.populate_ion_matrix()

# Solve ionization balance equations
time = np.geomspace(1e-8, 1e0, 1000)
initial_abundances = np.zeros(num_metas)
initial_abundances[0] = 1 # 100% of the population in the neutral charge state
ion_td.solve_time_dependent(time, initial_abundances)


#%% Plot time-dependent fractional abundances
def plot_fractional_abundance(Te_index, ne_index, TH_index=None, nH_index=None):
    title = (
        f"Fractional abundances for {element_name}" + "\n"
        fr"$T_\mathrm{{e}}$ = {ion_td.data['user']['temp_grid'][Te_index]:.1f} eV  "
        fr"$n_\mathrm{{e}}$ = {ion_td.data['user']['dens_grid'][ne_index]:5.1e} $\mathrm{{cm}}^{{-3}}$  "
        fr"$\tau$ = {ne_tau / ion_td.data['user']['dens_grid'][ne_index]:5.1e} s  "
    )
    if np.any(ion_td.data["user"]["use_cx"]):
        title += (
            "\n"
            fr"$T_\mathrm{{H}}$ = {ion_td.data['user']['htemp_grid'][TH_index]:.1f} eV  "
            fr"$n_\mathrm{{H}}$ = {ion_td.data['user']['hdens_grid'][nH_index]:5.1e} $\mathrm{{cm}}^{{-3}}$  "
        )
    
    fig, ax = plt.subplots(constrained_layout=True)

    metas = ion_td.data["input_file"]["scd"]["metas"]  # number of metastables for each charge state
    m = 0  # index of the metastable state corresponding to the ground state of the current charge state
    colors = plt.get_cmap("tab10").colors  # Will need to expand this for Ne & beyond
    linestyles = ["--", ":", "-.", (5, (10, 3))]  # Will need to expand this for atoms with more than 4 metastables per charge state
    linestyles = linestyles[:np.max(metas)]  # Reduce number of linestyles down to just what is needed
    
    for charge_state in range(num_charge_states): # loop over all charge states
        for meta in range(metas[charge_state]):
            ax.plot(
                time,
                ion_td.data['processed']['pops_td'][m + meta, :, Te_index, ne_index],
                color=colors[charge_state],
                linestyle=linestyles[meta],
            )
        if charge_state == 0:
            element_label = f'$\mathrm{{{element_symbol}}}^{{{charge_state}}}$'
        else:
            element_label = f'$\mathrm{{{element_symbol}}}^{{{charge_state}\plus}}$'
        ax.plot(
            time,
            np.sum(ion_td.data['processed']['pops_td'][m:m + metas[charge_state], :, Te_index, ne_index], axis=0),
            color=colors[charge_state],
            linestyle="-",
            label=element_label,
        )
        m += metas[charge_state]  # Increment to get metastable index for ground of next charge state

    leg1 = ax.legend(title="charge state", loc="lower left", bbox_to_anchor=(0.01, 0.44, 0.99, 0.55))
    ax.add_artist(leg1)
    leg1.set_draggable(True)
    handles = []
    for i, linestyle in enumerate(linestyles):
        if i == 0:
            label = "ground"
        else:
            label = str(i)
        handles.append(mlines.Line2D([], [], linestyle=linestyle, color="black", label=label))
    handles.append(mlines.Line2D([], [], linestyle="-", color="black", label="total"))
    leg2 = ax.legend(handles=handles, title="metastable", loc="upper left", bbox_to_anchor=(0.01, 0, 0.99, 0.44))
    leg2.set_draggable(True)
    
    ax.set_xscale('log')
    ax.set_ylim([0, 1])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fractional abundance")
    ax.set_title(title)
    
    # filename = f"fractional_abundance_{element_symbol.lower()}_td_Te={ion_td.data['user']['temp_grid'][Te_index]:.1f}eV_ne={ion_td.data['user']['dens_grid'][ne_index]:.2e}_ntau={ne_tau:.2e}"
    # fig.savefig(os.path.join(os.getcwd(), "Results", filename + ".png"), dpi=300)

for Te_index in range(len(ion_td.data["user"]["temp_grid"])):
    for ne_index in range(len(ion_td.data["user"]["dens_grid"])):
        plot_fractional_abundance(Te_index, ne_index)
