#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:45:42 2020

@author: Matt Kriete
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import colradpy


#%% Specify element to analyze and path to ACD and SCD adf11 files
element_name = 'carbon'
element_symbol = 'C'
num_charge_states = 7
year = '96'
metastable_resolved = True

if metastable_resolved:
    r = 'r'
else:
    r = ''
files = np.array([
    f'Atomic data/scd{year}{r}_{element_symbol.lower()}.dat',
    f'Atomic data/acd{year}{r}_{element_symbol.lower()}.dat',
    f'Atomic data/qcd{year}{r}_{element_symbol.lower()}.dat',
    f'Atomic data/xcd{year}{r}_{element_symbol.lower()}.dat',
    # f'Atomic data/ccd{year}{r}_{element_symbol.lower()}.dat',
])


#%% Run time-independent (steady state) ionization balance

# Set up ionization balance
ion = colradpy.ionization_balance(
    files, # a text array of the files used for the ionization balance
    temp_grid=np.geomspace(2.5e-1, 1e4, num=500), # Electron temperature in eV
    dens_grid=np.array([1e13]), # Electron density in cm^-3
    htemp_grid=np.array([5]),  # Neutral hydrogen temperature in eV
    hdens_grid=np.array([5e11]),  # Neutral hydrogen density in cm^-3
    use_cx=False,
    input_files_type="adf11",
)

# Specify the transport and relative source for each charge state
ne_tau = np.inf  # cm^-3 s
# ne_tau = 1e11  # cm^-3 s
num_metas = sum(ion.data["input_file"]["scd"]["metas"])
source = np.zeros((num_metas, len(ion.data["user"]["temp_grid"]), len(ion.data["user"]["dens_grid"])))
source[0] = 1  # Pure neutral source: models transport from lower Te region
# source[-1] = 1  # Pure fully ionized source: models transport from higher Te region

# Take the rates interpolated from the adf11 file and put then in the ionization balance matrix
ion.populate_ion_matrix(ne_tau, source)

# Solve the ionization balance matrix
ion.solve_time_independent()


#%% Plot steady-state fractional abundances
def plot_fractional_abundance(ne_index, TH_index=None, nH_index=None):
    title = (
        f"Steady-state fractional abundance for {element_name}" + "\n"
        fr"$n_\mathrm{{e}}$ = {ion.data['user']['dens_grid'][ne_index]:5.1e} $\mathrm{{cm}}^{{-3}}$  "
        fr"$\tau$ = {ne_tau / ion.data['user']['dens_grid'][ne_index]:5.1e} s  "
    )
    if np.any(ion.data["user"]["use_cx"]):
        title += (
            "\n"
            fr"$T_\mathrm{{H}}$ = {ion.data['user']['htemp_grid'][TH_index]:.1f} eV  "
            fr"$n_\mathrm{{H}}$ = {ion.data['user']['hdens_grid'][nH_index]:5.1e} $\mathrm{{cm}}^{{-3}}$  "
        )

    fig, ax = plt.subplots(constrained_layout=True)

    metas = ion.data["input_file"]["scd"]["metas"]  # number of metastables for each charge state
    m = 0  # index of the metastable state corresponding to the ground state of the current charge state
    colors = plt.get_cmap("tab10").colors  # Will need to expand this for Ne & beyond
    linestyles = ["--", ":", "-.", (5, (10, 3))]  # Will need to expand this for atoms with more than 4 metastables per charge state
    linestyles = linestyles[:np.max(metas)]  # Reduce number of linestyles down to just what is needed

    for charge_state in range(num_charge_states):  # loop over all charge states
        for meta in range(metas[charge_state]):
            ax.plot(
                ion.data['user']['temp_grid'],
                ion.data['processed']['pops_ss'][m + meta, :, ne_index],
                color=colors[charge_state],
                linestyle=linestyles[meta],
            )
        if charge_state == 0:
            element_label = f'$\mathrm{{{element_symbol}}}^{{{charge_state}}}$'
        else:
            element_label = f'$\mathrm{{{element_symbol}}}^{{{charge_state}\plus}}$'
        ax.plot(
            ion.data['user']['temp_grid'],
            np.sum(ion.data['processed']['pops_ss'][m:m + metas[charge_state], :, ne_index], axis=0),
            color=colors[charge_state],
            linestyle="-",
            label=element_label,
        )
        m += metas[charge_state]  # Increment to get metastable index for ground of next charge state

    leg1 = ax.legend(title="charge state", loc="lower right", bbox_to_anchor=(0.5, 0.44, 0.49, 0.55))
    ax.add_artist(leg1)
    leg1.set_draggable(True)
    if metastable_resolved:
        handles = []
        for i, linestyle in enumerate(linestyles):
            if i == 0:
                label = "ground"
            else:
                label = str(i)
            handles.append(mlines.Line2D([], [], linestyle=linestyle, color="black", label=label))
        handles.append(mlines.Line2D([], [], linestyle="-", color="black", label="total"))
        leg2 = ax.legend(handles=handles, title="metastable", loc="upper right", bbox_to_anchor=(0.5, 0, 0.49, 0.44))
        leg2.set_draggable(True)

    ax.set_xlim([ion.data['user']['temp_grid'].min(), ion.data['user']['temp_grid'].max()])
    ax.set_xscale('log')
    ax.set_ylim([0, 1])
    ax.set_xlabel("Electron temperature (eV)")
    ax.set_ylabel("Fractional abundance")
    ax.set_title(title)
    
    # filename = f"fractional_abundance_{element_symbol.lower()}_ss_ne={ion.data['user']['dens_grid'][ne_index]:.2e}_ntau={ne_tau:.2e}"
    # fig.savefig(os.path.join(os.getcwd(), "Results", filename + ".png"), dpi=300)

for ne_index in range(len(ion.data["user"]["dens_grid"])):
    plot_fractional_abundance(ne_index)


#%% Save results
# fractional_abundance = xr.DataArray(
#     ion.data['processed']['pops_ss'],
#     dims=('charge_state','Te','ne'),
#     coords={'charge_state':np.arange(ion.data['input_file']['acd']['nuc_charge']+1),
#             'Te':temp_grid,
#             'ne':dens_grid*1e6})
# fractional_abundance.to_netcdf(
#     f'Atomic data/fractional_abundance_{element_symbol.lower()}.nc')
