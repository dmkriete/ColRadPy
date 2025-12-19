#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 22:21:59 2024

@author: Matt Kriete

The radiative loss function gives the radiated power due to the presence of an
ion species. It is defined by L_z = P_z / (n_e * n_z), where P_z is the
radiated power due to the presence of ions of species z, n_e is electron
density, and n_z is the ion species density.

ADAS ADF11 files PLT and PRB contain data on radiation losses. The PLT file has
radiation loss rates due to excitation-driven line emission, while the PRB
file has radiation loss rates due to recombination-driven line & continuum
emission plus bremsstrahlung. The total radiative loss rate is the sum of
these.

ADAS radiative loss rates are given for each ion charge state (or metastable
state for resolved files). To calculate the total radiative loss function for a
particular ion species, fractional abundance calculations must be performed for
each density/temperature case of interest. The total L_z is then the sum of the
L_z for each charge/metastable state, weighted by the fractional abundance.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import colradpy

#%% Specify element to analyze and path to PLT, PRB, ACD and SCD adf11 files
element_name = 'carbon'
element_symbol = 'C'
num_charge_states = 7
year = '96'
data_dir = 'Atomic data'  # Directory where ADF11 files are stored
prb_file = os.path.join(data_dir, f'prb{year}_{element_symbol.lower()}.dat')
plt_file = os.path.join(data_dir, f'plt{year}_{element_symbol.lower()}.dat')
fa_files = np.array([
    os.path.join(data_dir, f'scd{year}_{element_symbol.lower()}.dat'),
    os.path.join(data_dir, f'acd{year}_{element_symbol.lower()}.dat'),
    os.path.join(data_dir, f'ccd{year}_{element_symbol.lower()}.dat'),
])

#%% Run time-independent (steady state) fractional abundance calculation
# Set up calculation
ion = colradpy.ionization_balance(
    fa_files, # a text array of the files used for the ionization balance
    temp_grid=np.geomspace(5e-1, 10e3, num=500), # Electron temperature in eV
    dens_grid=np.array([1e14]), # Electron density in cm^-3
    htemp_grid=np.array([5]),  # Neutral hydrogen temperature in eV
    hdens_grid=np.array([5e11]),  # Neutral hydrogen density in cm^-3
    use_cx=False,
    input_files_type="adf11",
)

# Specify the transport and relative source for each charge state
ne_tau = np.inf  # cm^-3 s, no transport
# ne_tau = 1e11  # cm^-3 s, moderate transport
# ne_tau = 1e9  # cm^-3 s, strong transport
source = np.zeros((num_charge_states, len(ion.data["user"]["temp_grid"]), len(ion.data["user"]["dens_grid"])))
source[0] = 1  # Pure neutral source -> transport from lower Te region
# source[-1] = 1  # Pure fully ionized source -> transport from higher Te region

# Take the rates interpolated from the adf11 file and put then in the ionization balance matrix
ion.populate_ion_matrix(ne_tau, source)

# Solve the ionization balance matrix
ion.solve_time_independent()

#%% Calculate radiative cooling curve
prb_file = colradpy.read_adf11(prb_file)
plt_file = colradpy.read_adf11(plt_file)
nuc_charge = prb_file["input_file"]["nuc_charge"]
num_temp = len(ion.data['user']['temp_grid'])
num_dens = len(ion.data['user']['dens_grid'])
prb_rates = np.zeros((nuc_charge, num_temp, num_dens))
plt_rates = np.zeros((nuc_charge, num_temp, num_dens))
for i in range(nuc_charge):
    prb_rates[i] = colradpy.ionization_balance_class.interp_rates_adf11(
        prb_file["input_file"]["temp_grid"],
        prb_file["input_file"]["dens_grid"],
        ion.data['user']['temp_grid'],
        ion.data['user']['dens_grid'], 
        prb_file["input_file"][str(i)]
    )[0, 0, :, :]  # Discard metastable info since we're using unresolved files
    plt_rates[i] = colradpy.ionization_balance_class.interp_rates_adf11(
        plt_file["input_file"]["temp_grid"],
        plt_file["input_file"]["dens_grid"],
        ion.data['user']['temp_grid'],
        ion.data['user']['dens_grid'], 
        plt_file["input_file"][str(i)]
    )[0, 0, :, :]  # Discard metastable info since we're using unresolved files
rlf_prb = np.zeros((num_charge_states, num_temp, num_dens))
rlf_plt = np.zeros((num_charge_states, num_temp, num_dens))
rlf_prb[1:] = prb_rates * ion.data['processed']['pops_ss'][1:]  # Neutral stage does not contribute to recombination-driven radiation and bremsstrahlung
rlf_plt[:-1] = plt_rates * ion.data['processed']['pops_ss'][:-1]  # Fully ionized stage does not contribute to excitation-driven line radiation
rlf_total = rlf_prb + rlf_plt

#%% Plot results
def plot_radiative_loss_function(ne_index, TH_index=None, nH_index=None):
    title = (
        f"Radiative loss function for {element_name}" + "\n"
        fr"$n_\mathrm{{e}}$ = {ion.data['user']['dens_grid'][ne_index]:.2e} $\mathrm{{cm}}^{{-3}}$  "
        fr"$\tau$ = {ne_tau / ion.data['user']['dens_grid'][ne_index]:.2e} s  "
    )
    if np.any(ion.data["user"]["use_cx"]):
        title += (
            "\n"
            fr"$T_\mathrm{{H}}$ = {ion.data['user']['htemp_grid'][TH_index]:.1f} eV  "
            fr"$n_\mathrm{{H}}$ = {ion.data['user']['hdens_grid'][nH_index]:.2e} $\mathrm{{cm}}^{{-3}}$  "
        )

    fig, ax = plt.subplots(constrained_layout=True)
    for charge_state in range(num_charge_states): # loop over all charge states
        if charge_state == 0:
            element_label = f'$\mathrm{{{element_symbol}}}^{{{charge_state}}}$'
        else:
            element_label = f'$\mathrm{{{element_symbol}}}^{{{charge_state}\plus}}$'
        line, = ax.plot(
            ion.data['user']['temp_grid'],
            rlf_total[charge_state, :, ne_index],
            linestyle="-",
            label=element_label,
        )
        ax.plot(
            ion.data['user']['temp_grid'],
            rlf_plt[charge_state, :, ne_index],
            linestyle="--",
            color=line.get_color(),
        )
        ax.plot(
            ion.data['user']['temp_grid'],
            rlf_prb[charge_state, :, ne_index],
            linestyle=":",
            color=line.get_color(),
        )
    ax.plot(
        ion.data['user']['temp_grid'],
        rlf_total[:, :, ne_index].sum(axis=0),
        linewidth=3,
        linestyle="-",
        color="black",
        label="total",
        zorder=0,
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1, 10e3])
    ax.set_ylim([1e-29, 1e-25])
    ax.set_xlabel("Electron temperature (eV)")
    ax.set_ylabel("Radiative loss function (W $\mathrm{cm}^3$)")
    ax.set_title(title)

    leg1 = ax.legend(title="charge state", loc="upper right")
    ax.add_artist(leg1)
    leg1.set_draggable(True)
    dashed_line = mlines.Line2D([], [], linestyle="--", color="black", label="plt")
    dotted_line = mlines.Line2D([], [], linestyle=":", color="black", label="prb")
    solid_line = mlines.Line2D([], [], linestyle="-", color="black", label="total")
    leg2 = ax.legend(handles=[dashed_line, dotted_line, solid_line], title="processes", loc="upper right", bbox_to_anchor=(0, 0, 0.80, 1))
    leg2.set_draggable(True)

for ne_index in range(len(ion.data["user"]["dens_grid"])):
    plot_radiative_loss_function(ne_index)
