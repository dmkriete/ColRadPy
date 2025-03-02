import numpy as np
from scipy.interpolate import RegularGridInterpolator
from colradpy.colradpy_class import colradpy
from colradpy.solve_matrix_exponential import (
    solve_matrix_exponential_source,
    solve_matrix_exponential_steady_state,
    solve_matrix_exponential_ss_new,
)
from colradpy.read_adf11 import read_adf11


def interp_rates_adf11(logged_temp, logged_dens, temp, dens, logged_gcr):
    interp = RegularGridInterpolator(
        points=(logged_temp, logged_dens),
        values=np.moveaxis(logged_gcr, (0, 1), (-2, -1)),
        method="cubic",
    )
    temp_grid, dens_grid = np.meshgrid(np.log10(temp), np.log10(dens), indexing='ij')
    gcr_interp = 10 ** interp((temp_grid, dens_grid))
    return np.moveaxis(gcr_interp, (-2, -1), (0, 1))


class ionization_balance():
    """
    The ionization balance class takes adf04 file inputs, runs CR calculations
    'colradpy_class.py' to get GCR coefficients, assembles them into the
    ionization balance matrix, and solves the matrix in the same way as the
    time dependent CR problem with 'solve_matrix_exponential.py'. The matrix
    can be solved time dependently with and without a source. The time-
    independent solution can also be solved. See dictionary structure
    documentation.
    """
    def __init__(
        self,
        files,
        temp_grid,
        dens_grid,
        htemp_grid=np.array([]),
        hdens_grid=np.array([]),
        temp_dens_pair=False,
        metas=np.array([]),
        use_ionization=True,
        supplement_with_ecip=True,
        use_recombination_three_body=True,
        use_recombination=True,
        use_cx=False,
        keep_charge_state_data=False,
        scale_file_ioniz=False,
        input_files_type="adf04",
    ):
        """
        Initializes an ionization balance for some atom species in a plasma.
    
        Parameters
        ----------
        files : list of str
            ADF04, ADF11, or HDF5 files containing atomic data for the modelled
            species.
        temp_grid : 1D float array
            Electron temperatures in electron-volts of the background plasma.
        dens_grid : 1D float array
            Electron densities in particles per cm^3 of the background plasma.
        htemp_grid : 1D float array, optional
            Temperatures in electron-volts of neutral hydrogen for thermal CX
            reactions. Must be aligned with `temp_grid`. Only used if `use_cx`
            is True.
        hdens_grid : 1D float array, optional
            Densities in particles per cm^3 of neutral hydrogen for thermal CX
            reactions. Must be aligned with `dens_grid`. Only used if `use_cx`
            is True.
        temp_dens_pair : bool, optional
            If True, then a 1D temperature/density grid formed from pairs of
            values from the temperature and density grids is used. This
            requires `temp_grid` and `dens_grid` to have the same shape.
            Otherwise, a 2D temperature/density grid is formed from the outer
            product of `temp_grid` and `dens_grid`. Defaults to a 2D grid.
        metas : int array, optional
            Indices of metastable levels for each charge state. Only needed if
            ADF04 data is used.
        use_ionization : bool or bool array, optional
            Choose if ionization will be used (only needed if ADF04 data is
            used). This will process the SCD rates. This should probably always
            be true if running an ionization balance, but the option is there.
            Can be array of bools or just a single bool. If just a single bool
            is supplied then all charge states will have the same value.
        supplement_with_ecip : bool or bool array, optional
            Supplements ionization rates with ECIP when ionization rates are
            not present in the adf04 file (only needed if ADF04 data is used).
            Can be array of bools or just a single bool. If just a single bool
            is supplied then all charge states will have the same value.
        use_recombination_three_body : bool or bool array, optional
            Adds in three body recombination (only needed if ADF04 data is
            used). There must be ionization present for this to work. Can be
            array of bools or just a single bool. If just a single bool is
            supplied then all charge states will have the same value.
        use_recombination : bool or bool array, optional
            Use recombination in the ionization balance (only needed if ADF04
            data is used). This will produce the ACD and possible XCD rates.
            Can be array of bools or just a single bool. If just a single bool
            is supplied then all charge states will have the same value.
        use_cx : bool or bool array, optional
            Use thermal charge exchange in the ionization balance. This will
            produce the CCD rates if ADF04 data is used and requires CCD input
            files if ADF11 data is used. Can be array of bools or just a single
            bool. If just a single bool is supplied then all charge states will
            have the same value.
        keep_charge_state_data : bool, optional
            If ADF04 data is used, controls whether to keep all of the data
            associated with the CR solutions for each charge state. This can
            potentially take up lots of memory, and only the GCR coefficients
            are used for an ionization balance, so it's kind of a waste to keep
            all the data. Defaults to False.
        scale_file_ioniz : bool, optional
            Scale ionization in the file. Defaults to False.
        input_files_type : {"adf04", "adf11", "hdf5"}, optional
            The type of atomic data input files to use. Defaults to "adf04".
        """
        # Basic input data that is the same for adf04 vs adf11 inputs
        self.data = {}
        self.data['cr_data'] = {}
        self.data['cr_data']['gcrs'] = {}
        self.data['user'] = {}
        self.data['user']['temp_grid'] = np.asarray(temp_grid) #eV
        self.data['user']['dens_grid'] = np.asarray(dens_grid) #cm^-3
        self.data['user']['htemp_grid'] = np.asarray(htemp_grid) #eV
        self.data['user']['hdens_grid'] = np.asarray(hdens_grid) # cm^-3
        self.data['user']['fils'] = np.asarray(files)

        # Collect input data that depends on input file format (adf04 vs adf11)
        if input_files_type == "adf11":
            self.data['input_file'] = {}            
            for file in files:
                self.data['user']['temp_dens_pair'] = False

                adf11 = read_adf11(file)

                if(type(use_cx) == bool):
                    self.data['user']['use_cx'] = np.ones(len(adf11['input_file']['metas'])-1, dtype=bool)
                    self.data['user']['use_cx'][:] = use_cx

                for j in range(len(adf11['input_file']['metas'])-1):
                    if( str(j) not in self.data['cr_data']['gcrs']):
                        self.data['cr_data']['gcrs'][str(j)] = {}

                    if 'scd' in file:
                        self.data['cr_data']['gcrs'][str(j)]['scd'] = interp_rates_adf11(
                            adf11['input_file']['temp_grid'],
                            adf11['input_file']['dens_grid'],
                            temp_grid,
                            dens_grid,
                            adf11['input_file'][str(j)],
                        )
                        self.data['input_file']['scd'] = adf11['input_file']
                        
                    if 'acd' in file:
                        self.data['cr_data']['gcrs'][str(j)]['acd'] = interp_rates_adf11(
                            adf11['input_file']['temp_grid'],
                            adf11['input_file']['dens_grid'],
                            temp_grid,
                            dens_grid,
                            adf11['input_file'][str(j)],
                        )
                        self.data['input_file']['acd'] = adf11['input_file']
                        
                    if 'qcd' in file:
                        # Interpolator cannot handle negative infinities along
                        # diagonal (caused by log of zero rate). Convert the
                        # diagonal to zero before and after interpolation as a
                        # workaround
                        n = adf11['input_file'][str(j)].shape[0]
                        adf11['input_file'][str(j)][np.diag_indices(n)] = 0
                        self.data['cr_data']['gcrs'][str(j)]['qcd'] = interp_rates_adf11(
                            adf11['input_file']['temp_grid'],
                            adf11['input_file']['dens_grid'],
                            temp_grid,
                            dens_grid,
                            adf11['input_file'][str(j)],
                        )
                        # Rates along diagonal get set to 1 due to
                        # exponentiation inside interpolator, so set back to 0
                        self.data['cr_data']['gcrs'][str(j)]['qcd'][np.diag_indices(n)] = 0
                        self.data['input_file']['qcd'] = adf11['input_file']
                        
                    if 'xcd' in file:
                        # Interpolator cannot handle negative infinities along
                        # diagonal (caused by log of zero rate). Convert the
                        # diagonal to zero before and after interpolation as a
                        # workaround
                        n = adf11['input_file'][str(j)].shape[0]
                        adf11['input_file'][str(j)][np.diag_indices(n)] = 0
                        self.data['cr_data']['gcrs'][str(j)]['xcd'] = interp_rates_adf11(
                            adf11['input_file']['temp_grid'],
                            adf11['input_file']['dens_grid'],
                            temp_grid,
                            dens_grid,
                            adf11['input_file'][str(j)],
                        )
                        # Rates along diagonal get set to 1 due to
                        # exponentiation inside interpolator, so set back to 0
                        self.data['cr_data']['gcrs'][str(j)]['xcd'][np.diag_indices(n)] = 0
                        self.data['input_file']['xcd'] = adf11['input_file']                               
                    if 'ccd' in file:
                        self.data['cr_data']['gcrs'][str(j)]['ccd'] = interp_rates_adf11(
                            adf11['input_file']['temp_grid'],
                            adf11['input_file']['dens_grid'],
                            htemp_grid,
                            hdens_grid,
                            adf11['input_file'][str(j)],
                        )
                        self.data['input_file']['ccd'] = adf11['input_file']                        
                        
                    if 'qcd' not in self.data['cr_data']['gcrs'][str(j)]:
                        # TODO: this throws an error if scd file is not first in list (same for xcd below)
                        self.data['cr_data']['gcrs'][str(j)]['qcd'] = np.zeros(
                            (
                                np.shape(self.data['cr_data']['gcrs'][str(j)]['scd'])[0],
                                np.shape(self.data['cr_data']['gcrs'][str(j)]['scd'])[0],
                                len(temp_grid),
                                len(dens_grid),
                            ),
                        )

                    if 'xcd' not in self.data['cr_data']['gcrs'][str(j)]:
                        self.data['cr_data']['gcrs'][str(j)]['xcd'] = np.zeros(
                            (
                                np.shape(self.data['cr_data']['gcrs'][str(j)]['scd'])[0],
                                np.shape(self.data['cr_data']['gcrs'][str(j)]['scd'])[0],
                                len(temp_grid),
                                len(dens_grid),
                            ),
                        )

        elif input_files_type == "hdf5":

            import h5py
            import hdfdict

            if(type(use_cx) == bool):
                self.data['user']['use_cx'] = np.ones_like(files, dtype=bool)
                self.data['user']['use_cx'][:] = use_cx

            
            self.data['user']['temp_dens_pair'] = False  # Probably should remove at some point
            self.data['cr_data']['stage_data'] = {}
            for i, file in enumerate(files):
                c_hdf5 = hdfdict.load(file, lazy=False)
                self.data['cr_data']['gcrs'][str(i)] = {}
                self.data['cr_data']['gcrs'][str(i)]['scd'] = np.copy(c_hdf5['processed']['scd'])
                self.data['cr_data']['gcrs'][str(i)]['acd'] = np.copy(c_hdf5['processed']['acd'])
                self.data['cr_data']['gcrs'][str(i)]['qcd'] = np.copy(c_hdf5['processed']['qcd'])
                self.data['cr_data']['gcrs'][str(i)]['xcd'] = np.copy(c_hdf5['processed']['xcd'])
                self.data['cr_data']['stage_data'][str(i)] = c_hdf5
                
        elif input_files_type == "adf04":
            
            # These user settings have ADF04-specific values
            self.data['user']['temp_dens_pair'] = temp_dens_pair
            self.data['user']['metas'] = np.asarray(metas)
            self.data['user']['keep_charge_state_data'] = keep_charge_state_data

            # Give the option to the user to choose different ionization and
            # recombination settings for each charge state. This also just
            # allows the same thing to be specified for all charge states.
            if type(use_ionization) == bool:
                self.data['user']['use_ionization'] = np.ones_like(files, dtype=bool)
                self.data['user']['use_ionization'][:] = use_ionization
            else:
                self.data['user']['use_ionization'] = use_ionization
            if type(supplement_with_ecip) == bool:
                self.data['user']['supplement_with_ecip'] = np.ones_like(files, dtype=bool)
                self.data['user']['supplement_with_ecip'][:] = supplement_with_ecip
            else:
                self.data['user']['supplement_with_ecip'] = supplement_with_ecip
            if type(use_recombination_three_body) == bool:
                self.data['user']['use_recombination_three_body'] = np.ones_like(files, dtype=bool)
                self.data['user']['use_recombination_three_body'][:] = use_recombination_three_body
            else:
                self.data['user']['use_recombination_three_body'] = use_recombination_three_body
            if type(use_recombination) == bool:
                self.data['user']['use_recombination'] = np.ones_like(files, dtype=bool)
                self.data['user']['use_recombination'][:] = use_recombination
            else:
                self.data['user']['use_recombination'] = use_recombination
            if type(use_cx) == bool:
                self.data['user']['use_cx'] = np.ones_like(files, dtype=bool)
                self.data['user']['use_cx'][:] = use_cx
            else:
                self.data['user']['use_cx'] = use_cx
            if type(use_ionization) == bool:
                self.data['user']['use_ionization'] = np.ones_like(files, dtype=bool)
                self.data['user']['use_ionization'][:] = use_ionization
            else:
                self.data['user']['use_ionization'] = use_ionization
            if type(scale_file_ioniz) == bool:
                self.data['user']['scale_file_ioniz'] = np.ones_like(files, dtype=bool)
                self.data['user']['scale_file_ioniz'][:] = scale_file_ioniz
            else:
                self.data['user']['scale_file_ioniz'] = scale_file_ioniz

            # Initialize structure to hold full CR data if requested
            if self.data['user']['keep_charge_state_data']:
                self.data['cr_data']['stage_data'] = {}

            # Cycle over all of the ion stages that the user chose and run the
            # CR calculation to get the GCR coefficients for that stage
            for i in range(len(files)):
                # Default is for user to just give metastables of the ground
                # state and then choose metastables just based off ionization
                # potentials in adf04 file the user can also give a list of
                # metastable states for every ionstage and this will override
                # the metastables in the adf04 files
                if type(metas) == list:
                    meta = metas[i]
                else:
                    if i == 0:
                        meta = metas
                    else:
                        m = np.shape(self.data['gcrs'][str(i -1)]['scd'])[1]
                        meta = np.linspace(0, m-1, m, dtype=int)

                # Setup the CR
                tmp = colradpy(
                    fil=str(files[i]),
                    metas=meta,
                    temp_grid=temp_grid,
                    electron_den=dens_grid,
                    htemp_grid=htemp_grid,
                    hdens_grid=hdens_grid,
                    use_ionization=self.data['user']['use_ionization'][i],
                    suppliment_with_ecip=self.data['user']['supplement_with_ecip'][i],
                    use_recombination_three_body=self.data['user']['use_recombination_three_body'][i],
                    use_recombination=self.data['user']['use_recombination'][i],
                    use_cx=self.data['user']['use_cx'][i],
                    temp_dens_pair=self.data['user']['temp_dens_pair'],
                )

                # Solve the CR equations
                tmp.solve_cr()

                # Keep all the CR data if requested
                if self.data['user']['keep_charge_state_data']:
                    self.data['cr_data']['stage_data'][str(i)] = tmp.data

                # Keep the GCR data, ionization can be run from this data only
                self.data['cr_data']['gcrs'][str(i)] = {}
                self.data['cr_data']['gcrs'][str(i)]['qcd'] = tmp.data['processed']['qcd']
                self.data['cr_data']['gcrs'][str(i)]['scd'] = tmp.data['processed']['scd']
                self.data['cr_data']['gcrs'][str(i)]['acd'] = tmp.data['processed']['acd']
                self.data['cr_data']['gcrs'][str(i)]['xcd'] = tmp.data['processed']['xcd']
                if self.data['user']['use_cx'][i]:
                    self.data['cr_data']['gcrs'][str(i)]['ccd'] = tmp.data['processed']['ccd']

        else:
            raise ValueError(
                f"Unknown input file type '{input_files_type}'. Must be "
                "'adf04', 'adf11', or 'hdf5'."
            )


    def populate_ion_matrix(self, ne_tau=np.inf, source=None):
        """
        Populates the ionization matrix with the various GCR coefficients.

        This matrix in mostly zeros because we don't have ways to connect
        charge states that are more than one change away. For example if only
        ground states are included this matrix becomes diagonal.

        QCDs bring the atom between metastables states in a charge state.
        SCDS bring the atom to the next charge state.
        ACDS bring the atom to the previous charge state.
        XCDS bring the atom to next charge state between the metastables of
        that level, through the previous charge state.

        The effects of transport on the ionization balance can be approximately
        modelled by including decay terms for each ionization state in the
        ionization matrix and a source of particles that balances this decay.
        Note that an alternative approach to modelling transport effects is to
        perform a time-dependent CRM calculation and truncate the result at the
        dwell time.

        Parameters
        ----------
        ne_tau : float array, optional
            Simulates the effect of transport on the ionization balance by
            adding a 1/tau term to each state's evolution equation. Tau is a
            characteristic timescale or "dwell time" for the transport. By
            convention, tau is scaled by the density, and this parameter should
            have units of cm^-3 s. The dwell time is assumed to be the same for
            each metastable state. Can be a scalar or an array aligned with
            the input (temperature, density) grid. By default, this parameter
            is infinity (i.e. no transport).
        source : float array, optional
            Source rate of each metastable state for use in transport
            modelling. The total source of all states is balanced against
            transport, so only the relative values matter. Can be a 1D array
            of length num_metastables, or a higher dimensionality array with
            the first axis of length num_metastables and the last axes aligned
            with the input (temperature, density) grid. This parameter is not
            used if ne_tau is infinity. Defaults to a 100% neutral ground state
            source, which is useful for modelling transport from regions of
            lower to higher electron temperature.
        """
        # Find the total number of states to be tracked (metastables)
        num_meta = np.shape(self.data['cr_data']['gcrs']['0']['qcd'])[0]  # Num metastables in neutral atom
        for i in range(len(self.data['cr_data']['gcrs'])):
            num_meta += np.shape(self.data['cr_data']['gcrs'][str(i)]['scd'])[1]  # Num metastables in each ionization stage

        # Create an empty matrix to hold the GCR rates
        if self.data['user']['temp_dens_pair']:
            self.data['ion_matrix'] = np.zeros((num_meta, num_meta, len(self.data['user']['temp_grid'])))
        else:
            self.data['ion_matrix'] = np.zeros(
                (num_meta, num_meta, len(self.data['user']['temp_grid']), len(self.data['user']['dens_grid']))
            )

        # Populate the ionization matrix
        m = 0  # Index of the lowest metastable state (i.e. ground state) for each ionization stage
        for i in range(len(self.data['cr_data']['gcrs'])):
            num_meta_lower = np.shape(self.data['cr_data']['gcrs'][str(i)]['qcd'])[0]
            num_meta_upper = np.shape(self.data['cr_data']['gcrs'][str(i)]['scd'])[1]
            diag_lower = np.diag_indices(num_meta_lower)
            diag_upper = np.diag_indices(num_meta_upper)

            # Consider pre-calculating m + diag_lower[0] since it gets used a lot?
            # Consider pre-calculating m + num_meta_lower since it gets used a lot?

            # Populate SCDs in ion balance
            self.data['ion_matrix'][m+diag_lower[0], m+diag_lower[1]] -= (
                np.sum(self.data['cr_data']['gcrs'][str(i)]['scd'], axis=1)
            )
            self.data['ion_matrix'][m+num_meta_lower:m+num_meta_lower+num_meta_upper, m:m+num_meta_lower] += (
                np.swapaxes(self.data['cr_data']['gcrs'][str(i)]['scd'], 0, 1)
            )

            # Populate ACDs in ion balance
            self.data['ion_matrix'][m+num_meta_lower+diag_upper[0], m+num_meta_lower+diag_upper[1]] -= (
                np.sum(self.data['cr_data']['gcrs'][str(i)]['acd'], axis=0)
            )
            self.data['ion_matrix'][m:m+num_meta_lower, m+num_meta_lower:m+num_meta_lower+num_meta_upper] += (
                self.data['cr_data']['gcrs'][str(i)]['acd']
            )

            # Populate QCDs in ion balance
            self.data['ion_matrix'][m+diag_lower[0], m+diag_lower[1]] -= (
                np.sum(self.data['cr_data']['gcrs'][str(i)]['qcd'], axis=1)
            )
            self.data['ion_matrix'][m:m+num_meta_lower, m:m+num_meta_lower] += (
                np.swapaxes(self.data['cr_data']['gcrs'][str(i)]['qcd'], 0, 1)
            )

            # Populate XCDs in ion balance
            self.data['ion_matrix'][m+num_meta_lower+diag_upper[0], m+num_meta_lower+diag_upper[1]] -= (
                np.sum(self.data['cr_data']['gcrs'][str(i)]['xcd'], axis=1)
            )
            self.data['ion_matrix'][m+num_meta_lower:m+num_meta_lower+num_meta_upper, m+num_meta_lower:m+num_meta_lower+num_meta_upper] += (
                np.swapaxes(self.data['cr_data']['gcrs'][str(i)]['xcd'], 0, 1)
            )

            # Populate CCDs in ion balance
            # This assumes the electron density and neutral hydrogen density grids are aligned
            if(self.data['user']['use_cx'][i]):
                self.data['ion_matrix'][m+num_meta_lower+diag_upper[0], m+num_meta_lower+diag_upper[1]] -= (
                    np.sum(
                        self.data['cr_data']['gcrs'][str(i)]['ccd'] * self.data['user']['hdens_grid'] / self.data['user']['dens_grid'],
                        axis=0,
                    )
                )
                self.data['ion_matrix'][m:m+num_meta_lower, m+num_meta_lower:m+num_meta_lower+num_meta_upper] += (
                    self.data['cr_data']['gcrs'][str(i)]['ccd'] * self.data['user']['hdens_grid'] / self.data['user']['dens_grid']
                )

            # Determine metastable state index for next ionization stage's ground state
            m += num_meta_lower

        # Populate transport term in ion balance
        self.data['ion_matrix'][np.diag_indices(num_meta)] -= 1 / ne_tau

        # Set the source vector, defaulting to only sourcing the neutral charge
        # state. For a user-given source, normalize it so the total source of
        # all charge states is 1. Technically, the source is not part of the
        # ionization balance matrix, but since it is part of the ionization
        # balance system of equations, it gets set here.
        if source is None:
            self.data['user']['source'] = np.zeros(num_meta)
            self.data['user']['source'][0] = 1
        else:
            if source.shape[0] != num_meta:
                raise ValueError("source for each metastable must be provided")
            self.data['user']['source'] = source / source.sum(axis=0)
        self.data['user']['source'] *= self.data['user']['dens_grid'] / ne_tau


    def solve_time_dependent(self, soln_times, init_abund=np.array([])):
        """
        Solves the ionization balance system of equations as a function of time.

        Populates the "pops_td", "eigen_val", and "eigen_vec" entries in the
        processed data dict.

        Parameters
        ----------
        soln_times : float array
            Times in seconds to calculate the solution at.
        init_abund : float array, optional
            The initial fractional abundances (at t=0) for each metastable.
            Defaults to 100% initial population in the ground state of the
            neutral charge state.
        """
        self.data['user']['soln_times'] = soln_times

        # Set default value for initial fractional abundance
        if init_abund.size < 1:
            self.data['user']['init_abund'] = np.zeros(self.data["ion_matrix"].shape[0])
            self.data['user']['init_abund'][0] = 1
        else:
            self.data['user']['init_abund'] = init_abund

        # Initialize data structure to hold solution data
        if 'processed' not in self.data.keys():
            self.data['processed'] = {}

        # Solve the ionization balance set of equations
        (
            self.data['processed']['pops_td'],
            self.data['processed']['eigen_val'],
            self.data['processed']['eigen_vec'],
        ) = solve_matrix_exponential_source(
            self.data['ion_matrix'] * self.data['user']['dens_grid'],
            self.data['user']['init_abund'],
            self.data['user']['source'],
            self.data['user']['soln_times'],
        )


    def solve_time_independent(self):
        """
        Finds the ionization balance system of equations steady-state solution.

        Populates the "pops_ss", "eigen_val", and "eigen_vec" entries in the
        processed data dict.
        """
        # Initialize data structure to hold solution data
        if 'processed' not in self.data.keys():
            self.data['processed'] = {}

        # Solve the ionization balance set of equations
        (
            self.data['processed']['pops_ss'],
            self.data['processed']['eigen_val'],
            self.data['processed']['eigen_vec'],
        ) = solve_matrix_exponential_ss_new(
            self.data['ion_matrix'] * self.data['user']['dens_grid'],
            self.data['user']['source'],
        )
