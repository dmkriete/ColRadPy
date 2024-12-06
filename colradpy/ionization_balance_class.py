import numpy as np
from colradpy.colradpy_class import colradpy
from colradpy.solve_matrix_exponential import (
    solve_matrix_exponential_source,
    solve_matrix_exponential_steady_state,
    solve_matrix_exponential_ss_new,
)
from scipy.interpolate import interp2d
from colradpy.read_adf11 import read_adf11

def interp_rates_adf11(logged_temp,logged_dens,temp,dens,logged_gcr):
    gcr_arr = np.zeros( (np.shape(logged_gcr)[0],np.shape(logged_gcr)[1],len(temp),len(dens)) )
    #there has to be a better way to do this mess with the for loops but I can't be bothered
    #to spend the time to figure it out if this actually gets used for much it should be changed
    for i in range(0,np.shape(logged_gcr)[0]):
        for j in range(0,np.shape(logged_gcr)[1]):
            for k in range(0,len(temp)):
                for l in range(0,len(dens)):
                    interp_gcr = interp2d(logged_temp,
                                          logged_dens,
                                          logged_gcr[i,j,:,:].transpose(1,0),
                                          kind="cubic")
                    gcr_arr[i,j,k,l] = interp_gcr(np.log10(temp[k]),np.log10(dens[l]))
    return 10**gcr_arr


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
                        self.data['cr_data']['gcrs'][str(j)]['acd'] = interp_rates_adf11(adf11['input_file']['temp_grid'],
                                                                                         adf11['input_file']['dens_grid'],
                                                                          temp_grid,dens_grid,adf11['input_file'][str(j)])
                        self.data['input_file']['acd'] = adf11['input_file']
                        
                    if 'qcd' in file:
                        self.data['cr_data']['gcrs'][str(j)]['qcd'] = interp_rates_adf11(adf11['input_file']['temp_grid'],
                                                                                        adf11['input_file']['dens_grid'],
                                                                         temp_grid,dens_grid,adf11['input_file'][str(j)])
                        self.data['input_file']['qcd'] = adf11['input_file']
                        
                    if 'xcd' in file:
                        self.data['cr_data']['gcrs'][str(j)]['xcd'] = interp_rates_adf11(adf11['input_file']['temp_grid'],
                                                                                        adf11['input_file']['dens_grid'],
                                                                         temp_grid,dens_grid,adf11['input_file'][str(j)])
                        self.data['input_file']['xcd'] = adf11['input_file']                               
                    if 'ccd' in file:
                        self.data['cr_data']['gcrs'][str(j)]['ccd'] = interp_rates_adf11(adf11['input_file']['temp_grid'],
                                                                                        adf11['input_file']['dens_grid'],
                                                                         htemp_grid,hdens_grid,adf11['input_file'][str(j)])
                        self.data['input_file']['ccd'] = adf11['input_file']                        
                        
                    if 'qcd' not in self.data['cr_data']['gcrs'][str(j)]:
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


    def populate_ion_matrix(self, ne_tau=np.inf, source=np.array([])):
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

        The columns of this matrix sum to zero.

        Parameters
        ----------
        ne_tau : float
            Simulates the effect of transport on the ionization balance by
            adding a 1/tau term to each state's evolution equation. Tau is a
            characteristic timescale or "dwell time" for the transport. By
            convention, tau is scaled by the density, and this parameter should
            have units of cm^-3 s. Note that it is more common to simulate
            transport effects by performing a time-dependent CRM calculation
            and truncating the result at the dwell time, rather than the
            approach here of including a 1/tau decay term directly in the
            CRM equations. Positive values indicate transport away the
            simulation volume (resulting in decay of each charge state), while
            negative values indicate transport into the simulation volume
            (resulting in growth of each charge state). By default, this
            parameter is infinity (i.e. no transport).
        source : float array, optional
            The source rate in atoms per second for each metastable. Defaults
            to zero for all charge states.
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

        # Set the source vector (defaulting to zero). Technically, this is not
        # part of the ionization balance matrix, but since it is part of the
        # ionization balance system of equations, it gets set here.
        if source.size < 1:
            self.data['user']['source'] = np.zeros(num_meta)
        else:
            self.data['user']['source'] = source


    def solve_time_dependent(self, soln_times, init_abund=np.array([])):
        """
        Solves the ionization balance matrix given an initial populations and
        times when a source term is present

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

        # Solve the ionization balance set of equations with a source term
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
        Solves the ionization balance matrix for the steady-state (time-independent) solution.

        This is going to use the time dependent method just solving at 8 e-folding times
        for the second to smallest eigen value. Note that there are faster methods to do this
        but its more work to code it up and the time-dependent part is already done.
        This function is mostly a niceity for the casual user.
        The smallest eigenvalue corresponds to the
        steady state so we want to wait until the second to last componet completely dies off.
        This is done for the smallest over the given temperature and density parameter space
        this has been tested for 17 orders of magnitude and I haven't run into overflow problems.

        populates the pops, eigen_val and eigen_vec
        """
        # TODO: this solution does not account for transport effects (even with
        # dwell time decay added directly to the ionization matrix). I think
        # because transport effects require a source, and the matrix_exponential_steady_state function does not account for the source vector
        if 'processed' not in self.data.keys():
            self.data['processed'] = {}
        (
            self.data['processed']['pops_ss'],
            self.data['processed']['eigen_val'],
            self.data['processed']['eigen_vec'],
        # ) = solve_matrix_exponential_steady_state(
        ) = solve_matrix_exponential_ss_new(
            self.data['ion_matrix'] * self.data['user']['dens_grid'],
            self.data['user']['source'],
        )
