BYND
===
   This is the initial version of BYND.
   BYND is a code which combines approximate frequency
   results with exact short time dynamics to achieve
   an efficent calculation of electronic excitation
   spectra. Essentially, BYND is able to give
   highly reliable results when other super-resolution
   techniques such as compressed-sensing typically fail.

   BYND is totally independent of the underlying
   electronic structrue method/code.
   
   The only requirement is a time-dependent
   dipole signal (short time dynamics) and
   a sufficient accurate initial guess for the
   excitation spectrum.

   BYND is still under active development, and 
   future version will include support for:
   
   - quadrupole moments
   - basis set extrapolation
   - replacing the line-search with more
     advanced techniques
   - routines for easier data handling

For a preprint article see also:
https://arxiv.org/abs/2401.06929


STRUCTURE
   ----------------------------------------------
   Main code:

   BYND essentially consists of two routines:

      1. src/simple_line_search.py
         -> perform_line_search_tensor_off_diagonal

      2. src/continuum_amplitudes.py
         -> get_continuum_freq

   These routines do not depend on the electronic 
   structure code. As long the data is provided
   correctly, these will run.

   See the provided example to find out how to use
   both routines to perform an optimization.

   These two routines use helper routines to 
   perform the task which are located in:

      src/helpers_line_search.py

         -> update_amplitudes_tensor
         -> sort_amplitudes_tensor
         -> objective_tensor
         -> generate_signal_tensor
         -> get_search_grid

   All other routines are helper 
   functions to handle the data structure of 
   the FHIaims RT-TDDFT routines.

   ----------------------------------------------
   Examples:

      examples/optimize_frequencies.py

   This is an example for how to perform a
   calculation with BYND. Note, this example
   takes long time dynamics and cuts out a
   very short time signal. All operations
   are performed using this short time signal.
   In the end the long time signal is then used to
   compare the exact result with the result obtained 
   from BYND. For a real scenario, the long time
   dynamics signal is not known.

   This script is essentially a wrapper
   around different functions which mostly handles I/O
   operations. At the heart of the script is 
   the function call to 
   'perform_line_search_tensor_off_diagonal'
   and 'get_continuum_freq'. This is where the
   optimization is happening.

   The script assumes that all files which are
   present in data/ are in the same folder.

   Notes on how to use the example script:

   - copy all files in data/ into the folder where
     optimize_frequencies.py is located. It might
     be necessary to adjust the system paths in
     optimize_frequencies.py so that the script
     can find all necessary helper functions.

   - Then simply run the script
     ```bash
     python3 optimize_frequencies.py
     ```

   - The script should perform all necessary 
     calculations. In the end you should
     have spectrum_bynd_vs_exact.pdf in your
     folder which compares the spectrum of
     a long-time RT-TDDFT simulation with the
     spectrum obtained from BYND.

   Please use the example script for inspiration
   how to use BYND in its current state. 

   ----------------------------------------------
   Helpers:

   This folder contains helper functions
   which are there for I/O or for calculating
   the final excitation spectrum.
   This is highly specific for FHIaims and needs
   to be adabted/changed if one uses other
   electronic structure codes.

      helpers/get_spectrum_time_domain.py

      helpers/helpers_rt_tddft_fhiaims_utilities.py

      helpers/spectrum_helpers.py

      helpers/plot_spectrum.py
      -> simple script to plot the BYND
         results compared to the exact
         long time dynamics result

   ----------------------------------------------
   Data:

   Contains all relevant data to perform a simple
   example. 

      -> sma_tddft.data
         SMA frq and osci. strength

      -> trans_mom_sma.data
         SMA trans. dipole moments

      -> x.rt-tddft.dipole.dat
         RT-TDDFT dipole data E-field puls in x

      -> x.rt-tddft.ext-field.dat
         RT-TDDFT applied external field

      -> y.rt-tddft.dipole.dat
         RT-TDDFT dipole data E-field puls in y

      -> z.rt-tddft.dipole.dat
         RT-TDDFT dipole data E-field puls in z

   ----------------------------------------------
