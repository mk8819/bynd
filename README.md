BYND
===
   This is the development version of BYND. 
   BYND is a code which combines approximate frequency
   results with exact short time dynamics to achieve
   an efficent calculation of electronic excitation
   spectra.

   BYND is totally independet of the underlying
   electronic structrue method/code.
   
   The only requirement is a time-dependent
   dipole signal (short time dynamics) and
   a sufficient accurate initial guess for the
   excitation spectrum.


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
   correctly.

   These two routines use helper routines to 
   perform the task which are located in:

      src/helpers_line_search.py

         -> update_amplitudes_tensor
         -> sort_amplitudes_tensor
         -> objective_tensor
         -> generate_signal_tensor
         -> get_search_grid

   That is all. All other routines are helper 
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
   from BYND. For a real scenary the long time
   dynamics signal is not known.

   This script is essentially some sort of wrapper
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

   ----------------------------------------------
   Helpers:

   This folder contains only helper functions
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