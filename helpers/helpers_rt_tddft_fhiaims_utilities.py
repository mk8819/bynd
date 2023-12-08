# class for all sma data stuff

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
from scipy import signal as sig
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import interp1d
from copy import deepcopy as dc
import warnings
warnings.filterwarnings('ignore')




class VectorData:
  """
  Container class for all data in column format 'TIME VALUE_X VALUE_Y VALUE_Z', i.e.
  a time series of a vector-valued function.
  """

  def __init__(self, file_name, descriptor, flag_info_only=False):

    self.file_name      = str(file_name)
    self.descriptor     = str(descriptor)
    self.flag_info_only = bool(flag_info_only)

    self.info = []
    self.data = { "t": [], "x": [], "y": [], "z": [], "abs": [] }
    self.n_info = 0
    self.n_data = 0

    self.units = { "time": None, "values": None }

    self.valid_units_time   = ["fs", "ps", "au"]
    self.valid_units_values = []

    self._read_data()

  def _read_data(self):

    if self.file_name is None:
      return

    def fix_format_E(self, spln):

      out = spln

      for i, sp in enumerate(spln):
    
        if sp.rfind("E") > -1:
          continue
        else:
          out[i] = sp[0] + sp[1:].replace("+", "E+") if sp[1:].rfind("+") > -1 \
                    else sp[0] + sp[1:].replace("-", "E-")

      return out    
      
    n_max_info = 50

    with open(self.file_name) as f:

      n_info = 0
      n_data = 0

      while True:

        ln = f.readline()

        if self.flag_info_only and n_data > 0 and n_data+n_info > n_max_info:
          print("> Stopping after reading info block")
          break

        if not ln:
          break

        if ln.rfind("#") != -1:
          self.info.append(ln)
          n_info += 1
        elif len(ln) == 1 or len(ln) == 0:
          continue
        else:
          spln = ln.split()

          try:  
            t, x, y, z = map(float, spln)
          except ValueError:
            format_check_E = -1 in [ x.rfind("E") for x in spln ]

            if format_check_E:
              t, x, y, z = map(float, fix_format_E(self, spln))
            else:
              raise ValueError("Unknown value error in line {0}".format(ln))

          self.data["t"].append(t)
          self.data["x"].append(x)
          self.data["y"].append(y)
          self.data["z"].append(z)
          self.data["abs"].append(np.linalg.norm([x,y,z]))
      
          n_data += 1 

      found_step = False
      for k in ["t", "x", "y", "z", "abs"]:

        if abs(self.data[k][0]-self.data[k][1]) > 1e-3:
          found_step = True
        self.data[k] = np.array(self.data[k])
  
    print("> Read input file {0} and found {1} data values.".format(self.file_name, n_data))        

    if found_step:
      print("> WARNING: it appears like you have read in data for a t=0 kick. There is a step in \n" \
            + "         the data that will cause unwanted artificial spectral behaviour. \n" \
            + "         It is recommended to remove the t=0 value and shift the time axis.")

  def _check_units(self):

    t_unit = self.units["time"]
    v_unit = self.units["values"]

    if t_unit == "a.u.":
      self.units["time"] = "au"
    if v_unit == "a.u.":
      self.units["values"] = "au"

    if not self.units["time"] in self.valid_units_time:
      raise Exception("> Invalid time unit: {0}".format(t_unit))

    if not self.units["values"] in self.valid_units_values:
      raise Exception("> Invalid values unit: {0}".format(v_unit))

class FieldData(VectorData):
  """
  Container for external field input data, inherits from VectorData class
  """

  def __init__(self, file_name, descriptor, flag_analytic_field=False):

    VectorData.__init__(self, file_name, descriptor, flag_info_only=flag_analytic_field)

    self.settings = { "gauge": None, "type": None, "t_on": None, "t_off": None, \
                      "sim_time": None, "amplitude": None, "frequency": None,   \
                      "freq_fac": None, "gauss_ctr": None, "gauss_wdt": None    }

    self.file_desc = "RT-TDDFT EXTERNAL FIELD INFO"

    self.valid_units_values = ["au", "ev_m"] # TBD

    self._check_settings()
    self._check_units()

  def _check_settings(self):

    flag_correct_file = False

    for iln in self.info:

      if iln.rfind(self.file_desc):
        flag_correct_file = True

      if iln.rfind("GAUGE:") != -1:
        if iln.rfind("LENGTH") != -1:
          self.settings["gauge"] = "length"
        elif iln.rfind("VELOCITY") != -1:
          self.settings["gauge"] = "velocity"
        else:
          raise Exception("Gauge choice invalid")
  
      if iln.rfind("TYPE") != -1:
        self.settings["type"] = int(iln.split("=")[-1]) 

      if iln.rfind("T_ON") != -1:
        self.settings["t_on"] = float(iln.split("=")[-1]) 
  
      if iln.rfind("T_OFF") != -1:
        self.settings["t_off"] = float(iln.split("=")[-1])  
      
      if iln.rfind("SIM_TIME") != -1:
        self.settings["sim_time"] = float(iln.split("=")[-1]) 

      if iln.rfind("AMPLITUDE") != -1:
        self.settings["amplitude"] = [ float(a) for a in iln.split("=")[-1].split() ] 

      if iln.rfind("FREQUENCY") != -1:
        self.settings["frequency"] = float(iln.split("=")[-1])  

      if iln.rfind("FREQ_FAC") != -1:
        self.settings["freq_fac"] = float(iln.split("=")[-1]) 

      if iln.rfind("GAUSS_CTR") != -1:
        self.settings["gauss_ctr"] = float(iln.split("=")[-1])  

      if iln.rfind("GAUSS_WDT") != -1:
        self.settings["gauss_wdt"] = float(iln.split("=")[-1])  

      if iln.rfind("UNITS") != -1:
        spln = iln.split()

        self.units["time"]   = str(spln[4])
        self.units["values"] = str(spln[8])

    for sett in self.settings:
      if self.settings[sett] is None:
        raise Exception("> Input parameter missing: {0}".format(sett))

    for unit in self.units:
      if self.units[unit] is None:
        raise Exception("> Units missing or incorrect: {0}".format(unit))

    if not flag_correct_file:
      raise Exception("> Couldn't find descriptor {0} - wrong file?".format(self.file_desc))

    print("> Confirmed settings for external field.")

class DipoleData(VectorData):
  """
  Container for time-dependent dipole input data, inherits from VectorData class
  """

  def __init__(self, file_name, descriptor):

    VectorData.__init__(self, file_name, descriptor)

    self.file_desc = "TIME-DEPENDENT DIPOLE MOMENT"

    self.valid_units_values = ["au", "cang"] # TBD

    self._check_settings()
    self._check_units()

  def _check_settings(self):

    flag_correct_file = False

    for iln in self.info:

      if iln.rfind(self.file_desc):
        flag_correct_file = True

        if iln.rfind("UNITS") != -1:
          spln = iln.split()

          self.units["time"]   = str(spln[4])
          self.units["values"] = str(spln[8])

    for unit in self.units:
      if self.units[unit] is None:
        raise Exception("> Units missing or incorrect: {0}".format(unit))

class CurrentData(VectorData):
  """
  Container for time-dependent current input data, inherits from VectorData class
  """

  def __init__(self, file_name, descriptor):

    VectorData.__init__(self, file_name, descriptor)

    self.file_desc = "TIME-DEPENDENT CURRENT"

    self.valid_units_values = ["au", "X"] # TBD

    self._check_settings()
    self._check_units()

  def _check_settings(self):

    flag_correct_file = False

    for iln in self.info:

      if iln.rfind(self.file_desc):
        flag_correct_file = True

        if iln.rfind("UNITS") != -1:
          spln = iln.split()

          self.units["time"]   = str(spln[4])
          self.units["values"] = str(spln[8])

    for unit in self.units:
      if self.units[unit] is None:
        raise Exception("> Units missing or incorrect: {0}".format(unit))


class MagMomData(VectorData):
  """
  Container for time-dependent magnetic moment input data, inherits from VectorData class
  """

  def __init__(self, file_name, descriptor):

    VectorData.__init__(self, file_name, descriptor)

    self.file_desc = "TIME-DEPENDENT MAGNETIC MOMENT"

    self.valid_units_values = ["au", "X"] # TBD

    self._check_settings()
    self._check_units()

  def _check_settings(self):

    flag_correct_file = False

    for iln in self.info:

      if iln.rfind(self.file_desc):
        flag_correct_file = True

        if iln.rfind("UNITS") != -1:
          spln = iln.split()

          self.units["time"]   = str(spln[4])
          self.units["values"] = str(spln[8])

    for unit in self.units:
      if self.units[unit] is None:
        raise Exception("> Units missing or incorrect: {0}".format(unit))

class HHGData(VectorData):
  """
  Container for time-dependent HHG input data, inherits from VectorData class
  """

  def __init__(self, file_name, descriptor):

    VectorData.__init__(self, file_name, descriptor)

    self.file_desc = "TIME-DEPENDENT HHG POTENTIAL GRADIENT PART"

    self.valid_units_values = ["au", "X"] # TBD

    self._check_settings()
    self._check_units()

  def _check_settings(self):

    flag_correct_file = False

    for iln in self.info:

      if iln.rfind(self.file_desc):
        flag_correct_file = True

        if iln.rfind("UNITS") != -1:
          spln = iln.split()

          self.units["time"]   = str(spln[4])
          self.units["values"] = str(spln[8])

    for unit in self.units:
      if self.units[unit] is None:
        raise Exception("> Units missing or incorrect: {0}".format(unit))

class ExternalField:
  """
  Container and calculator for analytical external field when input field not used.
  """

  def __init__(self, field_settings):

    self.field_settings = field_settings

  def get_field_func_ft(self):

    sett  = self.field_settings

    ftype = sett["type"]

    w0  = sett["frequency"]
    amp = sett["amplitude"]
    fw  = sett["freq_fac"]
    gct = sett["gauss_ctr"]
    gwd = sett["gauss_wdt"]

    # Gaussian as approx. for delta function, if needed
    def delta_g(w, w0, wd):
      return np.exp(-(w-w0)**2/(2*wd**2))

    if ftype == 1:

      return lambda w: 1.0

    elif ftype == 2:

      # E(t) = exp( -(t-t0)^2 / (4 * wd^2) ) ---> effectively a delta function, therefore we have
      # F[delta(t-t0)] = exp(-2*pi*i*omega*t0) (since in numpy 2*pi is used in the (F)FT)
      # return lambda w: 1.0 # np.exp(-2.0 * np.pi * 1.0j * w * gct)
      sig = gwd * 2.845  # 1.5888 # 0.7965  
      # return lambda w: 0.25 * np.sqrt(2*np.pi) * gwd * np.exp(-2*(np.pi*w*sig)**2) * np.exp(-1.0j*w*50)
      return lambda w: 1.0 # 0.25 * np.sqrt(2*np.pi) * sig # * np.exp(-2*(np.pi*w*sig)**2) * np.exp(-1.0j*w*50)

    elif ftype == 3:

      c0 = np.sqrt(np.pi/8.0) * w0

      return lambda w: -c0 * ( delta_g(w, w0, gwd) + delta_g(w, -w0, gwd) )

    elif ftype == 4:

      c0 = np.sqrt(np.pi/8.0) 
      c1 = np.sqrt(np.pi**3/8.0) / (w0 * fw)
      w1 = 2.0 * np.pi / fw
      
      return lambda w:   1.0j * c0 * ( delta_g(w, w0, gwd) - delta_g(w, -w0, gwd) )              \
                       - 0.5j * c0 * (   delta_g(w, w0 + w1, gwd) + delta_g(-w, -w0 + w1, gwd)   \
                                       - delta_g(w, -w0 - w1, gwd) - delta_g(w, -w0 + w1, gwd) ) \
                       + 1.0j * c1 * (   delta_g(w, w0 + w1, gwd) - delta_g(-w, -w0 + w1, gwd)   \
                                       - delta_g(w, -w0 - w1, gwd) + delta_g(w, -w0 + w1, gwd) )

    elif ftype == 5: 

      c0 = np.sqrt(np.pi/8.0) * w0
      c1 = np.sqrt(np.pi/8.0) / fw

      return lambda w: - c0 * ( delta_g(w, w0, gwd) + delta_g(w, -w0, gwd) ) \
                       + ( c1 + c0/2.0 ) * delta_g(w, (-1.0+1.0/c1)*w0, gwd)  \
                       + ( c1 + c0/2.0 ) * delta_g(w, (1.0-1.0/c1)*w0, gwd)   \
                       - ( c1 - c0/2.0 ) * delta_g(-w, (1.0+1.0/c1)*w0, gwd)  \
                       - ( c1 - c0/2.0 ) * delta_g(w, (1.0+1.0/c1)*w0, gwd)  

    else:
      raise Exception("> Electric field type {0} not supported!".format(ftype))

class UnitConversion:
  """
  All functionality for unit conversion
  """
  
  def __init__(self, input_data=VectorData, convert_values_fac=1.0):

    self.input_data           = input_data
    self.conv_unit_values_fac = float(convert_values_fac)

    self.dat_unit_time = input_data.units["time"]
    self.dat_unit_vals = input_data.units["values"] 

    self._convert_units_time_to_au()
    self._convert_units_vals_to_au()

  def return_data(self):
    return self.input_data

  # All internal computations are done with atomic units, so we need to convert time,
  # which is usually in fs, to a.u.
  def _convert_units_time_to_au(self):

    in_unit_t  = self.dat_unit_time
    out_unit_t = "au"

    if in_unit_t == out_unit_t:
      return

    conv_t = 1.0

    t_au_fs = 0.024188843265857 

    if in_unit_t == "ps":
      conv_t = 1e-3 * t_au_fs
    elif in_unit_t == "fs":
      conv_t = t_au_fs
    else:
      raise Exception("> Time unit conversion invalid!")

    self.input_data.data["t"] /= conv_t

  # Input observables are usually in au, not much to do
  def _convert_units_vals_to_au(self):
  
    in_unit_v  = self.dat_unit_vals
    out_unit_v = "au"

    if in_unit_v == out_unit_v:
      return

    if in_unit_v == "si" and out_unit_v == "au":
      self.input_data.data["x"]   *= 1.0/self.conv_unit_values_fac
      self.input_data.data["y"]   *= 1.0/self.conv_unit_values_fac
      self.input_data.data["z"]   *= 1.0/self.conv_unit_values_fac
      self.input_data.data["abs"] *= 1.0/self.conv_unit_values_fac

class Observable:

  conv_t_au_to_fs       = 0.024188843265857
  conv_2pi_per_au_to_ev = 27.211386245988 # 4.13566769692 * 0.024188843265857

  def __init__(self, flag_plot=False, flag_normalize=False,flag_intp_plot=False, flag_analytic_ft=True, \
               flag_copy_fields=True, intp_fac=None, conv_obs_au2si=1.0, conv_fld_au2si=1.0, t_min=0, \
               lpeak_hgt=None, lpeak_wdt=None, t_shift=0, n_ft_pade=-1, fmin_pade=0.0, fmax_pade=-1.0, \
               t_max=0, f_min=0, f_max=-1, add_zeros=0, t0_damp=-1.0, damp_type="poly", expfac_dp=0.00001, \
               calc_type="standard",  out_file_name=None):

    self.flag_plot        = bool(flag_plot)
    self.flag_normalize   = bool(flag_normalize)
    self.flag_intp_plot   = bool(flag_intp_plot) 
    self.flag_analytic_ft = bool(flag_analytic_ft)
    self.flag_copy_fields = bool(flag_copy_fields)
    self.intp_fac         = float(intp_fac) if intp_fac is not None else None
    self.conv_obs_au2si   = float(conv_obs_au2si)
    self.conv_fld_au2si   = float(conv_fld_au2si)
    self.locpeak_height   = float(lpeak_hgt) if lpeak_hgt is not None else None
    self.locpeak_width    = float(lpeak_wdt) if lpeak_wdt is not None else None
    self.t_min            = float(t_min)
    self.t_max            = float(t_max)
    self.t_shift          = float(t_shift)
    self.f_min            = float(f_min)
    self.f_max            = float(f_max) 
    self.add_zeros        = int(add_zeros)
    self.n_ft_pade        = int(n_ft_pade)
    self.fmin_pade        = float(fmin_pade)
    self.fmax_pade        = float(fmax_pade)
    self.t0_damp          = float(t0_damp)
    self.expfac_damp      = float(expfac_dp)
    self.damp_type        = str(damp_type) # poly / exp / sinsq
    self.calc_type        = str(calc_type)
    self.out_file_name    = str(out_file_name) if out_file_name is not None else None

    self.flag_damp_ft = False if self.t0_damp < 0.0 else True
    self.flag_detrend = True # default
    self.flag_locpeak = True if self.locpeak_height is not None and self.locpeak_width is not None else False
    self.flag_ft_via_pade = True if self.n_ft_pade >= 0 and self.fmax_pade >= 0 else False 

  def _write_output_data(self, out_name, desc, comm, xvals, *values):

    def println(x, v):
      ostr = ""
      vs  = [x] + [ vi for vi in v ] if isinstance(v, tuple) else [x, v]
      for vi in np.array(vs):
        ostr += "{0:20.10f}   ".format(vi)
      return ostr + "\n"

    out_file = out_name + "." + desc + ".dat"

    with open(out_file, "w") as f_out:

      f_out.write(comm)
    
      for xvi in zip(xvals, *values):
        f_out.write(println(xvi[0], xvi[1:]))
        
    print("> File written: {0}".format(out_file)) 

  def _interpolate_data(self, flag_unshift, intp_data, intp_fac):

    if intp_fac <= 1.0:
      raise Exception("> Invalid value for interpolation factor. Needs to be > 1")

    if flag_unshift:  

      n_mean_start = len(intp_data) // 2

      mean_curr = sum(intp_data[n_mean_start:]) / len(intp_data[n_mean_start:])

      intp_data = np.array([ vi - mean_curr for vi in intp_data ])

    n_data = len(intp_data)
    x_intp = np.arange(0, n_data, 1/intp_fac)
    intp = interp1d(range(n_data), intp_data, kind="cubic", fill_value="extrapolate")

    return intp(x_intp)[:-1]

  def _locate_peaks(self, f_axis, val_axis):

    # percent of max for setting min values
    if self.locpeak_height > 1e-5:
      
      vmax = np.amax(val_axis)

      minhgt = self.locpeak_height * vmax 

    else:

      minhgt = None

    if self.locpeak_width > 1e-5:

      df_ev = (f_axis[1] - f_axis[0]) * self.conv_2pi_per_au_to_ev

      minwdt = int(self.locpeak_width / df_ev)

    else:

      winwdt = None

    return sig.find_peaks(val_axis, height=minhgt, width=minwdt)

class VectorObservable(Observable):

  def __init__(self, data_x=None, data_y=None, data_z=None, input_field=None, **kwargs):

    Observable.__init__(self, **kwargs)

    self.data_x = dc(data_x) # VectorData
    self.data_y = dc(data_y) # VectorData
    self.data_z = dc(data_z) # VectorData

    self.field = dc(input_field) # FieldData

    self.obs_data   = { "t": [], "x": [], "y": [], "z": [], "abs": [] }

    self.field_ft_data = { "f": [], "x": [], "y": [], "z": [], "abs": [] }
    self.obs_ft_data   = { "f": [], "x": [], "y": [], "z": [], "abs": [] }

    self.delta_t = None

    self._check_data()
    self._convert_units()
    self._setup_data()
    self._calculate_fts()

    if self.intp_fac is not None:
      self._interpolate_ft_vec()

  def _check_data(self):

    dat = [self.data_x, self.data_y, self.data_z]

    if not None in dat:

      lx = len(self.data_x.data["t"])
      ly = len(self.data_y.data["t"])
      lz = len(self.data_z.data["t"])

      #TBD
      if not ( lx == ly == lz ):
        raise Exception("> Incompatible input data lengths! [IMPLEMENT CROP]")

      uxt = self.data_x.units["time"] 
      uyt = self.data_y.units["time"] 
      uzt = self.data_z.units["time"] 

      #TBD
      if not ( uxt == uyt == uzt ):
        raise Exception("> Incompatible time units! [IMPLEMENT CONVERSION]")

      uxv = self.data_x.units["values"] 
      uyv = self.data_y.units["values"] 
      uzv = self.data_z.units["values"] 

      #TBD
      if not ( uxv == uyv == uzv ):
        raise Exception("> Incompatible values units! [IMPLEMENT CONVERSION]")

    else:

      if dat.count(None) != 2:
        raise Exception("> If input data is given as None, the amount can be only 2, else all must not be None")

  def _convert_units(self):

    dat = [self.data_x, self.data_y, self.data_z]

    if self.data_x is not None:
      uconv = UnitConversion(self.data_x, convert_values_fac=self.conv_obs_au2si)
      self.data_x = uconv.return_data() 

    if self.data_y is not None:
      uconv = UnitConversion(self.data_y, convert_values_fac=self.conv_obs_au2si)
      self.data_y = uconv.return_data() 

    if self.data_z is not None:
      uconv = UnitConversion(self.data_z, convert_values_fac=self.conv_obs_au2si)
      self.data_z = uconv.return_data() 

    if self.field is not None:
      uconv = UnitConversion(self.field, convert_values_fac=self.conv_fld_au2si)
      self.field = uconv.return_data()  

    print("> Converted units: time, observable & field --> atomic units")

  def _setup_data(self):

    tc = Observable.conv_t_au_to_fs

    if self.calc_type == "standard":

      i_tmin = 0
      i_tmax = len(self.data_x.data["t"])-1

      if self.t_min > 0.0:
        for i, ti in enumerate(self.data_x.data["t"]):
          if ti >= self.t_min/tc:
            i_tmin = i
            print("> t(min) = {0:0.2f} fs @ index {1}".format(ti*tc, i))
            break 

      if self.t_max > 0.0:
        for i, ti in enumerate(self.data_x.data["t"]):
          if self.t_max/tc <= ti:
            i_tmax = i
            print("> t(max) = {0:0.2f} fs @ index {1}".format(ti*tc, i))
            break
      else:
        print("> t(max) = {0:0.2f} fs".format(self.data_x.data["t"][-1]*tc))

      inp_data = [self.data_x, self.data_y, self.data_z]
      if None in inp_data:

        ref_data = None

        print("> Using given input data as input for all x, y, z components")

        for dat in inp_data:
          if dat is not None:
            ref_data = dc(dat)
            break

        xvals = dc(ref_data.data["x"])
        yvals = dc(ref_data.data["y"])
        zvals = dc(ref_data.data["z"])
 
      else:

        xvals = dc(self.data_x.data["x"])
        yvals = dc(self.data_y.data["y"])
        zvals = dc(self.data_z.data["z"])

      self.obs_data["t"]   = np.array(self.data_x.data["t"][i_tmin:i_tmax])
      self.obs_data["x"]   = np.array(xvals[i_tmin:i_tmax])
      self.obs_data["y"]   = np.array(yvals[i_tmin:i_tmax])
      self.obs_data["z"]   = np.array(zvals[i_tmin:i_tmax])
      self.obs_data["abs"] = np.array([ np.linalg.norm([xi, yi, zi]) for (xi, yi, zi) \
                                        in zip(xvals, yvals, zvals)                  ])

      self.field.data["t"]   = self.field.data["t"][i_tmin:i_tmax]
      self.field.data["x"]   = self.field.data["x"][i_tmin:i_tmax]
      self.field.data["y"]   = self.field.data["y"][i_tmin:i_tmax]
      self.field.data["z"]   = self.field.data["z"][i_tmin:i_tmax]
      self.field.data["abs"] = self.field.data["abs"][i_tmin:i_tmax]

      self.delta_t = abs(self.obs_data["t"][1] - self.obs_data["t"][0])

      if self.delta_t < 1.e-5:
        raise Exception("> Time step seems to be zero. Check time values in input.")

      #TBD
      # if not self.flag_analytic_ft:
      #  raise Exception("> Calculation with non-analytic field not implemented!")

    else:
      raise Exception("> Calculation type {0} not implemented!".format(self.calc_type))

  def _interpolate_ft_vec(self):

    obs_ft_save = dc(self.obs_ft_data)

    keys = [ "f", "x", "y", "z", "abs" ]

    print("> Interpolating FT data. N_old = {0}, N_intp = {1}".format(len(self.obs_ft_data["f"]),
          int(len(self.obs_ft_data["f"])*self.intp_fac)))

    for k in keys:

      self.obs_ft_data[k] = self._interpolate_data(False, self.obs_ft_data[k], self.intp_fac)
      self.field_ft_data[k] = self._interpolate_data(False, self.field_ft_data[k], self.intp_fac)
  
    if self.flag_intp_plot:

      xlabel = "Frequency (au)"

      fig, ax = plt.subplots(1, 1, sharex=True)

      xaxis = self.obs_ft_data["f"]

      ax.plot(xaxis, self.obs_ft_data["x"], color="red", label=r"FT$_x^\mathrm{intp}$", linewidth=1)
      ax.scatter(obs_ft_save["f"], obs_ft_save["x"], color="red", label=r"FT$_x^\mathrm{orig}$")

      ax.plot(xaxis, self.obs_ft_data["y"], color="blue", label=r"FT$_y^\mathrm{intp}$", linewidth=1)
      ax.scatter(obs_ft_save["f"], obs_ft_save["y"], color="blue", label=r"FT$_y^\mathrm{orig}$")

      ax.plot(xaxis, self.obs_ft_data["z"], color="green", label=r"FT$_z^\mathrm{intp}$", linewidth=1)
      ax.scatter(obs_ft_save["f"], obs_ft_save["z"], color="green", label=r"FT$_z^\mathrm{orig}$")

      ax.grid()
      ax.legend(loc="upper right")
      ax.set_title("Interpolated FT data")

  def _damp_signal(self, data):

    def f_damp(t, t0, t_tot):
      """
        FIX: at the moment hardcoded to exponential damping with tmax threshold 0.00001
             alternatively use polynomial damping: this does not affect the f-sum rule (exponential damp. does)
      """

      ex_thresh = self.expfac_damp

      if self.damp_type == "sinsq":
        return np.sin(np.pi*(t-t_tot/2.)/t_tot)**2
      elif self.damp_type == "exp":
        # return np.exp(-t/(-t_tot/np.log(ex_thresh)))
        return np.exp(-t/self.expfac_damp*Observable.conv_t_au_to_fs)
      elif self.damp_type == "poly":
        return 1.0 - 3.0 * ((t-t0)/(t_tot-t0))**2 + 2.0 * ((t-t0)/(t_tot-t0))**3 if t >= t0 else 1.0
      else:
        raise Exception("Damping function type invalid")

    tc = self.conv_t_au_to_fs

    t0    = self.t0_damp / tc
    t_tot = data["t"][-1]

    xavg50 = np.average(data["x"][int(len(data["x"])*0.95):])
    yavg50 = np.average(data["y"][int(len(data["y"])*0.95):])
    zavg50 = np.average(data["z"][int(len(data["z"])*0.95):])

    dsh_x = np.array(data["x"]) - xavg50
    dsh_y = np.array(data["y"]) - yavg50
    dsh_z = np.array(data["z"]) - zavg50

    data["x"] = [ x * f_damp(t, t0, t_tot) for (t, x) in zip(data["t"], dsh_x) ] # data["x"]) ]
    data["y"] = [ y * f_damp(t, t0, t_tot) for (t, y) in zip(data["t"], dsh_y) ] # data["y"]) ]
    data["z"] = [ z * f_damp(t, t0, t_tot) for (t, z) in zip(data["t"], dsh_z) ] # data["z"]) ]

    with open("check_damp.dat", "w") as f:
      for ti, xi, yi, zi in zip(data["t"], data["x"], data["y"], data["z"]):
        f.write("{0} {1} {2} {3}\n".format(ti, xi, yi, zi))

    print("> Signal damped with {0} type function".format(self.damp_type) \
          +", t0 = {0:0.1f}, tmax = {1:0.1f} fs".format(t0*tc, t_tot*tc))
    print("-> Converging to data averages (50% of points before damp): {0:1.3g} {1:1.3g} {2:1.3g}" \
          .format(xavg50, yavg50, zavg50))

  def _calculate_fts(self):

    ft_data = { "t": self.obs_data["t"], "x": self.obs_data["x"], "y": self.obs_data["y"], \
                "z": self.obs_data["z"] }
      
    if self.flag_damp_ft:
      self._damp_signal(ft_data)

    if self.flag_ft_via_pade:

      cv = self.conv_2pi_per_au_to_ev

      fminp = self.fmin_pade/cv if self.fmin_pade >= self.f_min else self.f_min/cv
      fmaxp = self.fmax_pade/cv if self.fmax_pade >= self.f_max else self.f_max/cv

      f_obs, ftx_obs, fty_obs, ftz_obs, fta_obs             \
        = calculate_fft_vec_pade(ft_data["x"], ft_data["y"], ft_data["z"], \
          self.delta_t, n_bins=self.n_ft_pade, fmin=fminp, fmax=fmaxp, normalize=self.flag_normalize)
    else:
      f_obs, ftx_obs, fty_obs, ftz_obs, fta_obs             \
        = calculate_fft_vec(ft_data["x"], ft_data["y"], ft_data["z"], \
          self.delta_t, detrend=self.flag_detrend, add_zeros=self.add_zeros, normalize=self.flag_normalize)

    self.obs_ft_data["f"] = f_obs.real # * np.pi * 2.0

    tsh = self.t_shift / Observable.conv_t_au_to_fs

    if abs(tsh) > 1e-5:

      self.obs_ft_data["x"]   = [ xi * np.exp(1.0j * wi * tsh) for (wi, xi) in zip(f_obs.real, ftx_obs) ]
      self.obs_ft_data["y"]   = [ yi * np.exp(1.0j * wi * tsh) for (wi, yi) in zip(f_obs.real, fty_obs) ]
      self.obs_ft_data["z"]   = [ zi * np.exp(1.0j * wi * tsh) for (wi, zi) in zip(f_obs.real, ftz_obs) ]

    else:

      self.obs_ft_data["x"]   = ftx_obs
      self.obs_ft_data["y"]   = fty_obs
      self.obs_ft_data["z"]   = ftz_obs

    self.obs_ft_data["abs"] = fta_obs

    if not self.flag_analytic_ft:

      ft_data = { "t": self.field.data["t"], "x": self.field.data["x"], "y": self.field.data["y"], \
                  "z": self.field.data["z"] }

      #TBD ever used?
      if False:
        self._damp_signal()

      if self.flag_ft_via_pade:
        f_fld, ftx_fld, fty_fld, ftz_fld, fta_fld             \
          = calculate_fft_vec_pade(ft_data["x"], ft_data["y"], ft_data["z"], \
            self.delta_t, n_bins=self.n_ft_pade, fmin=fminp, fmax=fmaxp, normalize=self.flag_normalize)
      else:
        f_fld, ftx_fld, fty_fld, ftz_fld, fta_fld             \
          = calculate_fft_vec(ft_data["x"], ft_data["y"], ft_data["z"], \
            # self.delta_t, detrend=False, add_zeros=self.add_zeros, normalize=self.flag_normalize)
            self.delta_t, detrend=self.flag_detrend, add_zeros=self.add_zeros, normalize=self.flag_normalize)

      self.field_ft_data["f"]   = f_fld.real # * np.pi * 2.0
      self.field_ft_data["x"]   = ftx_fld
      self.field_ft_data["y"]   = fty_fld
      self.field_ft_data["z"]   = ftz_fld
      self.field_ft_data["abs"] = fta_fld

    else:

      #TBD after copying this should be OK, but in general this is risky
      field = ExternalField(self.field.settings)

      if field.field_settings["type"] == 2:
        print("> Gaussian pulse center: {0:12.4f} fs".format(field.field_settings["gauss_ctr"]*Observable.conv_t_au_to_fs))
        print("> Gaussian pulse width:  {0:12.4f} fs".format(field.field_settings["gauss_wdt"]*Observable.conv_t_au_to_fs))

      self.field_ft_data["f"] = dc(self.obs_ft_data["f"])

      dt = self.obs_data["t"][1] - self.obs_data["t"][0]

      ft_anl = list(map(field.get_field_func_ft(), self.field_ft_data["f"]))

      if True in np.isnan(ft_anl):
        raise Exception("> Found bad value/s (NaN) in analytic FT of electric field. Try non-analytic FT option.")

      if abs(sum(ft_anl)) < 1e-10 :
        raise Exception("> Found bad value/s (all zero) in analytic FT of electric field. Try non-analytic FT option.")

      # if field_x.field_settings["gauge"] == "velocity":
      #   fac = -1.0/self.c
      #   print("> Velocity gauge chosen: scaling amplitudes by -1/c to generate electric field data")
      # elif field_x.field_settings["gauge"] == "length":
      #   fac = 1.0
      # else:
      #   raise Exception("> Gauge {0} invalid!".format(field_x.field_settings["gauge"]))
 
      self.field_ft_data["x"] = np.array([ field.field_settings["amplitude"][0] * ft for ft in ft_anl ])
      self.field_ft_data["y"] = np.array([ field.field_settings["amplitude"][1] * ft for ft in ft_anl ])
      self.field_ft_data["z"] = np.array([ field.field_settings["amplitude"][2] * ft for ft in ft_anl ])

      self.field_ft_data["abs"] = np.array([ np.sqrt(ftx**2+fty**2+ftz**2) for (ftx, fty, ftz) in                   \
                                    zip(self.field_ft_data["x"], self.field_ft_data["y"], self.field_ft_data["z"]) ])

      # if self.flag_normalize:
      #   self.field_ft_data["x"]   /=  np.amax(self.field_ft_data["x"])
      #   self.field_ft_data["y"]   /=  np.amax(self.field_ft_data["x"])
      #   self.field_ft_data["z"]   /=  np.amax(self.field_ft_data["x"])
      #   self.field_ft_data["abs"] /=  np.amax(self.field_ft_data["abs"])

    i_fmin = 0
    i_fmax = 1

    if self.f_max <= 0.0:
      i_fmax = len(self.obs_ft_data["f"]) - 1

    fmin = self.f_min
    fmax = self.f_max 

    cv = self.conv_2pi_per_au_to_ev

    for f in self.obs_ft_data["f"]:

      if f < fmin/cv:
        i_fmin += 1
  
      if f < fmax/cv and fmax > 0.0:
        i_fmax += 1

    for (ef, of) in zip(self.field_ft_data, self.obs_ft_data):

      self.obs_ft_data[of]   = dc(self.obs_ft_data[of][i_fmin:i_fmax])
      self.field_ft_data[ef] = dc(self.field_ft_data[ef][i_fmin:i_fmax])

    print("> FFT for input data and field calculated.")
    print("-> f(min) = {0:8.4f} eV @ index {1}".format(float(self.obs_ft_data["f"][0])*cv, i_fmin))
    print("-> f(max) = {0:8.4f} eV @ index {1}".format(float(self.obs_ft_data["f"][i_fmax-2-i_fmin])/cv, i_fmax))
    if self.add_zeros > 0:
      tmax_p = (self.obs_data["t"][-1] + self.delta_t * self.add_zeros) * Observable.conv_t_au_to_fs
      print("-> t(max, zero-padded) = {0:8.4f} fs".format(tmax_p))

  def normalize_max(self, *data_series):

    out_series = []

    for d in data_series:

      n = np.amax(d)

      if abs(n) < 1.0e-10:
        n = 1.0

      out_series.append([ di / n for di in d ])

    return out_series[0] if len(out_series) == 1 else tuple(out_series)

class ElectronicDipole(VectorObservable):
  """
  Class for any electronic diple data from which we can calculate the adsorption strength etc.
  """

  desc = "electronic_dipole"
  conv_obs_au2si = 8.4783536255e-30 # Cm
  conv_fld_au2si = 5.14220674763e11 # V/m

  def __init__(self, **kwargs):

    VectorObservable.__init__(self, conv_obs_au2si=ElectronicDipole.conv_obs_au2si, \
      conv_fld_au2si=ElectronicDipole.conv_fld_au2si, **kwargs)

    self.pol   = { "f": [], "xx": [], "yy": [], "zz": [], "axx": [], "ayy": [], "azz": [] }
    self.stren = { "f": [], "val": [], "aval": [], "int": 0 }
    self.powsp = { "f": [], "val": [], "aval": [], "int": 0, "f_peaks": [], "val_peaks": [] }

    self._calculate_polarisability()
    self._calculate_power_spectrum()
    self._calculate_abs_strength()

    if self.flag_plot:
      self._plot_results()

    if self.out_file_name is not None:
      self._write_results()

  def _write_results(self):

    en_vals = self.pol["f"] * Observable.conv_2pi_per_au_to_ev

    desc = "dipole_ft"
    comm = "# FT OF DIPOLE MOMENT\n # UNITS: [ENER] = eV | [POL] = atomic \n # ENERGY | |D_X| | |D_Y| | |D_Z| \n"

    vals = [ np.abs(self.obs_ft_data[vv]) for vv in ["x", "y", "z"] ]

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *vals)

    desc = "polarisability"
    comm = "# POLARISABILITY \n # UNITS: [ENER] = eV | [POL] = atomic \n # ENERGY | |P_XX| | |P_YY| | |P_ZZ| \n"

    vals = [ self.pol[vv] for vv in ["axx", "ayy", "azz"] ]

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *vals)

    desc = "abs_strength"
    comm = "# ABSORPTION STRENGTH \n # UNITS: [ENER] = eV | [ABS] = 1/eV \n # ENERGY | ABS \n"

    conv_har = 27.211386245

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *[self.stren["val"]/conv_har])

    desc = "pow_spec"
    comm = "# POWER SPECTRUM \n # UNITS: [ENER] = eV | [POWSP] = atomic \n # ENERGY | |POWSP| \n"

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *[self.powsp["aval"]])

    if self.flag_locpeak:
      desc = "pow_spec_peaks"
      comm = "# POWER SPECTRUM PEAKS \n # UNITS: [ENER] = eV | [POWSP] = atomic \n # ENERGY | POWSP \n"

      self._write_output_data(self.out_file_name, desc, comm, self.powsp["f_peak"]*Observable.conv_2pi_per_au_to_ev, \
                              *[self.powsp["val_peaks"]])

  def _calculate_polarisability(self):

    obs_ft = dc(self.obs_ft_data)
    fld_ft = dc(self.field_ft_data)

    self.pol["xx"] = [ p / e if abs(e) > 1.0e-12 else 0 for p, e in zip(obs_ft["x"], fld_ft["x"]) ]
    self.pol["yy"] = [ p / e if abs(e) > 1.0e-12 else 0 for p, e in zip(obs_ft["y"], fld_ft["y"]) ]
    self.pol["zz"] = [ p / e if abs(e) > 1.0e-12 else 0 for p, e in zip(obs_ft["z"], fld_ft["z"]) ]

    self.pol["axx"] = np.abs(self.pol["xx"])
    self.pol["ayy"] = np.abs(self.pol["yy"])
    self.pol["azz"] = np.abs(self.pol["zz"])

    if self.flag_normalize:

      self.pol["axx"], self.pol["ayy"], self.pol["azz"] \
        = self.normalize_max(self.pol["axx"], self.pol["ayy"], self.pol["azz"])

    self.pol["f"] = self.obs_ft_data["f"]

    print("> Polarisability calculated.")

  def _calculate_power_spectrum(self):

    pol_xx = dc(self.pol["xx"])
    pol_yy = dc(self.pol["zz"])
    pol_zz = dc(self.pol["yy"])

    psum = [ np.abs(xx)**2 + np.abs(yy)**2 + np.abs(zz)**2 for xx, yy, zz in zip(pol_xx, pol_yy, pol_zz) ]

    w_a = zip(self.pol["f"], psum)

    self.powsp["val"] = np.array([ (2./3.) * a for _, a in w_a ])
    
    self.powsp["aval"] = np.abs(self.powsp["val"])
    
    self.powsp["f"] = self.pol["f"] 
 
    self.powsp["int"] = np.trapz(self.powsp["aval"], self.powsp["f"])
    
    if self.flag_normalize:
      self.powsp["aval"] = self.normalize_max(self.powsp["aval"])

    if self.flag_locpeak:

      idx_peaks, info = self._locate_peaks(self.powsp["f"], self.powsp["aval"])

      self.powsp["f_peak"], self.powsp["val_peaks"] \
        = np.array([ self.powsp["f"][i] for i in idx_peaks]), np.array([ self.powsp["aval"][i] for i in idx_peaks ])

    print("> Power spectrum calculated.")

  def _calculate_abs_strength(self):
 
    pol_xx = dc(self.pol["xx"])
    pol_yy = dc(self.pol["zz"])
    pol_zz = dc(self.pol["yy"])
    
    trace = [ xx + yy + zz for xx, yy, zz in zip(pol_xx, pol_yy, pol_zz) ]
    
    w_a = zip(self.pol["f"], trace)
    
    self.stren["val"] = np.array([ (2.0/3.0) * (w / np.pi) * np.imag(a) for w, a in w_a ])
    # self.stren["val"] = np.array([ (4.0*np.pi/(3.0*137)) * w * np.imag(a) for w, a in w_a ])
    
    self.stren["aval"] = np.abs(self.stren["val"]) 
    
    self.stren["f"] = self.pol["f"]
    
    self.stren["int"] = np.trapz(self.stren["aval"], self.stren["f"])
    
    if self.flag_normalize:
      self.stren["aval"] = self.normalize_max(self.stren["aval"])
    
    print("> Absorption strength function calculated.")
    print("-> Integral over absorption strength function (number of electrons contributing to"\
          +" absorption): " + "{0:8.4f}".format(self.stren["int"]))
    print("   NOTE: this value should equal the number of valence electrons. It is only correctly")
    print("         calculated when damping (-d) is NOT included in this calculation.")

  def _plot_results(self):

    print("> Plotting data ...")
    
    if self.flag_normalize:
      norm = " (normalized by maximum)"
    else:
      norm = ""
    
    xlabel = r"Energy $E=\hbar\omega$ (eV)"
    xaxis  = self.pol["f"] * Observable.conv_2pi_per_au_to_ev
    
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14,9))
  
    obs_ax = np.abs(self.obs_ft_data["x"])
    obs_ay = np.abs(self.obs_ft_data["y"])
    obs_az = np.abs(self.obs_ft_data["z"])
    
    fld_ax = np.abs(self.field_ft_data["x"])
    fld_ay = np.abs(self.field_ft_data["y"])
    fld_az = np.abs(self.field_ft_data["z"])

    if self.flag_normalize:
      obs_ax = self.normalize_max(obs_ax)
      obs_ay = self.normalize_max(obs_ay)
      obs_az = self.normalize_max(obs_az)
      fld_az = self.normalize_max(fld_az)

    ax[0].plot(xaxis, obs_ax, "y", label=r"|$\mathcal{F}[d_x](\omega)|$", linewidth=1)
    ax[0].plot(xaxis, obs_ay, "g", label=r"|$\mathcal{F}[d_y](\omega)|$", linewidth=1)
    ax[0].plot(xaxis, obs_az, "b", label=r"|$\mathcal{F}[d_z](\omega)|$", linewidth=1)
    ax[0].plot(xaxis, fld_ax, "c", label=r"|$\mathcal{F}[E_z](\omega)|$", linewidth=1) 
    ax[0].plot(xaxis, fld_ay, "c", label=r"|$\mathcal{F}[E_z](\omega)|$", linewidth=1, linestyle="--") 
    ax[0].plot(xaxis, fld_az, "c", label=r"|$\mathcal{F}[E_z](\omega)|$", linewidth=1, linestyle=":") 
    ax[0].set_ylabel("Amplitude" + norm)
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[0].set_title(r"FT of dipole moment $\mathbf{d}(t)=-\mathbf{d}(0)+\int\,d\mathbf{r}\,\rho"\
                    +r"(\mathbf{r},t)\mathbf{r}~$ and electric field $\mathbf{E}(t)$", y=1.05)
    # ax[0].set_yticks(np.linspace(0, max(list(obs_ax) + list(obs_ay) + list(obs_az)), 10))
    #ax[0].set_yticks(np.linspace(0, max(list(self.obs_ft_data["abs"]) + list(obs_ax) + list(obs_ay) + list(obs_az))))
    ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[0].margins(0,0.01)
    ax[0].grid()
    
    px = [ x for x in self.pol["axx"] ] # [ abs(x.imag) for x in self.pol["xx"] ]
    py = [ y for y in self.pol["ayy"] ] # [ abs(y.imag) for y in self.pol["yy"] ]
    pz = [ z for z in self.pol["azz"] ] # [ abs(z.imag) for z in self.pol["zz"] ]
    
    ax[1].plot(xaxis, px, "r", label=r"$|\alpha_{xx}(\omega)|$", linewidth=2)
    ax[1].plot(xaxis, py, "g", label=r"$|\alpha_{yy}(\omega)|$", linewidth=2)
    ax[1].plot(xaxis, pz, "b", label=r"$|\alpha_{zz}(\omega)|$", linewidth=2)
    
    ax[1].set_ylabel("Amplitude" + norm)
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[1].set_title(r"Polarisability tensor $\alpha_{ij}(\omega)=\mathcal{F}[d_i](\omega)"\
                    +r"/\mathcal{F}[E_j](\omega)$", y=1.05)
    ax[1].set_yticks(np.linspace(0, max(list(px)+list(py)+list(pz)), 10))
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[1].margins(0,0.01)
    ax[1].grid()
    
    pow_scaled  = self.powsp["val"] / max(self.powsp["val"]) * max(self.stren["aval"])

    if self.flag_locpeak:
      powp_scaled = self.powsp["val_peaks"] / max(self.powsp["val_peaks"]) * max(self.stren["aval"])
 
    ax[2].plot(xaxis, self.stren["aval"], "r", linewidth=2, label="|S|")
    ax[2].plot(xaxis, pow_scaled, "blue", linewidth=2, label="|P|~max(|S|)")

    if self.flag_locpeak:
      ax[2].scatter(self.powsp["f_peak"]*Observable.conv_2pi_per_au_to_ev, powp_scaled, color="royalblue", \
                    marker="x", label="Peaks: |P|", zorder=10)

    ax[2].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[2].set_xlabel(xlabel)
    ax[2].set_ylabel("Amplitude" + norm)
    ax[2].set_title(r"Absorption strength function $|S(\omega)|=|\frac{2\omega}{\pi}\mathrm{Im}"\
                    +r"\left(\frac{1}{3}\mathrm{Tr}[\alpha(\omega)]\right)|$ and power spectrum "\
                    +r"$P(\omega)=\frac{2}{3}\sum_{\alpha\beta}\mathcal{F}[p_\alpha^\beta](\omega)$", y=1.05)
    ax[2].set_yticks(np.linspace(0, max(self.stren["aval"]), 10))
    ax[2].set_xticks(np.linspace(xaxis[0], xaxis[-1], 15))
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[2].margins(0,0.01)
    ax[2].grid()
    
    fig.subplots_adjust(hspace=0.18)
    plt.setp([ax[0].get_xticklabels(), ax[1].get_xticklabels()], visible=False)
    plt.subplots_adjust(left=0.05, right=0.89, bottom=0.05, top=0.95) #, top=1.1, wspace=0.1, hspace=0)

class ElectronicCurrent(VectorObservable):
  """
  Class for any electronic current data from which we can calculate the conductivity etc.
  """

  #TBD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
  desc = "electronic_current"
  conv_obs_au2si = 1.0 # 6.623618237510e-3 # A for electric field
  conv_fld_au2si = 1.0 # 5.14220674763e11 # V/m for electric field amplitude which is read in
  #TBD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

  def __init__(self, **kwargs):

    VectorObservable.__init__(self, conv_obs_au2si=ElectronicDipole.conv_obs_au2si, \
      conv_fld_au2si=ElectronicDipole.conv_fld_au2si, **kwargs)

    self.con   = { "f": [], "xx": [], "yy": [], "zz": [], "axx": [], "ayy": [], "azz": [] }
    self.conf  = { "f": [], "xx": [], "yy": [], "zz": [], "axx": [], "ayy": [], "azz": [] }
    self.dielf = { "f": [], "val": [], "aval": [], "int": 0, "fsum": 0 }

    self._calculate_conductivity()
    self._calculate_dielectric_fn()

    if self.flag_plot:
      self._plot_results()

    if self.out_file_name is not None:
      self._write_results()

  def _write_results(self):

    en_vals = self.con["f"] * Observable.conv_2pi_per_au_to_ev

    desc = "current_ft"
    comm = "# FT OF CURRENT \n # UNITS: [ENER] = eV | [FT_CUR] = TBD \n # ENERGY | |C_X| | |C_Y| | |C_Z| \n"

    vals = [ np.abs(self.obs_ft_data[vv]) for vv in ["x", "y", "z"] ]

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *vals)

    desc = "conductivity"
    comm = "# CONDUCTIVITY \n # UNITS: [ENER] = eV | [CON] = TBD \n # ENERGY | |S_XX| | |S_YY| | |S_ZZ| \n"

    vals = [ self.con[vv] for vv in ["axx", "ayy", "azz"] ]

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *vals)

    desc = "diel_fn"
    comm = "# DIELECTRIC FUNCTION \n # UNITS: [ENER] = eV | [DIEL] = TBD \n # ENERGY | REAL | IMAG \n"

    vals = [ self.dielf["val"].real, self.dielf["val"].imag ]

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *vals)

  def _calculate_conductivity(self):

    obs_ft = dc(self.obs_ft_data)
    fld_ft = dc(self.field_ft_data)

    self.con["xx"] = [ p / e if abs(e) > 1.0e-7 else 0 for p, e in zip(obs_ft["x"], fld_ft["x"]) ]
    self.con["yy"] = [ p / e if abs(e) > 1.0e-7 else 0 for p, e in zip(obs_ft["y"], fld_ft["y"]) ]
    self.con["zz"] = [ p / e if abs(e) > 1.0e-7 else 0 for p, e in zip(obs_ft["z"], fld_ft["z"]) ]

    self.con["axx"] = np.abs(self.con["xx"])
    self.con["ayy"] = np.abs(self.con["yy"])
    self.con["azz"] = np.abs(self.con["zz"])

    if self.flag_normalize:

      self.con["axx"], self.con["ayy"], self.con["azz"] \
        = self.normalize_max(self.con["axx"], self.con["ayy"], self.con["azz"])

    self.con["f"] = self.obs_ft_data["f"]

    # remove artificial w=0 peak
    t = self.obs_data["t"]
    dt = t[-1]-t[-2]
    jxf = self.obs_data["x"][-1] * np.ones(len(t))
    jyf = self.obs_data["y"][-1] * np.ones(len(t))
    jzf = self.obs_data["z"][-1] * np.ones(len(t))
    jxf[1] = self.obs_data["x"][0]
    jyf[1] = self.obs_data["y"][0]
    jzf[1] = self.obs_data["z"][0]

    w, Fjxf, Fjyf, Fjzf, _ = calculate_fft_vec(jxf, jyf, jzf, dt)
  
    self.conf["xx"] = [ p/e if abs(e) > 1e-7 else 0 for p, e in zip(Fjxf, fld_ft["x"]) ] 
    self.conf["yy"] = [ p/e if abs(e) > 1e-7 else 0 for p, e in zip(Fjyf, fld_ft["y"]) ] 
    self.conf["zz"] = [ p/e if abs(e) > 1e-7 else 0 for p, e in zip(Fjzf, fld_ft["z"]) ] 

    print("> Conductivity calculated.")

  def _calculate_dielectric_fn(self):

    con_xx = dc(self.con["xx"])
    con_yy = dc(self.con["zz"])
    con_zz = dc(self.con["yy"])

    trace = [ xx + yy + zz for xx, yy, zz in zip(con_xx, con_yy, con_zz) ]
   
    w_s = zip(self.con["f"], trace)
    
    self.dielf["val"] = np.array([ 1.0 + 4.0 * np.pi * 1.0j * s / w if np.abs(w) > 1.e-10 else 0 for w, s in w_s ])
   
    conf_xx = dc(self.conf["xx"])
    conf_yy = dc(self.conf["zz"])
    conf_zz = dc(self.conf["yy"])

    tracef = [ xx + yy + zz for xx, yy, zz in zip(conf_xx, conf_yy, conf_zz) ]
    w_s_f = zip(self.con["f"], tracef)
    dielff = np.array([ 1.0 + 4.0 * np.pi * 1.0j * s / w if np.abs(w) > 1.e-10 else 0 for w, s in w_s_f ])

    PEAK_HACK = False

    if PEAK_HACK:
      l_mod = len(dielff)
      l_dat = len(self.dielf["val"])
      if l_mod < l_dat:
        zz = np.zeros(l_dat)
        zz[0:l_mod] = dielff[0:l_mod]
        zz[l_mod:] = 0
        dielff = zz
      self.dielf["val"] = self.dielf["val"] - dielff

    self.dielf["aval"] = np.abs(self.dielf["val"]) 
    
    self.dielf["f"] = self.con["f"]
    
    self.dielf["int"] = np.trapz(self.dielf["aval"], self.dielf["f"])
   
    fsum_p = np.array([ w * eps.imag for (w, eps) in zip(self.dielf["f"], self.dielf["val"]) ])
    self.dielf["fsum"] = np.trapz(fsum_p, self.dielf["f"])

    fsum_w = []
    for i, fsi in enumerate(fsum_p):
      fsum_w.append(np.trapz(fsum_p[:i+1], self.dielf["f"][:i+1]))

    with open("check_fsum_eps.dat", "w") as f:
      for i, wi in enumerate(self.dielf["f"]):
        f.write("{0} {1}\n".format(wi, fsum_w[i]))

    print("-> Integral over imag(epsilon)*frequency (2 * pi^2 * number of electrons contributing to"\
          +" absorption/unit cell volume): " + "{0:8.4f}".format(self.dielf["fsum"]))
 
    print("> Dielectric function calculated.")

  def _plot_results(self):

    print("> Plotting data ...")
    
    if self.flag_normalize:
      norm = " (normalized by maximum)"
    else:
      norm = ""
    
    xlabel = r"Energy $E=\hbar\omega$ (eV)"
    xaxis  = self.con["f"] * Observable.conv_2pi_per_au_to_ev
    
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14,9))
  
    obs_ax = np.abs(self.obs_ft_data["x"])
    obs_ay = np.abs(self.obs_ft_data["y"])
    obs_az = np.abs(self.obs_ft_data["z"])
    
    fld_ax = np.abs(self.field_ft_data["x"])
    fld_ay = np.abs(self.field_ft_data["y"])
    fld_az = np.abs(self.field_ft_data["z"])

    if self.flag_normalize:
      obs_ax = self.normalize_max(obs_ax)
      obs_ay = self.normalize_max(obs_ay)
      obs_az = self.normalize_max(obs_az)
      fld_az = self.normalize_max(fld_az)

    ax[0].plot(xaxis, obs_ax, "y", label=r"|$\mathcal{F}[I_x](\omega)|$", linewidth=1)
    ax[0].plot(xaxis, obs_ay, "g", label=r"|$\mathcal{F}[I_y](\omega)|$", linewidth=1)
    ax[0].plot(xaxis, obs_az, "b", label=r"|$\mathcal{F}[I_z](\omega)|$", linewidth=1)
#   ax[0].plot(xaxis, fld_az, "c", label=r"|$\mathcal{F}[E_z](\omega)|$", linewidth=2) #only z-component since all are equal!
    ax[0].set_ylabel("Amplitude" + norm)
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[0].set_title(r"FT of electronic current $\mathbf{I}(t)=-\frac{1}{V}\int\,d\mathbf{r}\,\mathbf{j}"\
                    +r"(\mathbf{r},t)\mathbf{r}~$ and electric field $\mathbf{E}(t)$", y=1.05)
    # ax[0].set_yticks(np.linspace(0, max(list(obs_ax) + list(obs_ay) + list(obs_az)), 10))
    #ax[0].set_yticks(np.linspace(0, max(list(self.obs_ft_data["abs"]) + list(obs_ax) + list(obs_ay) + list(obs_az))))
    ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[0].margins(0,0.01)
    ax[0].grid()
    
    px = [ x for x in self.con["axx"] ] # [ abs(x.imag) for x in self.pol["xx"] ]
    py = [ y for y in self.con["ayy"] ] # [ abs(y.imag) for y in self.pol["yy"] ]
    pz = [ z for z in self.con["azz"] ] # [ abs(z.imag) for z in self.pol["zz"] ]
    
    ax[1].plot(xaxis, px, "r", label=r"$|\sigma_{xx}(\omega)|$", linewidth=2)
    ax[1].plot(xaxis, py, "g", label=r"$|\sigma_{yy}(\omega)|$", linewidth=2)
    ax[1].plot(xaxis, pz, "b", label=r"$|\sigma_{zz}(\omega)|$", linewidth=2)
    
    ax[1].set_ylabel("Amplitude" + norm)
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[1].set_title(r"Conductivity tensor $\sigma_{ij}(\omega)=\mathcal{F}[I_i](\omega)"\
                    +r"/\mathcal{F}[E_j](\omega)$", y=1.05)
    ax[1].set_yticks(np.linspace(0, max(list(px)+list(py)+list(pz)), 10))
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[1].margins(0,0.01)
    ax[1].grid()
    
    ax[2].plot(xaxis, self.dielf["val"].real, "red", linewidth=2, label=r"Re($\epsilon$)")
    ax[2].plot(xaxis, self.dielf["val"].imag, "blue", linewidth=2, label=r"Im($\epsilon$)")
    ax[2].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[2].set_xlabel(xlabel)
    ax[2].set_ylabel("Amplitude" + norm)
    ax[2].set_title(r"Dielectric function $\epsilon(\omega)=1+\frac{i4\pi}{\omega}\mathrm{Tr}[\sigma(\omega)]$", y=1.05)
    #ax[2].set_yticks(np.linspace(0, max(self.stren["aval"]), 10))
    ax[2].set_xticks(np.linspace(xaxis[0], xaxis[-1], 15))
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[2].margins(0,0.01)
    ax[2].grid()
    
    fig.subplots_adjust(hspace=0.18)
    plt.setp([ax[0].get_xticklabels(), ax[1].get_xticklabels()], visible=False)
    plt.subplots_adjust(left=0.05, right=0.89, bottom=0.05, top=0.95) #, top=1.1, wspace=0.1, hspace=0)

class MagneticMoment(VectorObservable):
  """
  Class for any electronic current data from which we can calculate the conductivity etc.
  """

  #TBD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
  desc = "magnetic_moment"
  conv_obs_au2si = 1.0
  conv_fld_au2si = 1.0
  #TBD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

  def __init__(self, **kwargs):

    VectorObservable.__init__(self, conv_obs_au2si=ElectronicDipole.conv_obs_au2si, \
      conv_fld_au2si=ElectronicDipole.conv_fld_au2si, **kwargs)

    self.rtens = { "f": [], "xx": [], "yy": [], "zz": [], "axx": [], "ayy": [], "azz": [] }
    self.rstr  = { "f": [], "val": [], "aval": [], "int": 0 }

    self._calculate_rot_response_tensor()
    self._calculate_rot_strength()

    if self.flag_plot:
      self._plot_results()

    if self.out_file_name is not None:
      self._write_results()

  def _write_results(self):

    en_vals = self.rtens["f"] * Observable.conv_2pi_per_au_to_ev

    desc = "magmom_ft"
    comm = "# FT OF MAGNETIC MOMENT \n # UNITS: [ENER] = eV | [FT_CUR] = TBD \n # ENERGY | |C_X| | |C_Y| | |C_Z| \n"

    vals = [ np.abs(self.obs_ft_data[vv]) for vv in ["x", "y", "z"] ]

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *vals)

    desc = "rotres_tensor"
    comm = "# ROTATORY RESPONSE TENSOR\n # UNITS: [ENER] = eV | [CON] = TBD \n # ENERGY | |S_XX| | |S_YY| | |S_ZZ| \n"

    vals = [ self.rtens[vv] for vv in ["axx", "ayy", "azz"] ]

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *vals)

    desc = "rot_stren"
    comm = "# ROTATORY STRENGTH\n # UNITS: [ENER] = eV | [DIEL] = TBD \n # ENERGY | ABS \n"

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *[self.rstr["val"]])

  def _calculate_rot_response_tensor(self):

    obs_ft = dc(self.obs_ft_data)
    fld_ft = dc(self.field_ft_data)

    self.rtens["xx"] = [ p / e if abs(e) > 1.0e-7 else 0 for p, e in zip(obs_ft["x"], fld_ft["x"]) ]
    self.rtens["yy"] = [ p / e if abs(e) > 1.0e-7 else 0 for p, e in zip(obs_ft["y"], fld_ft["y"]) ]
    self.rtens["zz"] = [ p / e if abs(e) > 1.0e-7 else 0 for p, e in zip(obs_ft["z"], fld_ft["z"]) ]

    self.rtens["axx"] = np.abs(self.rtens["xx"])
    self.rtens["ayy"] = np.abs(self.rtens["yy"])
    self.rtens["azz"] = np.abs(self.rtens["zz"])

    if self.flag_normalize:

      self.rtens["axx"], self.rtens["ayy"], self.rtens["azz"] \
        = self.normalize_max(self.rtens["axx"], self.rtens["ayy"], self.rtens["azz"])

    self.rtens["f"] = self.obs_ft_data["f"]

    print("> Rotatory response tensor calculated.")

  def _calculate_rot_strength(self):

    rtens_xx = dc(self.rtens["xx"])
    rtens_yy = dc(self.rtens["zz"])
    rtens_zz = dc(self.rtens["yy"])

    trace = [ xx + yy + zz for xx, yy, zz in zip(rtens_xx, rtens_yy, rtens_zz) ]
    
    self.rstr["val"] = np.real(trace) * 2.0 * 137.0 / np.pi
    
    self.rstr["aval"] = np.abs(self.rstr["val"]) 
    
    self.rstr["f"] = self.rtens["f"]
    
    self.rstr["int"] = np.trapz(self.rstr["aval"], self.rstr["f"])
    
    print("> Rotatory strength calculated.")

  def _plot_results(self):

    print("> Plotting data ...")
    
    if self.flag_normalize:
      norm = " (normalized by maximum)"
    else:
      norm = ""
    
    xlabel = r"Energy $E=\hbar\omega$ (eV)"
    xaxis  = self.rtens["f"] * Observable.conv_2pi_per_au_to_ev
    
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(14,9))
  
    obs_ax = np.abs(self.obs_ft_data["x"])
    obs_ay = np.abs(self.obs_ft_data["y"])
    obs_az = np.abs(self.obs_ft_data["z"])
    
    fld_ax = np.abs(self.field_ft_data["x"])
    fld_ay = np.abs(self.field_ft_data["y"])
    fld_az = np.abs(self.field_ft_data["z"])

    if self.flag_normalize:
      obs_ax = self.normalize_max(obs_ax)
      obs_ay = self.normalize_max(obs_ay)
      obs_az = self.normalize_max(obs_az)
      fld_az = self.normalize_max(fld_az)

    ax[0].plot(xaxis, obs_ax, "y", label=r"|$\mathcal{F}[m_x](\omega)|$", linewidth=1)
    ax[0].plot(xaxis, obs_ay, "g", label=r"|$\mathcal{F}[m_y](\omega)|$", linewidth=1)
    ax[0].plot(xaxis, obs_az, "b", label=r"|$\mathcal{F}[m_z](\omega)|$", linewidth=1)
#   ax[0].plot(xaxis, fld_az, "c", label=r"|$\mathcal{F}[E_z](\omega)|$", linewidth=2) #only z-component since all are equal!
    ax[0].set_ylabel("Amplitude" + norm)
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[0].set_title(r"FT of magnetic moment $\mathbf{m}(t)$ and electric field $\mathbf{E}(t)$", y=1.05)
    # ax[0].set_yticks(np.linspace(0, max(list(obs_ax) + list(obs_ay) + list(obs_az)), 10))
    #ax[0].set_yticks(np.linspace(0, max(list(self.obs_ft_data["abs"]) + list(obs_ax) + list(obs_ay) + list(obs_az))))
    ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[0].margins(0,0.01)
    ax[0].grid()
    
    px = [ x for x in self.rtens["axx"] ] # [ abs(x.imag) for x in self.pol["xx"] ]
    py = [ y for y in self.rtens["ayy"] ] # [ abs(y.imag) for y in self.pol["yy"] ]
    pz = [ z for z in self.rtens["azz"] ] # [ abs(z.imag) for z in self.pol["zz"] ]
    
    ax[1].plot(xaxis, px, "r", label=r"$|\beta_{xx}(\omega)|$", linewidth=2)
    ax[1].plot(xaxis, py, "g", label=r"$|\beta_{yy}(\omega)|$", linewidth=2)
    ax[1].plot(xaxis, pz, "b", label=r"$|\beta_{zz}(\omega)|$", linewidth=2)
    
    ax[1].set_ylabel("Amplitude" + norm)
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[1].set_title(r"Rotatory strength tensor $\beta_{ij}(\omega)=\mathcal{F}[m_i](\omega)"\
                    +r"/\mathcal{F}[E_j](\omega)$", y=1.05)
    ax[1].set_yticks(np.linspace(0, max(list(px)+list(py)+list(pz)), 10))
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[1].margins(0,0.01)
    ax[1].grid()
    
    ax[2].plot(xaxis, self.rstr["val"], "red", linewidth=2, label=r"R($\omega$)")
    ax[2].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[2].set_xlabel(xlabel)
    ax[2].set_ylabel("Amplitude" + norm)
    ax[2].set_title(r"Rotatory response strength $R(\omega)=\frac{1}{\pi}\mathrm{Re}\left[\mathrm{Tr}[\beta(\omega)]\right]$", y=1.05)
    #ax[2].set_yticks(np.linspace(0, max(self.stren["aval"]), 10))
    ax[2].set_xticks(np.linspace(xaxis[0], xaxis[-1], 15))
    ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[2].margins(0,0.01)
    ax[2].grid()
    
    fig.subplots_adjust(hspace=0.18)
    plt.setp([ax[0].get_xticklabels(), ax[1].get_xticklabels()], visible=False)
    plt.subplots_adjust(left=0.05, right=0.89, bottom=0.05, top=0.95) #, top=1.1, wspace=0.1, hspace=0)

class HHGSpectrum(VectorObservable):
  """
  Class for any HHG data from which we can calculate the HHG spectrum
  """

  #TBD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
  desc = "hhg_spectrum"
  conv_obs_au2si = 1.0
  conv_fld_au2si = 1.0
  #TBD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

  def __init__(self, input_file=None, **kwargs):

    VectorObservable.__init__(self, conv_obs_au2si=HHGSpectrum.conv_obs_au2si, \
      conv_fld_au2si=HHGSpectrum.conv_fld_au2si, **kwargs)

    self.n_elec   = -1
    self.hhg_spec = { "f": [], "val": [], "int": 0 }

    self.mode = None # either 'current', 'dipole' or 'grad', based on input file

    self._get_number_of_electrons_from_file(input_file)
    self._calculate_hhg_spectrum()

    if self.flag_plot:
      self._plot_results()

    if self.out_file_name is not None:
      self._write_results()

  def _get_number_of_electrons_from_file(self, infile):

    i = 0

    if infile.endswith(".current.dat"):
      self.mode = "current"
      return
    elif infile.endswith(".dipole.dat"):
      self.mode = "dipole"
      return
    else:
      self.mode = "grad"

    with open(infile, "r") as f:
     
      while i < 5:
        ln = f.readline() 

        if ln.rfind("ELECTRONS") > -1:
          self.n_elec = float(ln.split()[-1])
          print("> Found number of electrons in field file: {0}".format(self.n_elec))
          return
   
    raise Exception("ERROR: could not retrieve number of electrons from HHG gradient potential file!")

  def _write_results(self):

    en_vals = self.hhg_spec["f"] * Observable.conv_2pi_per_au_to_ev

    if self.mode == "grad":
      desc = "hhg_pot_grad_ft"
      comm = "# FT OF HHG PART \n # UNITS: [ENER] = eV | [FT_POT] = TBD \n # ENERGY | |P_X| | |P_Y| | |P_Z| \n"
    elif self.mode == "dipole":
      desc = "dipole_ft"
      comm = "# FT OF DIPOLE \n # UNITS: [ENER] = eV | [FT_POT] = TBD \n # ENERGY | |D_X| | |D_Y| | |D_Z| \n"
    else:
      desc = "current_ft"
      comm = "# FT OF CURRENT \n # UNITS: [ENER] = eV | [FT_POT] = TBD \n # ENERGY | |C_X| | |C_Y| | |C_Z| \n"

    vals = [ np.abs(self.obs_ft_data[vv]) for vv in ["x", "y", "z"] ]

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *vals)

    desc = "hhg_spectrum"
    comm = "# HHG SPECTRUM\n # UNITS: [ENER] = eV | [HHG] = TBD \n # ENERGY | ABS \n"

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *[self.hhg_spec["val"]])

    desc = "field_ft"
    comm = "# EXTERNAL FIELD FT\n # UNITS: [ENER] = eV | [F_i] = TBD \n # ENERGY | |F_x| | |F_y| | |F_z| \n"

    vals = [ np.abs(self.field_ft_data[vv]) for vv in ["x", "y", "z"] ]

    self._write_output_data(self.out_file_name, desc, comm, en_vals, *vals)

  def _calculate_hhg_spectrum(self):

    if self.n_elec < 0.0 and self.mode == "grad":
      raise Exception("ERROR: number of electrons cannot be negative. Check input file.")

    obs_ft = dc(self.obs_ft_data)

    self.hhg_spec["f"] = self.obs_ft_data["f"]

    obs_ft_xyz = np.array(list(zip(obs_ft["x"], obs_ft["y"], obs_ft["z"])))

    if self.mode == "grad":

      fld_ft = self.field_ft_data
      field_xyz  = np.array(list(zip(fld_ft["x"], fld_ft["y"], fld_ft["z"])))

      if self.field.settings["gauge"] == "velocity":
        raise Exception(">Error: this can only be done with an electric field atm (length gauge)")

      hhg_calc_ft = obs_ft_xyz + field_xyz * self.n_elec

    elif self.mode == "current":

      # Need to determine field polarization vector first      
      amp = self.field.settings["amplitude"]
      n_amp = amp / np.linalg.norm(amp)

      hhg_calc_ft = []
      for wi, ji in zip(self.hhg_spec["f"], obs_ft_xyz):
        hhg_calc_ft.append(np.dot(ji, n_amp)*wi) # omega * J(omega) \dot n(field)

    elif self.mode == "dipole":

      # Need to determine field polarization vector first      
      amp = self.field.settings["amplitude"]
      n_amp = amp / np.linalg.norm(amp)

      hhg_calc_ft = []
      for wi, ji in zip(self.hhg_spec["f"], obs_ft_xyz):
        hhg_calc_ft.append(np.dot(ji, n_amp)*wi**2) # omega * J(omega) \dot n(field)

    self.hhg_spec["val"] = [ np.linalg.norm(sum_i)**2 for sum_i in hhg_calc_ft ]

    if self.flag_normalize:

      self.hhg_spec["val"] \
        = self.normalize_max(self.hhg_spec["val"])

    print("> HHG spectrum calculated.")

  def _plot_results(self):

    print("> Plotting data ...")
    
    if self.flag_normalize:
      norm = " (normalized by maximum)"
    else:
      norm = ""
    
    xlabel = r"Energy $E=\hbar\omega$ (eV)"
    xaxis  = self.hhg_spec["f"] * Observable.conv_2pi_per_au_to_ev
    
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(14,9))
  
    obs_ax = np.abs(self.obs_ft_data["x"])
    obs_ay = np.abs(self.obs_ft_data["y"])
    obs_az = np.abs(self.obs_ft_data["z"])
    
    fld_ax = np.abs(self.field_ft_data["x"])
    fld_ay = np.abs(self.field_ft_data["y"])
    fld_az = np.abs(self.field_ft_data["z"])

    if self.flag_normalize:
      obs_ax = self.normalize_max(obs_ax)
      obs_ay = self.normalize_max(obs_ay)
      obs_az = self.normalize_max(obs_az)
      fld_az = self.normalize_max(fld_az)

    ax[0].plot(xaxis, obs_ax, "y", label=r"|$\mathcal{F}[pgrad_x](\omega)|$", linewidth=1)
    ax[0].plot(xaxis, obs_ay, "g", label=r"|$\mathcal{F}[pgrad_y](\omega)|$", linewidth=1)
    ax[0].plot(xaxis, obs_az, "b", label=r"|$\mathcal{F}[pgrad_z](\omega)|$", linewidth=1)
    ax[0].set_ylabel("Amplitude" + norm)
    ax[0].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax[0].set_title(r"FT of HHG potential gradient part and electric field $\mathbf{E}(t)$", y=1.05)
    # ax[0].set_yticks(np.linspace(0, max(list(obs_ax) + list(obs_ay) + list(obs_az)), 10))
    #ax[0].set_yticks(np.linspace(0, max(list(self.obs_ft_data["abs"]) + list(obs_ax) + list(obs_ay) + list(obs_az))))
    ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[0].margins(0,0.01)
    ax[0].grid()
    
    ax[1].plot(xaxis, self.hhg_spec["val"], "r", label=r"$|\mathrm{HHG}(\omega)|$", linewidth=2)
    
    ax[1].set_ylabel("Amplitude" + norm)
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.,1.031))
  
    if self.mode == "grad":
      ax[1].set_title(r"HHG spectrum HHG$(\omega)=\left|\mathcal{F}\left[\int\rho\nabla V_\mathrm{ext}d^3\mathbf{r}\right](\omega)"\
                      +r"+N_e\mathcal{F}[\mathbf{E}](\omega)\right|^2$", y=1.05)
    else:
      ax[1].set_title(r"HHG spectrum HHG$(\omega)=\left|\omega\mathcal{F}[\mathbf{I}](\omega)\cdot \mathbf{n}_\mathrm{field}\right|^2$", y=1.05)

    ax[1].set_yticks(np.linspace(0, max(list(self.hhg_spec["val"])), 10))
    ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax[1].margins(0,0.01)
    ax[1].grid()
    
   
    fig.subplots_adjust(hspace=0.18)
    plt.setp([ax[0].get_xticklabels()], visible=False)
    plt.subplots_adjust(left=0.05, right=0.89, bottom=0.05, top=0.95) #, top=1.1, wspace=0.1, hspace=0)

class MyParser(argparse.ArgumentParser):

  def format_help(self):
    return("\n This script calculates the polarisability (tensor) and the absorption\n"
           " strength function OR the conductivity (tensor) and the dielectric function OR the rotatory \n"
           " response (tensor) and the rotatory strength. It can plot the data and/or write it to a file.\n\n"
           " Any of the above calculations needs: \n"
           "  * 3 files for response observables, corresponding to 3 individual calculations, each \n"
           "    for a perturbation along one cartesian axis\n"
           "  * 1 OR 3 files for the corresponding electric field. If only a single file is given, \n"
           "    the same field is applied in all individual directions. You can specify that this \n"
           "    script performs calculations with analytically given field shapes as defined in the \n"
           "    rt-tddft module. This results in less noisy output. Nevertheless, one field file must \n"
           "    be read in to obtain the field parameters given in the header.\n\n"
           " Inputs: (S: string, I: int, F: float)\n\n"
           "         '-pol' S S S     | Output files containing electronic dipole time series of type \n"
           "                            DESCRIPTOR.rt-tddft.dipole.dat for each direction x, y, z in this order. \n"
           "                            This will yield a calculation of polarisability, absorption strength, \n"
           "                            and power spectrum. \n"
           "         '-con' S S S     | Output files containing electronic current time series of type \n"
           "                            DESCRIPTOR.rt-tddft.current.dat for each direction x, y, z in this order. \n"
           "                            This will yield a calculation of conductivity and dielectric function. \n"
           "         '-rsp' S S S     | Output files containing electronic magnetic moment time series of type \n"
           "                            DESCRIPTOR.rt-tddft.magmom.dat for each direction x, y, z in this order. \n"
           "                            This will yield a calculation of rotatory response. \n"
           "         '-hhg' S         | Output file containing EITHER hhg potential gradient part time series of type \n"
           "                            DESCRIPTOR.rt-tddft.hhg.dat (only possible for non-periodic systems atm. i.e. \n"
           "                            length gauge) OR the electronic current of type DESCRIPTOR.rt-tddft.current.dat \n"
           "                            (possible for both cases, velocity gauge must be used). \n"
           "                            Both possibilities correspond to different formulas for a HHG spectrum.    \n"
           "                            This will yield a HHG spectrum calculation. \n"
           "                            In any case, this needs ONE observable and ONE field input file.\n"
           "         '-field' S (S S) | Output file/s containing the applied external field time series of type \n"
           "                            DESCRIPTOR.rt-tddft.ext-field.dat for each direction x, y, z in this order.\n"
           "                            Only one file can be provided, the corresponding field will then be applied \n"
           "                            to all cartesian directions for the calculations.\n"
           "         '-fa'            | If given, the FT of the external field will be computed analytically from \n"
           "                            the given type and parameters in the input file. The type must be defined \n"
           "                            in this program and at least one field file must be given via '-field'\n"
           "                            to obtain the parameters [OPTIONAL, default=True].\n"
           "         '-tmin' F        | Minimum time for evaluation of time series [OPTIONAL, default=0]\n"
           "         '-tmax' F        | Maximum time for evaluation of time series [OPTIONAL, default=inf]\n"
           "         '-tshift' F      | Temporal translation of input data, e.g. response to delayed external field\n"
           "                            delta kick. This can be IMPORTANT to consider phase information, especially\n"
           "                            for complex observables like dielectric function and rotatory strength\n"
           "                            [OPTIONAL, default=0]\n"
           "         '-fmin' F        | Minimum frequency for evaluation of polarisability and absorption strength \n"
           "                            [OPTIONAL, default = 0, in eV] \n"
           "         '-fmax' F        | Maximum frequency for evaluation of polarisability and absorption strength \n"
           "                            [OPTIONAL, default = f_max, in eV]\n"
           "         '-p'             | Plot results [OPTIONAL]\n"
           "         '-w' (S)         | Write results to file, optionally with file prefix [OPTIONAL]\n"
           "         '-i' (F)         | Interpolate FT of data to obtain better resolution. Optionally, the multplicative \n"
           "                            factor (1,N] may be provided to scale up interpolation points [OPTIONAL]\n"
           "         '-ip'            | Plot results of interpolation when requested [OPTIONAL]\n"
           "         '-n'             | If given, data is normalized for output [OPTIONAL, default=False]\n"
           "         '-z' I           | Number of additional zeros in the input signal (zero-padding) "
                                        "[OPTIONAL, default=0]\n"
           "         '-d' F S (F)     | If given, the dipole signal will be damped from t0=F via \n"
           "                            f(t,t0) = 1-3*((t-t0)/(tmax-t0))^2+2*((t-t0)/(tmax-t0))^3 (S=poly), \n"
           "                            f(t,t0) = exp(-t/( (F) in fs)), or \n"
           "                            f(t,t0) = sin(pi*(t-t_tot/2)/t_tot)^2 \n"
           "                            before its FT is calculated; this can reduce wiggling [OPTIONAL]\n"
           "         '-lp' F F        | Identify peaks in spectrum via numpy.find_peaks. 1st argument is minimum peak\n"
           "                            width (in eV), 2nd argument is minimum peak height (% of max. spectrum aplitude).\n"
           "                            An additional file containing this information will be written [OPTIONAL]\n"
           "         '-ft-pade' N F F | Perform all Fourier transforms via Pade approximation to increase resolution.\n"
           "                            First argument is the number of frequency bins, second/third argument min/max frequency.\n\n" 
           "       Detrending is used in FT's to get rid of the 0 frequency peak.\n\n"
           " Example: The following input will perform an absorption spectrum calculation based on the\n"
           "          electronic dipole response, use an analytically defined field, and plot the results:\n\n"
           "          eval_tddft.py -pol x.rt-tddft.dipole.dat y.rt-tddft.dipole.dat z.rt-tddft.dipole.dat\n"
           "                        -field x.rt-tddft.ext-field.dat -fa -p \n\n")

#########################################################################################
# Functions #############################################################################
#########################################################################################

def set_external_field(field_x, field_y, field_z, flag_copy_ref=False, flag_analytic_field=False):

  """
    field_i is an object of the FieldData class and contains the corresponding input fields
    -> Returns a FieldData object containing the desired x,y,z components
  """

  if field_x is not None and (field_y is None and field_z is None):

    ref_field = dc(field_x)

    if flag_copy_ref:
      print("> Only one input field was provided. This field be be assumed as reference field"\
            +" in this case and the remainig components will be derived from it.")

      ref_amp = dc(ref_field.settings["amplitude"])

      set_i = None
      amp_i = 0.0
      for xi, i in zip(ref_amp, ["x","y","z"]):
        if np.abs(xi) > 1e-5:
          print("> Reference field seems to be "+i+"-polarized. Copying components.") 
          set_i = i
          amp_i = xi
          break

      ref_field.settings["amplitude"] = np.ones(3)*amp_i 

      if not flag_analytic_field:     
 
        for i in ["x","y","z"]:
          if not i == set_i: ref_field.data[i] = ref_field.data[set_i]

    else:
      print("Only one input field was provided. This field will be used for calculations.")

    return ref_field

  elif field_x is not None and field_y is not None and field_z is not None:

    ref_field = dc(field_x)

    if not flag_analytic_field:     
      ref_field.data["y"] = dc(field_y.data["y"])
      ref_field.data["z"] = dc(field_z.data["z"])

    # could check for equal settings but too lazy now

    ref_field.settings["amplitude"] = np.array([field_x.settings["amplitude"][0], \
      field_y.settings["amplitude"][1], field_z.settings["amplitude"][2]])

    return ref_field

  else:
    raise Exception("> Provide either 1 or 3 defined input field objects!")

def calculate_fft_vec(a_x, a_y, a_z, dt, detrend=False, add_zeros=0, normalize=None, conjg=False):

  """
    Script to calculate the FFT of an input array representing a time series with x,y,z components, consindering options
      1. Detrend the data
      2. Zero-pad the data
      3. Normalize the data
    @Joscha Hekele
  
    Parameters:
    ----------
      a_x/y/z:    real/complex
                  input array of time signals in x/y/z direction
      dt:         real
                  time-step between successive values
      detrend:    boolean, optional
                  Determines if data is detrended before FFT is performed
      add_zeros:  int, optional
                  Number of zeros added to each axis (zero-padding)
      normalize:  string, optional
                  Determines if fourier transform is normalized by maximum value ("max") or maximum value of absolute ("abs")
  
    Returns:
    -------
      complex arrays f, ft_x, ft_y, ft_z, ft_abs where
        f:      frequency
        ft_x:   ft of x component of signal
        ft_y:   ft of y component of signal
        ft_z:   ft of z component of signal
        ft_abs: ft the of signal's absolute
                
  """ 

  from scipy import signal as sig
  import numpy as np


  if len(a_x) != len(a_y) != len(a_z):
    raise Exception("Input arrays differ in lengths")

  if add_zeros < 0:
    raise Exception("Invalid value for add_zeros")

  if dt < 1e-10:
    raise Exception("Time increment zero or negative")

  sig_abs = [ np.sqrt(x**2+y**2+z**2) for x, y, z in zip(a_x, a_y, a_z) ]

  fft_fxyza = []

  for s in np.array([a_x, a_y, a_z, sig_abs]):

    if detrend:
      s = sig.detrend(s)

    n_zp  = int(add_zeros) + len(s)
    fft_s = np.fft.fft(s, n=n_zp, norm=None) * dt

    n_ft  = int(len(fft_s)/2. - 1)
    fft_s = fft_s[:n_ft]

    if normalize == "max":
      fft_s = fft_s / np.amax(fft_s)
    elif normalize == "abs":
      fft_s = fft_s / np.amax(abs(fft_s))

    if conjg:
      fft_fxyza.append(np.conjugate(fft_s))
    else:
      fft_fxyza.append(fft_s)

  fft_fxyza = [np.fft.fftfreq(n_zp, d=dt)[:n_ft].real * 2.0 * np.pi ] + fft_fxyza     

  return np.array(fft_fxyza) # to angular frequency

def calculate_fft_vec_pade(a_x, a_y, a_z, dt, n_bins=10000, fmin=0, fmax=100, normalize=None, conjg=False):

  """
    Script to calculate the FFT of an input array representing a time series with x,y,z components, consindering options
      1. Setting the the frequency resolution via a number of frequency bins
      2. Setting a maximum frequency desired
      3. Normalize the data
    @Joscha Hekele, optimized via https://github.com/jjgoings/pade/blob/master/pade.py
  
    Parameters:
    ----------
      a_x/y/z:    real/complex
                  input array of time signals in x/y/z direction
      dt:         real
                  time-step between successive values
      n_bins:     int
                  Determines frequency resolution
      fmin:       float
                  Minimum frequency
      fmax:       float
                  Maximum frequency
      normalize:  string, optional
                  Determines if fourier transform is normalized by maximum value ("max") or maximum value of absolute ("abs")
  
    Returns:
    -------
      complex arrays f, ft_x, ft_y, ft_z, ft_abs where
        f:      frequency
        ft_x:   ft of x component of signal
        ft_y:   ft of y component of signal
        ft_z:   ft of z component of signal
        ft_abs: ft the of signal's absolute
                
  """ 

  import numpy as np
  from scipy import signal as sig
  from scipy.linalg import toeplitz as tp

  if len(a_x) != len(a_y) != len(a_z):
    raise Exception("Input arrays differ in lengths")

  if dt < 1e-10:
    raise Exception("Time increment zero or negative")

  sig_abs = [ np.sqrt(x**2+y**2+z**2) for x, y, z in zip(a_x, a_y, a_z) ]

  ftp_fxyza = []

  for a in np.array([a_x, a_y, a_z, sig_abs]):

    # s = np.array(a) - a[0]
    s = sig.detrend(a)

    m = len(s)
    n = int(np.floor(m/2))

    d = -s[n+1:2*n]
    g = s[n+np.arange(1,n)[:,None] - np.arange(1,n)]

    b = np.linalg.solve(g, d)
    b = np.hstack((1, b)) 

    a = np.dot(np.tril(tp(s[0:n])), b)

    series_a = np.poly1d(a)
    series_b = np.poly1d(b)

    f = np.linspace(fmin, fmax, n_bins)

    z = np.exp(-1.0j*f*dt)

    s_p = series_a(z) / series_b(z)

    if normalize == "max":
      s_p = s_p / np.amax(s_p)
    elif normalize == "abs":
      s_p = s_p / np.amax(abs(s_p))

    if conjg:
      ftp_fxyza.append(np.conjugate(s_p))
    else:
      ftp_fxyza.append(s_p)

  ftp_fxyza = [f] + ftp_fxyza     

  return np.array(ftp_fxyza) 

def check_set_parse_list(parse_in):
 
  if parse_in is None:
    return None
 
  p = [parse_in] if not isinstance(parse_in, list) else parse_in

  if not ( len(parse_in) == 1 or len(parse_in) == 3 ):  
    raise Exception("> Input for data must be either one or three file/s!")

  while len(p) < 3:
    p.append(None)

  return p

def main(argv=None):

  if argv is None:
    argv = sys.argv
  
  parser = MyParser(usage="help msg")
  parser.add_argument("-pol", action="store", dest="pol", nargs="+", default=None)
  parser.add_argument("-con", action="store", dest="con", nargs="+", default=None)
  parser.add_argument("-rsp", action="store", dest="rsp", nargs="+", default=None)
  parser.add_argument("-hhg", action="store", dest="hhg", nargs=1,   default=None)
  parser.add_argument("-field", action="store", dest="field", nargs="+", default=None)
  parser.add_argument("-fa", action="store_true", dest="fa", default=False)
  parser.add_argument("-fc", action="store_true", dest="fc", default=True)
  parser.add_argument("-tmin", action="store", dest="tmin", default=0.0)
  parser.add_argument("-tmax", action="store", dest="tmax", default=-1.0)
  parser.add_argument("-tshift", action="store", dest="tshift", default=0.0)
  parser.add_argument("-fmin", action="store", dest="fmin", default=0.0)
  parser.add_argument("-fmax", action="store", dest="fmax", default=-1.0)
  parser.add_argument("-p", action="store_true", dest="p")
  parser.add_argument("-i", action="store", dest="i", const="1.5", nargs="?", default=None)
  parser.add_argument("-ip", action="store_true", dest="ip", default=False)
  parser.add_argument("-w", action="store", dest="w", const="output", nargs="?", default=None)
  parser.add_argument("-n", action="store_true", dest="n", default=False)
  parser.add_argument("-z", action="store", dest="z", default=0)
  parser.add_argument("-d", action="store", dest="d", nargs="+", default=[-1, "poly", 0.0001])
  parser.add_argument("-lp", action="store", dest="lp", nargs=2, default=(None,None))
  parser.add_argument("-ft-pade", action="store", dest="ftp", nargs=3, default=(-1,0.0,-1.0))
  parses = parser.parse_args()

  if not parses.p and not parses.w:
    raise IOError("> Expected at least one control parameter (p/w). Type -h for help.")

  if not parses.pol is None and not parses.con is None and not parses.rsp is None and not parses.hhg:
    raise IOError("> Cannot do polarization and conductivity and rotatory response and HHG calculation at once!")
  elif parses.pol is None and parses.con is None and parses.rsp is None and parses.hhg is None:
    raise IOError("> Missing input files for dipole or current or rotatory response or HHG data!") 

  if parses.field is None:
    raise IOError("> External field file missing.")

  facd = 0.000001
  if len(parses.d) < 2:
    raise IOError("> -d option requires at least 2 arguments")
  elif len(parses.d) == 2:
    damp = parses.d
  elif len(parses.d) == 3:
    damp = parses.d[0:2]
    facd = parses.d[2]

  field = check_set_parse_list(parses.field)
  pol   = check_set_parse_list(parses.pol)
  con   = check_set_parse_list(parses.con)
  rsp   = check_set_parse_list(parses.rsp)
  hhg   = check_set_parse_list(parses.hhg)

  fld_x = FieldData(field[0], "field_x", parses.fa)
  fld_y = FieldData(field[1], "field_y", parses.fa) if field[1] is not None else None
  fld_z = FieldData(field[2], "field_z", parses.fa) if field[2] is not None else None


  if not pol is None:

    input_field = set_external_field(fld_x, fld_y, fld_z, flag_copy_ref=True, flag_analytic_field=parses.fa)
 
    if parses.w == "output":
      parses.w = pol[0].split(".dat")[0]

    dip_x = DipoleData(pol[0], "dipole_x")
    dip_y = DipoleData(pol[1], "dipole_y") if pol[1] is not None else None
    dip_z = DipoleData(pol[2], "dipole_z") if pol[2] is not None else None
  
    dipole = ElectronicDipole(flag_plot=parses.p, flag_normalize=parses.n, flag_intp_plot=parses.ip,   \
               flag_analytic_ft=parses.fa, intp_fac=parses.i, t_min=parses.tmin, t_max=parses.tmax,    \
               t_shift=parses.tshift, f_min=parses.fmin, f_max=parses.fmax, lpeak_hgt=parses.lp[0],    \
               lpeak_wdt=parses.lp[1], add_zeros=parses.z, t0_damp=parses.d[0], damp_type=parses.d[1], \
               expfac_dp = facd, n_ft_pade=parses.ftp[0], fmin_pade=parses.ftp[1], fmax_pade=parses.ftp[2], \
               data_x=dip_x, data_y=dip_y, data_z=dip_z, \
               input_field=input_field, out_file_name=parses.w                                        )

  elif not con is None:

    input_field = set_external_field(fld_x, fld_y, fld_z, flag_copy_ref=True, flag_analytic_field=parses.fa)

    if parses.w == "output":
      parses.w = con[0].split(".dat")[0]

    cur_x = CurrentData(con[0], "current_x")
    cur_y = CurrentData(con[1], "current_y") if con[1] is not None else None
    cur_z = CurrentData(con[2], "current_z") if con[2] is not None else None

    current = ElectronicCurrent(flag_plot=parses.p, flag_normalize=parses.n, flag_intp_plot=parses.ip,   \
                flag_analytic_ft=parses.fa, intp_fac=parses.i, t_min=parses.tmin, t_max=parses.tmax,     \
                t_shift=parses.tshift, f_min=parses.fmin, f_max=parses.fmax,                             \
                add_zeros=parses.z, t0_damp=parses.d[0], damp_type=parses.d[1], n_ft_pade=parses.ftp[0], \
                fmin_pade=parses.ftp[1], fmax_pade=parses.ftp[2], data_x=cur_x, data_y=cur_y, data_z=cur_z,                       \
                expfac_dp = facd, \
                input_field=input_field, out_file_name=parses.w                                          )

  elif not rsp is None:

    input_field = set_external_field(fld_x, fld_y, fld_z, flag_copy_ref=True, flag_analytic_field=parses.fa)

    if parses.w == "output":
      parses.w = hhg[0].split(".dat")[0]

    mmom_x = MagMomData(rsp[0], "magmom_x")
    mmom_y = MagMomData(rsp[1], "magmom_y") if rsp[1] is not None else None
    mmom_z = MagMomData(rsp[2], "magmom_z") if rsp[2] is not None else None

    magmom = MagneticMoment(flag_plot=parses.p, flag_normalize=parses.n, flag_intp_plot=parses.ip,      \
               flag_analytic_ft=parses.fa, intp_fac=parses.i, t_min=parses.tmin, t_max=parses.tmax,     \
               t_shift=parses.tshift, f_min=parses.fmin, f_max=parses.fmax,                             \
               add_zeros=parses.z, t0_damp=parses.d[0], damp_type=parses.d[1], n_ft_pade=parses.ftp[0], \
               fmin_pade=parses.ftp[1], fmax_pade=parses.ftp[2], data_x=mmom_x, data_y=mmom_y, data_z=mmom_z,                    \
               expfac_dp = facd, \
               input_field=input_field, out_file_name=parses.w                                          )

  elif not hhg is None:

    input_field = set_external_field(fld_x, fld_y, fld_z, flag_copy_ref=False, flag_analytic_field=parses.fa)

    if parses.w == "output":
      parses.w = hhg[0].split(".dat")[0]

    hhg_data_xyz = HHGData(hhg[0], "hhg_data_xyz")

    hhg_spec = HHGSpectrum(input_file=hhg[0], flag_plot=parses.p, flag_normalize=parses.n, flag_intp_plot=parses.ip, \
                 flag_analytic_ft=parses.fa, intp_fac=parses.i, t_min=parses.tmin, t_max=parses.tmax,                \
                 t_shift=parses.tshift, f_min=parses.fmin, f_max=parses.fmax,                                        \
                 add_zeros=parses.z, t0_damp=parses.d[0], damp_type=parses.d[1], n_ft_pade=parses.ftp[0], fmax_pade=parses.ftp[1],             \
                 data_x=hhg_data_xyz, data_y=None, data_z=None,                                                   \
                 input_field=input_field, out_file_name=parses.w                                                     )



  plt.rc('mathtext', fontset="cm")
  plt.show()

  with open(".eval_tddft.log", "a") as f:
    astr = ""
    for a in argv:
      astr += str(a) + " "
    f.write(astr + "\n")

  print("> Logfile written to .eval_tddft.log")


def plot_sma_tddft(file_tddft, file_sma, normalize):

    """
    it reads in the sma data file and
    the rt-tddft output and plot the result
    together for comparison
    """

    print("> Plotting data ...")
    
    if normalize:
      norm = " (normalized by maximum)"
    else:
      norm = ""
    
    xlabel = r"Energy $E=\hbar\omega$ (eV)"
    
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(14,9))

    sma_frq = []
    sma_osc = []
    with open(file_sma,'r') as f_sma:
        for line in f_sma:
            if line.startswith('#'):
                continue
            else:
                sma_frq.append(float(line.split()[3]))
                sma_osc.append(float(line.split()[4]))

    tddft_frq = []
    tddft_osc = []
    with open(file_tddft,'r') as f_tddft:
        for line in f_tddft:
            if line.startswith('#'):
                continue
            if line.startswith(' #'):
                continue
            else:
                tddft_frq.append(float(line.split()[0]))
                tddft_osc.append(float(line.split()[1]))


    # gaussian broadening for sma_data            
    def gaussian_i(e_i,f_i,pre_fac,start,end,n_points):
            '''
            gives back a single gaussian function
            in a given range of eV
            '''

            f_i = f_i.real

            energy_range = np.linspace(start,end,n_points)

            argument = np.zeros_like(energy_range)
            argument = argument - e_i
            argument = argument + energy_range
            argument = argument/0.04  # sigma is 0.04 eV
            argument = np.square(argument)

            gauss = pre_fac * f_i/0.04 * np.exp(-argument)

            return gauss


    # function for normalizing data
    def normalize_max(*data_series):

       out_series = []

       for d in data_series:

         n = np.amax(np.abs(d))

         if abs(n) < 1.0e-10:
           n = 1.0

         out_series.append([ di / n for di in d ])

       return out_series[0] if len(out_series) == 1 else tuple(out_series)


    if normalize:
        sma_osc = normalize_max(sma_osc)
        tddft_osc = normalize_max(tddft_osc)


    # calculate SMA-TDDFT spectra
    sma_abs_strength = np.zeros(4000)
    sma_grid = np.linspace(0,sma_frq[-1],4000)

    for i in range(len(sma_frq)):
        sma_abs_strength += gaussian_i(sma_frq[i],sma_osc[i],1.,0,sma_frq[-1],4000)

    if normalize:
        sma_abs_strength = normalize_max(sma_abs_strength)
        tddft_osc = normalize_max(tddft_osc)

    ax.plot(sma_grid, sma_abs_strength,label='SMA')
    ax.plot(tddft_frq,np.abs(tddft_osc),label='RT-TDDFT')

    ax.legend(loc="upper left", bbox_to_anchor=(1.,1.031))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude" + norm)
    ax.set_title(r"Absorption strength function $|S(\omega)|=|\frac{2\omega}{\pi}\mathrm{Im}"\
                    +r"\left(\frac{1}{3}\mathrm{Tr}[\alpha(\omega)]\right)|$ and power spectrum", y=1.05)
    ax.set_yticks(np.linspace(0, max(np.abs(tddft_osc)), 10))
#    ax.set_xticks(np.linspace(xaxis[0], xaxis[-1], 15))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.margins(0,0.01)
    ax.grid()
