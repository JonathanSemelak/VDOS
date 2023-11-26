#######   PLEASE READ THE FOLLOWING INSTRUCTION BEFORE RUNNING SCRIPT   #######
###                                                                         ###
###  NOTE: The position file needs to align before run total VACF script!!  ###
###                                                                         ###
###    The Format for Running This Script:                                  ###
###    ./VACF_KW.py INPUT_FILE_NAME DELTA_T OUTPUT_FILE_NAME                ###
###                                                                         ###
###    The values need to input manually when runing this script            ###
###                                                                         ###
###    (1) DIRECTORY_OF_YOUR_DATA: The path contains your data              ###
###                                                                         ###
###    (2) INPUT_FILE_NAME: The POSITION.xyz file                           ###
###                     (NOTE: do NOT need to re-split the Position file)   ###
###                                                                         ###
###    (3) DELTA_T: The Time_step set in simulation, in unit of fs          ###
###                                                                         ###
###    (4) OUTPUT_FILE_NAME: The Name of the Output File.                   ###
###                    (NOTE: do NOT need to type ">" sign!)                ###
###                                                                         ###
###    After inputing the above mentioned values, the program will list     ###
###  the atoms and their corresponding indices (only the first 35 atoms     ###
###  will show if the system is too large).                                 ###
###    And then the program will ask the user to enter the type of mode,    ###
###  e.g. "s": stretch, "b": bend, "w": wag, ect. and the indices of the    ###
###  atoms in order to choose the group of atoms involve the mode (the      ###
###  partial VACF). If the user enter "all" or "-1", the program will       ###
###  choose all atoms to calcluate the total VACF.                          ### .
###    Consequently, the program will ask the user to enter the type of     ###
###  window function, e.g. "Gaussian", "BlackmanHarris", "Hamming "or       ###
###  "Hann", ect.                                                           ###
###    After all steps of inputs finishing, the calculation begins.         ###
###                                                                         ###
###############################  Let's Try It! ################################

import argparse
import sys
import math

c = 2.9979245899e10 # speed of light in vacuum in [cm/s], from Wikipedia.
# ------------------------------------------
# Functions definition

def check_mode(args):
    if args.mode not in ['full', 'bond']:
        raise argparse.ArgumentTypeError("Mode must be 'full' or 'bond'")
    if args.mode == 'bond':
        if args.bond is None or len(args.bond) != 2:
            raise argparse.ArgumentTypeError("When mode is 'bond', -b must be provided with two integers")
    return args.mode

def check_window_kind(window_kind):
    if window_kind not in ['Gaussian','Blackman-Harris','Hamming','Hann']:
        sys.exit("Error: Window kind for FFT (-w) must be 'Gaussian' or 'Blackman-Harris' or 'Hamming' or 'Hann'")

def check_libraries_and_file(input_name):
    # Check and import common libraries
    try:
        from scipy import signal
    except ImportError:
        sys.exit("Error: SIGNAL from SCIPY library is required but is not installed.")
    try:
        import numpy as np
    except ImportError:
        sys.exit("Error: NUMPY library is required but is not installed.")

    # Check file extension and import specific libraries if necessary
    if input_name.endswith('.xyz'):
        try:
            from ase.io import read
        except ImportError:
            sys.exit("Error: ASE library is required for XYZ files but is not installed.")
    elif input_name.endswith(('.nc', '.netcdf', '.crd')):
        # Assuming no extra libraries are needed for NetCDF files
        pass
    else:
        sys.exit(f"Error: Unsupported file format '{input_name}'.")
# ------------------------------------------
# This functions were copied from https://github.com/LePingKYXK/Velocity-ACF

def calc_derivative(array_1D, delta_t):
    ''' The derivatives of the angle_array were obtained by using the
    finite differences method.
    '''
    dy = np.gradient(array_1D)
    return np.divide(dy, delta_t)

def choose_window(nsteps, window_kind):
    if window_kind == "Gaussian":
        sigma = 2 * math.sqrt(2 * math.log(2))
        std = 4000.0
        window_function = signal.windows.gaussian(nsteps, std/sigma, sym=False)
    elif window_kind == "Blackman-Harris":
        window_function = signal.windows.blackmanharris(nsteps, sym=False)
    elif window_kind == "Hamming":
        window_function = signal.windows.hamming(nsteps, sym=False)
    elif window_kind == "Hann":
        window_function = signal.windows.hann(nsteps, sym=False)
    return window_function

def calc_FFT(array_1D, window):
    """
    This function is for calculating the "intensity" of the ACF at each frequency
    by using the discrete fast Fourier transform.
    """
####
#### http://stackoverflow.com/questions/20165193/fft-normalization
####
    WE = sum(window) / len(array_1D)
    wf = window / WE
    sig = array_1D * wf
    # A series of number of zeros will be padded to the end of the \
    # VACF array before FFT.
    N = zero_padding(sig)

    yfft = np.fft.fft(sig, N, axis=0) / len(sig)
    return np.square(np.absolute(yfft))

def zero_padding(sample_data):
    """ A series of Zeros will be padded to the end of the dipole moment
    array (before FFT performed), in order to obtain a array with the
    length which is the "next power of two" of numbers.
    #### Next power of two is calculated as: 2**math.ceil(math.log(x,2))
    #### or Nfft = 2**int(math.log(len(data_array)*2-1, 2))
    """
    return int(2 ** math.ceil(math.log(len(sample_data), 2)))

def calc_ACF(array_1D):
    yunbiased = array_1D - np.mean(array_1D, axis=0)
    ynorm = np.sum(np.power(yunbiased,2), axis=0)
    autocor = signal.fftconvolve(array_1D,
                                 array_1D[::-1],
                                 mode='full')[len(array_1D)-1:] / ynorm
    return autocor
#-------------------------------------------
# Create the parser
parser = argparse.ArgumentParser(description='This gut will get the VDOS out of your trajectory file.')
# Define the command-line arguments
parser.add_argument('-i', '--input', required=True, help='Input file name')
parser.add_argument('-o', '--output', required=True, help='Output file name')
parser.add_argument('-m', '--mode', default='full', help="Mode of operation ('full' or 'bond')")
parser.add_argument('-dt', '--delta_t', type=float, required=True, help='Delta time in femtoseconds')
parser.add_argument('-b', '--bond', nargs=2, type=int, help='Bond indices (two integers)')
parser.add_argument('-w', '--window_kind', default='Gaussian', help="Window kind for FFT ('Gaussian' or 'Blackman-Harris' or 'Hamming' or 'Hann')")
parser.add_argument('-n', '--force_numerical', type=bool, default=False, help="Force numerical calculation of velocities")

# Parse the arguments
args = parser.parse_args()

# Assign values from args
input_name = args.input
output_name = args.output
mode = args.mode
delta_t = args.delta_t * 1e-15  # Convert from femtoseconds to seconds
bond_indices = args.bond if mode == 'bond' else None
window_kind = args.window_kind
force_numerical = args.force_numerical
# Checks input_name extension and required libraries are installed
check_libraries_and_file(input_name)
# Checks the mode is an available option
check_mode(args)
# Checks the window_kind is an available option
check_window_kind(window_kind)
# Checks the force_numerical is an available option

# Conditional import based on file extension
if input_name.endswith('.xyz'): from ase.io import read
from scipy import signal
from scipy.io import netcdf_file
import numpy as np

# Set defaults
contains_velocities=False
# Reads data
if input_name.endswith('.xyz'):  # XYZ file case
    print("Coordinates from the xyz file will be read using the ASE library")
    print("Reading file...")
    trajectory = read(input_name, index=':')
    nsteps = len(trajectory)
    natoms = len(trajectory[0])
    # Preallocate the NumPy array
    coordinates = np.empty((nsteps, natoms, 3))  # Assuming 3D coordinates
    # Fill in the array
    for i, frame in enumerate(trajectory):
        coordinates[i] = frame.get_positions()
    if(mode=="full"):
        print("The VDOS will be obtained considering all atoms")
        print("Velocities will be calculated numerically")
        normal_vectors = np.linalg.norm(coordinates, axis=-1)
        print("nsteps,natoms=",nsteps,natoms)
        print("coordinates shape=",np.shape(coordinates))
        print("coordinates first",coordinates[0])
else: # NETCDF file case
    print("Coordinates/velocities from the netcdf file will be read using the scipy library")
    print("Reading file...")
    trajectory = netcdf_file(input_name, 'r')
    if(mode=="full"):
        print("The VDOS will be obtained considering all atoms")
        contains_velocities="velocities" in trajectory.variables
        if(contains_velocities and not force_numerical):
            print("Velocities will be read from the trajectory file.")
            velocities = np.array(trajectory.variables['velocities'].data)
            nsteps = len(velocities)
            natoms = len(velocities[0])
            normal_vectors = np.linalg.norm(velocities, axis=-1)
        else:
            if(contains_velocities and force_numerical):
                print("Found velocities but numerical calculation is forced")
            print("Velocities will be calculated numerically")
            coordinates = np.array(trajectory.variables['coordinates'].data)
            nsteps = len(coordinates)
            natoms = len(coordinates[0])
            print("nsteps,natoms=",nsteps,natoms)
            print("coordinates shape=",np.shape(coordinates))
            print("coordinates first",coordinates[0])
            normal_vectors = np.linalg.norm(coordinates, axis=-1)
        print("The program will deal with all atoms one by one.")

window=choose_window(nsteps,window_kind)

used_normalized_vectors=True
used_normalized_vectors=False
if(used_normalized_vectors):
    for i in range(natoms):
        if (contains_velocities and not force_numerical):
            atom_velocities = normal_vectors[:,i]
        else:
            atom_velocities = calc_derivative(normal_vectors[:,i], delta_t)
        ACF = calc_ACF(atom_velocities)
        yfft_i = calc_FFT(ACF, window)
        if i == 0:
            yfft = yfft_i
        else:
            yfft += yfft_i
else:
    for i in range(natoms):
        for j in range(0,3): #xyz dimensions
            if (contains_velocities and not force_numerical):
                atom_velocities = velocities[:,i,j]
            else:
                atom_velocities = calc_derivative(coordinates[:,i,j], delta_t)
            ACF = calc_ACF(atom_velocities)
            yfft_i = calc_FFT(ACF, window)
            if i == 0:
                yfft = yfft_i
            else:
                yfft += yfft_i

wavenumber = np.fft.fftfreq(len(yfft), delta_t * c)[0:int(len(yfft) / 2)]
intensity = yfft[0:int(len(yfft)/2)]

print("VDOS saved to " + output_name + " file")
np.savetxt(output_name,np.column_stack((wavenumber,intensity)))