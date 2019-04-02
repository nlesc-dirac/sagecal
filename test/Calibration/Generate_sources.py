# The goal is here to generate thousands of random sources to populate a sky model.
# This sky model is solely used for profiling.
# It has to be pretty large because we need sufficiently long run times for adequate profiling.
# fields are (where h:m:s is RA, d:m:s is Dec):
# name h m s d m s I Q U V spectral_index0 spectral_index1 spectral_index2 RM extent_X(rad) extent_Y(rad)
# pos_angle(rad) freq0


import numpy as np
import warnings

number_of_parameters = 19
number_of_digits_for_sources = 5
number_of_sources = 10000
try:
    assert number_of_sources < 10**number_of_digits_for_sources
except AssertionError:
    number_of_sources = 10**number_of_digits_for_sources - 1
    print("Sorry, number of sources too large, reset to {0}".format(number_of_sources))


# Best to center sources around 3C196 if using the sm.ms measurement set with a 
# four degree tolerance
tol_seconds_of_decl = 3600 * 10 
tol_seconds_of_RA = tol_seconds_of_decl * 24/360

# I'll pretend 3C196 is at RA=0, decl=90 for now, to accommodate for a MS that I am using.
RA_hours_3C196 = 0
RA_minutes_3C196 = 0
RA_seconds_3C196 = 0
#RA_hours_3C196 = 8
#RA_minutes_3C196 = 13
#RA_seconds_3C196 = 35.981540

RA_seconds_3C196 = (RA_hours_3C196 * 60 + RA_minutes_3C196) *60 + RA_seconds_3C196

RA_seconds_low = RA_seconds_3C196 - tol_seconds_of_RA
RA_minutes_low, RA_seconds_low = divmod(RA_seconds_low, 60)
RA_hours_low, RA_minutes_low = divmod(RA_minutes_low, 60)

RA_seconds_high = RA_seconds_3C196 + tol_seconds_of_RA
RA_minutes_high, RA_seconds_high = divmod(RA_seconds_high, 60)
RA_hours_high, RA_minutes_high = divmod(RA_minutes_high, 60)

decl_degrees_3C196 = 90
decl_minutes_3C196 = 0
decl_seconds_3C196 = 0
#decl_degrees_3C196 = 48
#decl_minutes_3C196 = 12
#decl_seconds_3C196 = 59.174770

decl_seconds_3C196 = (decl_degrees_3C196 * 60 + decl_minutes_3C196) *60 + decl_seconds_3C196

decl_seconds_low = decl_seconds_3C196 - tol_seconds_of_decl
decl_minutes_low, decl_seconds_low = divmod(decl_seconds_low, 60)
decl_degrees_low, decl_minutes_low = divmod(decl_minutes_low, 60)

decl_seconds_high = decl_seconds_3C196 + tol_seconds_of_decl
decl_minutes_high, decl_seconds_high = divmod(decl_seconds_high, 60)
decl_degrees_high, decl_minutes_high = divmod(decl_minutes_high, 60)

I_low = 10
I_high = 100

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="using a non-integer number instead of an integer")
    # Maximum of 1e5 sources, so a 'P' + number_of_digits_for_sources digits,
    # that's why we have '...np.str_, number_of_digits_for_sources + 1'.
    source_parms_dtype = ((np.str_, number_of_digits_for_sources + 1),
                          int, int, float, int, int, float, float, int, int, int,
                          float, float, int, int, int, int, float, float)
    names = ('name', 'rah', 'ram', 'ras', 'dad', 'dam', 'das', 'I', 'Q', 'U', 'V',
             'sp0', 'sp1', 'sp2', 'RM', 'extX', 'extY',
             'pos_angle', 'freq0')

    formats = ['U6', 'i4', 'i4', 'f8', 'i4', 'i4', 'f8', 'f8', 'i4', 'i4', 'i4', 'f8', 'f8',
      'f8', 'f8', 'f8', 'f8', 'f8', 'f8']
    formats_reformatted = '%s  %d  %d  %f  %d  %d  %f  %f  %d  %d  %d  %f  %f  %f  %f  %f  %f  %f  %f'

    sources_parameters = np.recarray((number_of_sources,), formats=formats,
                                     names=names)

    for source_name in range(sources_parameters.shape[0]):
        sources_parameters.name[source_name] = 'P' + str(source_name).zfill(number_of_digits_for_sources)

    # Right ascension can have all values.
    # sources_parameters.rah = np.random.randint(ra_low, ra_high, size=number_of_sources)
    # sources_parameters.ram = np.random.randint(0, 59, size=number_of_sources)
    # sources_parameters.ras = 60 * np.random.rand(number_of_sources)

    sources_parameters.rah = (RA_hours_high - RA_hours_low) * np.random.random_sample(number_of_sources) + RA_hours_low 
    sources_parameters.ram = (RA_minutes_high - RA_minutes_low) * np.random.random_sample(number_of_sources) + RA_minutes_low 
    sources_parameters.ras = (RA_seconds_high - RA_seconds_low) * np.random.random_sample(number_of_sources) + RA_seconds_low

    sources_parameters.dad = (decl_degrees_high - decl_degrees_low) * np.random.random_sample(number_of_sources) + decl_degrees_low  
    sources_parameters.dam = (decl_minutes_high - decl_minutes_low) * np.random.random_sample(number_of_sources) + decl_minutes_low    
    sources_parameters.das = (decl_seconds_high - decl_seconds_low) * np.random.random_sample(number_of_sources) + decl_seconds_low    

    sources_parameters.I = (I_high - I_low) * np.random.rand(number_of_sources) + I_low
    sources_parameters.Q = 0
    sources_parameters.U = 0
    sources_parameters.V = 0

    # These spectral indices
    sources_parameters.sp0 = -np.random.rand(number_of_sources)
    sources_parameters.sp1 = 2 * np.random.rand(number_of_sources) - 1
    sources_parameters.sp2 = 0

    sources_parameters.RM = 0

    # Let's make the first half of the sources extended.
    first_half = int(number_of_sources/2)
    # Make major axes with a maximum of 0.1 degrees.
    major_axes_in_degrees = 0.1 * np.random.rand(first_half)
    semi_major_axes_in_radians = major_axes_in_degrees * np.pi/360.
    # For convenience, I'll just construct semi-minor axes half the size of the semi-major axes.
    semi_minor_axes_in_radians = semi_major_axes_in_radians/2.
    sources_parameters.extX[0: first_half] = semi_major_axes_in_radians
    sources_parameters.extX[first_half: number_of_sources] = 0
    sources_parameters.extY[0: first_half] = semi_minor_axes_in_radians
    sources_parameters.extY[first_half: number_of_sources] = 0
    sources_parameters.pos_angle[0: first_half] = 360. * np.random.rand(first_half)
    sources_parameters.pos_angle[first_half: number_of_sources] = 0

    sources_parameters.freq0 = 143000000.0

# print([elem for elem in sources_parameters[100]])

# sources_parameters.tofile("extended_source_list_using_tofile.txt", sep='\n')

with open("extended_source_list_centered_on_3C196.txt", 'wb') as f:
    f.write(b"##  From Generate_sources.py by Hanno Spreeuw.\n")
    f.write(b"##  Generates point sources at random positions with random brighnesses within some range.\n")
    f.write(b"##  this is an LSM text (hms/dms) file\n")
    f.write(b"##  fields are (where h:m:s is RA, d:m:s is Dec):\n")
    f.write(b"##  name h m s d m s I Q U V spectral_index0 spectral_index1 spectral_index2 " +
            b"RM extent_X(rad) extent_Y(rad) pos_angle(rad) freq0\n")
    f.write(b"\n")

    np.savetxt(f, sources_parameters, fmt=formats_reformatted)

# Now write the cluster file
# First add '1' and '1' to indicate the cluster id and chunk size.
cluster_array = np.concatenate((np.array(['1', '1']), sources_parameters.name))
with open("extended_source_list_centered_on_3C196.txt.cluster", 'wb') as f:
    np.savetxt(f, (cluster_array).reshape(1, cluster_array.shape[0]), fmt='%s', delimiter=' ')
