# This code is part of RRTM for GCM Applications - Parallel (RRTMGP)
#
# Contacts: Robert Pincus and Eli Mlawer
# email:  rrtmgp@aer.com
#
# Copyright 2015-2018,  Atmospheric and Environmental Research and
# Regents of the University of Colorado.  All right reserved.
#
# Use and duplication is permitted under the terms of the
#    BSD 3-clause license, see http://opensource.org/licenses/BSD-3-Clause
# -------------------------------------------------------------------------------------------------
#
# Routines for handling strings:
#   convert to lower case
#   does a string exist within an array of strings?
#   what is the location of a string with an array?
#
# -------------------------------------------------------------------------------------------------
module mo_util_string
  # implicit none
  # private
  # public :: lower_case, string_in_array, string_loc_in_array
  export lower_case, string_in_array, string_loc_in_array

  # List of character for case conversion
  # character(len=26), parameter :: LOWER_CASE_CHARS = 'abcdefghijklmnopqrstuvwxyz'
  # character(len=26), parameter :: UPPER_CASE_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# contains
  # -------------------------------------------------------------------------------------------------
  lower_case( input_string ) = lowercase(input_string)
    # character(len=*), intent(in) :: input_string
    # character(len=len(input_string)) :: output_string
    # integer :: i, n

  #   # Copy input string
  #   output_string = input_string

  #   # Convert case character by character
  #   do i = 1, len(output_string)
  #     n = index(UPPER_CASE_CHARS, output_string(i:i))
  #     if ( n /= 0 ) output_string(i:i) = LOWER_CASE_CHARS(n:n)
  #   end do
  # end function
  # --------------------------------------------------------------------------------------
  #
  # Is string somewhere in array?
  #
  function string_in_array(s, array)
    # character(len=*),               intent(in) :: string
    # character(len=*), dimension(:), intent(in) :: array
    # logical                                    :: string_in_array

    # integer :: i
    # character(len=len_trim(s)) :: lc_string

    s_in_array = false
    lc_string = lower_case(strip(s))
    for i in eachindex(array)
      if lc_string == lower_case(strip(array[i]))
        s_in_array = true
        break
      end
    end
    return s_in_array
  end
  # --------------------------------------------------------------------------------------
  #
  # Is string somewhere in array?
  #
  function string_loc_in_array(s, array)
    # character(len=*),               intent(in) :: string
    # character(len=*), dimension(:), intent(in) :: array
    # integer                                    :: string_loc_in_array

    # integer :: i
    # character(len=len_trim(string)) :: lc_string

    s_loc_in_array = -1
    lc_string = lower_case(strip(s))
    for i in eachindex(array)
      if lc_string == lower_case(strip(array[i]))
        s_loc_in_array = i
        break
      end
    end
    return s_loc_in_array
  end
  # --------------------------------------------------------------------------------------
end # module
