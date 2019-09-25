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
# This module reads an example file containing atomspheric conditions (temperature, pressure, gas concentrations)
#   and surface properties (emissivity, temperature), defined on nlay layers across a set of ncol columns subject to
#   nexp perturbations, and returns them in data structures suitable for use in rte and rrtmpg. The input data
#   are partitioned into a user-specified number of blocks.
# For the moment only quantities relevant to longwave calculations are provided.
#
# The example files comes from the Radiative Forcing MIP (https://www.earthsystemcog.org/projects/rfmip/)
#   The protocol for this experiment allows for different specifications of which gases to consider:
# all gases, (CO2, CH4, N2O) + {CFC11eq; CFC12eq + HFC-134eq}. Ozone is always included
# The protocol does not specify the treatmet of gases like CO
#
# -------------------------------------------------------------------------------------------------
module mo_rfmip_io
#  use mo_rte_kind,   only: wp
#  use mo_gas_concentrations, &
#                        only: ty_gas_concs
#  use mo_util_string,   only: lower_case, string_in_array, string_loc_in_array
#  use mo_simple_netcdf, only: read_field, write_field, get_dim_size
#  use netcdf
#  implicit none
#  private

  using ..mo_gas_concentrations
  using ..mo_util_string
  using NCDatasets

  export read_kdist_gas_names, determine_gas_names, read_size, read_and_block_pt, 
            read_and_block_sw_bc, read_and_block_lw_bc, read_and_block_gases_ty, 
            unblock_and_write

  ncol_l = 0
  nlay_l = 0 
  nexp_l = 0 # Local copies
  #--------------------------------------------------------------------------------------------------------------------
  # 
  # Find the size of the problem: columns, layers, perturbations (experiments)
  # 
  function read_size(fileName, ncol, nlay, nexp)
#    character(len=*),          intent(in   ) :: fileName
#    integer,         optional, intent(  out) :: ncol, nlay, nexp
#    ! ---------------------------
#    integer :: ncid
    # ---------------------------
#    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
#      call stop_on_err("read_size: can't find file " // trim(fileName))

    ds = Dataset(fileName,"r") # reading the NetCDF file in read only mode

    ncol = length( ds["site"]  )
    nlay = length( ds["layer"] )
    nexp = length( ds["expt"]  )

    if length( ds["level"] ) ≠ nlay+1
      stop_on_err("read_size: number of levels should be nlay+1")
    end

    ncol_l = ncol
    nlay_l = nlay
    nexp_l = nexp
  end
#--------------------------------------------------------------------------------------------------------------------
#
# Return layer and level pressures and temperatures as arrays dimensioned (ncol, nlay/+1, nblocks)
#   Input arrays are dimensioned (nlay/+1, ncol, nexp)
#   Output arrays are allocated within this routine
#
  function read_and_block_pt(fileName, blocksize, 
                               p_lay, p_lev, t_lay, t_lev)
#    character(len=*),           intent(in   ) :: fileName
#    integer,                    intent(in   ) :: blocksize
#    real(wp), dimension(:,:,:), allocatable, & ! [blocksize, nlay/+1, nblocks]
#                                intent(  out) :: p_lay, p_lev, t_lay, t_lev
    # ---------------------------
#    integer :: ncid
#    integer :: b, nblocks
#    real(wp), dimension(:,:  ), allocatable :: temp2d
#    real(wp), dimension(:,:,:), allocatable :: temp3d
    # ---------------------------

    wp = Float64

    if any([ncol_l, nlay_l, nexp_l]  .== 0) 
      stop_on_err("read_and_block_pt: Haven't read problem size yet.")
    end

    if (ncol_l*nexp_l)%blocksize ≠ 0 
      stop_on_err("read_and_block_pt: number of columns doesn't fit evenly into blocks.")
    end

    nblocks = (ncol_l*nexp_l)/blocksize

    p_lay = Array{wp},(undef,blocksize,nlay_1  ,nblocks)
    t_lay = Array{wp},(undef,blocksize,nlay_1  ,nblocks) 
    p_lev = Array{wp},(undef,blocksize,nlay_l+1,nblocks)
    t_lev = Array{wp},(undef,blocksize,nlay_l+1,nblocks)

#    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
#      call stop_on_err("read_and_block_pt: can't find file " // trim(fileName))
    # 
    # Read p, T data; reshape to suit RRTMGP dimensions
    # 
    ds = Dataset(fileName,"r")

    temp3d = reshape( repeat(ds["pres_layer"][:],1,1,nexp_l), nlay_l, blocksize, nblocks )

    for b = 1:nblocks
      p_lay[:,:,b] = transpose(temp3d[:,:,b])
    end

    temp3d = reshape( ds["temp_layer"][:], nlay_1, blocksize, nblocks )

    for b = 1:nblocks
      t_lay[:,:,b] = transpose(temp3d[:,:,b])
    end

    temp3d = []

    temp3d = reshape( repeat(ds["pres_level"][:],1,1,nexp_l), nlay_l+1, blocksize, nblocks )

    for b = 1:nblocks
      p_lev[:,:,b] = transpose(temp3d[:,:,b])
    end

    temp3d = reshape( ds["temp_level"][:],nlay_l+1, blocksize, nblocks )

    for b = 1:nblocks
      t_lev[:,:,b] = transpose(temp3d[:,:,b])
    end

    temp3d = []
  end
  #--------------------------------------------------------------------------------------------------------------------
  # 
  # Read and reshape shortwave boundary conditions
  #
  function read_and_block_sw_bc(fileName, blocksize, 
                               surface_albedo, total_solar_irradiance, solar_zenith_angle)
#    character(len=*),           intent(in   ) :: fileName
#    integer,                    intent(in   ) :: blocksize
#    real(wp), dimension(:,:), allocatable, &
#                                intent(  out) :: surface_albedo, total_solar_irradiance, solar_zenith_angle
#    ! ---------------------------
#    integer :: ncid
#    integer :: nblocks
#    real(wp), dimension(ncol_l, nexp_l) :: temp2D
    # ---------------------------
    wp = Float64

    if any([ncol_l, nlay_l, nexp_l] .== 0)
      stop_on_err("read_and_block_sw_bc: Haven't read problem size yet.")
    end
    if (ncol_l*nexp_l)%blocksize ≠ 0 
      stop_on_err("read_and_block_sw_bc: number of columns doesn't fit evenly into blocks.")
    end
    nblocks = (ncol_l*nexp_l)/blocksize
    #
    # Check that output arrays are sized correctly : blocksize, nlay, (ncol * nexp)/blocksize
    #

    ds = Dataset(fileName,"r")

    temp2D = repeat(ds["surface_albedo"][:],1,nexp_l)

    surface_albedo = reshape(temp2D,blocksize,nblocks)

    temp2D = repeat(ds["total_solar_irradiance"][:],1,nexp_l)
    total_solar_irradiance = reshape(temp2D,blocksize,nblocks)

    temp2D = repeat(ds["solar_zenith_angle"][:],1,nexp_l)
    solar_zenith_angle = reshape(temp2d,blocksize, nblocks)

  end
  #--------------------------------------------------------------------------------------------------------------------
  # 
  # Read and reshape longwave boundary conditions
  # 
  function read_and_block_lw_bc(fileName, blocksize, 
                                  surface_emissivity, surface_temperature)
#    character(len=*),           intent(in   ) :: fileName
#    integer,                    intent(in   ) :: blocksize
#    real(wp), dimension(:,:), allocatable, &
#                                intent(  out) :: surface_emissivity, surface_temperature
#    ! ---------------------------
#    integer :: ncid
#    integer :: nblocks
#    real(wp), dimension(ncol_l, nexp_l) :: temp2D ! Required to make gfortran 8 work, not sure why
    wp = Float64
    # ---------------------------
    if any([ncol_l, nlay_l, nexp_l]  .== 0)
      stop_on_err("read_and_block_lw_bc: Haven't read problem size yet.")
    end

    if (ncol_l*nexp_l)%blocksize ≠ 0
      stop_on_err("read_and_block_lw_bc: number of columns doesn't fit evenly into blocks.")
    end

    nblocks = (ncol_l*nexp_l)/blocksize

    ds = Dataset(fileName,"r")
    # 
    # Allocate on assignment
    # 
    temp2D = repeat(ds["surface_emissivity"][:],1,nexp_l)
    surface_emissivity  = reshape(temp2D,blocksize,nblocks)

    temp2D = repeat(ds["surface_temperature"][:],1,nexp_l)
    surface_temperature = reshape(temp2D,blocksize,nblocks)

  end
#--------------------------------------------------------------------------------------------------------------------
# 
# Create a pair of string arrays - one containing the chemical name of each gas, used by the k-distribution, and
#   one containing the name as contained in the RFMIP input files - depending on the forcing scenario
# Forcing index (1 = all available greenhouse gases;
#                2 = CO2, CH4, N2O, CFC11eq
#                3 = CO2, CH4, N2O, CFC12eq, HFC-134eq
#                All scenarios use 3D values of ozone, water vapor so those aren't listed here
# 
  function determine_gas_names(concentrationFile, kdistFile, forcing_index, names_in_kdist, names_in_file)
#    character(len=*),                             intent(in   ) :: concentrationFile, kdistFile
#    integer,                                      intent(in   ) :: forcing_index
#    character(len=32), dimension(:), allocatable, intent(inout) :: names_in_kdist, names_in_file
#    ! ----------------
#    integer :: num_gases, i
#    character(len=32), dimension(11) :: &
      chem_name = ["co", "ch4", "o2", "n2o", "n2", "co2",
        	   "CCl4", "ch4", "CH3Br", "CH3Cl", "cfc22"]

      conc_name = ["carbon_monoxide", "methane", "oxygen", 
          	   "nitrous_oxide", "nitrogen", 
        	   "carbon_dioxide", "carbon_tetrachloride",
        	   "methane", "methyl_bromide", 
        	   "methyl_chloride", "hcfc22"]
    # ----------------
    if forcing_index == 1
      read_kdist_gas_names(kdistFile, names_in_kdist)
      names_in_file = Array{String}(undef,length(names_in_kdist))
#      allocate(names_in_file(size(names_in_kdist)))
      for i = 1:length(names_in_kdist)
        names_in_file[i] = strip(lowercase(names_in_kdist[i]))
        # 
        # Use a mapping between chemical formula and name if it exists
        #
        if string_in_array(names_in_file[i], chem_name)
          names_in_file[i] = conc_name[string_loc_in_array(names_in_file[i], chem_name)]
        end
      end
    elseif forcing_index == 2
      num_gases = 6
      # 
      # Not part of the RFMIP specification, but oxygen is included because it's a major
      #    gas in some bands in the SW
      #
      names_in_kdist = ["co2", "ch4", "n2o", "o2", "cfc12", "cfc11"]
      names_in_file =  ["carbon_dioxide", "methane", "nitrous_oxide", 
                        "oxygen", "cfc12", "cfc11eq"]
    elseif forcing_index == 3
      num_gases = 6
      #
      # Not part of the RFMIP specification, but oxygen is included because it's a major
      #    gas in some bands in the SW
      #
      names_in_kdist = ["co2", "ch4", "n2o", "o2", "cfc12", "hfc134a"]
      names_in_file =  ["carbon_dioxide", "methane", "nitrous_oxide", 
                        "oxygen", "cfc12eq", "hfc134aeq"]
    else
      stop_on_err("determine_gas_names: unknown value of forcing_index")
    end

  end
#--------------------------------------------------------------------------------------------------------------------
#!
#! Read the names of the gases known to the k-distribution
#!
#!
  function read_kdist_gas_names(fileName, kdist_gas_names)
#    character(len=*),          intent(in   ) :: fileName
#    character(len=32), dimension(:), allocatable, &
#                               intent(  out) :: kdist_gas_names
#    ! ---------------------------
#    integer :: ncid, varid
#    character(len=9), parameter :: varName = "gas_names"
#    ! ---------------------------
    varName = "gas_names"

    ds = Dataset(fileName,"r")


#    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
#      call stop_on_err("read_kdist_gas_names: can't open file " // trim(fileName))

#    allocate(kdist_gas_names(get_dim_size(ncid, 'absorber')))

    ngases = size(ds["absorber"],1)

    kdist_gas_names = Array{String}(undef, ngases)

    if !haskey(ds,varName) 
      stop_on_err("read_kdist_gas_names: can't find variable " * strip(varName))
    end

    temp_str = ds[varName][:]

    for i = 1:ngases
      kdist_gas_names[i] = join(temp_str[:,i])
    end
    

#    if(nf90_inq_varid(ncid, trim(varName), varid) /= NF90_NOERR) &
#      call stop_on_err("read_kdist_gas_names: can't find variable " // trim(varName))
#    if(nf90_get_var(ncid, varid, kdist_gas_names)  /= NF90_NOERR) &
#      call stop_on_err("read_kdist_gas_names: can't read variable " // trim(varName))

#    ncid = nf90_close(ncid)
  end
  #!--------------------------------------------------------------------------------------------------------------------
  #!
  #! Read and reshape gas concentrations. RRTMGP requires gas concentrations to be supplied via a class
  #!   (ty_gas_concs). Gas concentrations are set via a call to gas_concs%set_vmr(name, values)
  #!   where `name` is nominally the chemical formula for the gas in question and `values` may be
  #!   a scalar, a 1-d profile assumed to apply to all columns, or an array of dimension (ncol, nlay).
  #! This routine outputs a vector nblocks long of these types so each element of the array can be passed to
  #!   the rrtmgp gas optics calculation in turn.
  #!
  #! This routine exploits RFMIP conventions: only water vapor and ozone vary by column within
  #!   each experiment.
  #! Fields in the RFMIP file have a trailing _GM (global mean); some fields use a chemical formula and other
  #!   a descriptive name, so a map is provided between these.
  #!
  function read_and_block_gases_ty(fileName, blocksize, gas_names, names_in_file, gas_conc_array)
#    character(len=*),           intent(in   ) :: fileName
#    integer,                    intent(in   ) :: blocksize
#    character(len=*),  dimension(:), &
#                                intent(in   ) :: gas_names ! Names used by the k-distribution/gas concentration type
#    character(len=*),  dimension(:), &
#                                intent(in   ) :: names_in_file ! Corresponding names in the RFMIP file
#    type(ty_gas_concs), dimension(:), allocatable, &
#                                intent(  out) :: gas_conc_array

#    ! ---------------------------
#    integer :: ncid
#    integer :: nblocks
#    integer :: b, g
#    integer,  dimension(:,:),   allocatable :: exp_num
#    real(wp), dimension(:),     allocatable :: gas_conc_temp_1d
#    real(wp), dimension(:,:,:), allocatable :: gas_conc_temp_3d
#    ! ---------------------------
    if any([ncol_l, nlay_l, nexp_l] .== 0) 
      stop_on_err("read_and_block_lw_bc: Haven't read problem size yet.")
    end
    if (ncol_l*nexp_l)%blocksize ≠ 0 
      stop_on_err("read_and_block_lw_bc: number of columns doesn't fit evenly into blocks.")
    end
    nblocks = (ncol_l*nexp_l)/blocksize
    gas_conc_array = Array{ty_gas_concs}(undef,nblocks)
#    allocate(gas_conc_array(nblocks))
    # Experiment index for each colum

    exp_num = permutedims( reshape( repeat( 1:nexp_l, ncol_l ), blocksize, nblocks ), [1,2] )

    #exp_num = reshape(spread([(b, b = 1, nexp_l)], 1, ncopies = ncol_l), shape = [blocksize, nblocks], order=[1,2])


#    if(nf90_open(trim(fileName), NF90_NOWRITE, ncid) /= NF90_NOERR) &
#      call stop_on_err("read_and_block_gases_ty: can't find file " // trim(fileName))

    ds = Dataset(fileName,"r")

    # 
    # Water vapor and ozone depend on col, lay, exp: look just like other fields
    # 
#    gas_conc_temp_3d = reshape(read_field(ncid, "water_vapor", nlay_l, ncol_l, nexp_l), &
#                               shape = [nlay_l, blocksize, nblocks]) * read_scaling(ncid, "water_vapor")

    gas_conc_temp_3d = reshape( ds["water_vapor"][:], nlay_l, blocksize, nblocks ) .* read_scaling(ds,"water_vapor")

    for b = 1:nblocks
      stop_on_err( set_vmr(gas_conc_array[b],"h2o", transpose(gas_conc_temp_3d[:,:,b])) )
    end

#    gas_conc_temp_3d = reshape(read_field(ncid, "ozone", nlay_l, ncol_l, nexp_l), &
#                               shape = [nlay_l, blocksize, nblocks]) * read_scaling(ncid, "ozone")

    gas_conc_temp_3d = reshape( ds["ozone"][:], nlay_l, blocksize, nblocks ) * read_scaling(ds,"ozone")

    for b = 1:nblocks
      stop_on_err(set_vmr(gas_conc_array[b],"o3", transpose(gas_conc_temp_3d[:,:,b])))
    end

    # 
    # All other gases are a function of experiment only
    # 
    for g = 1:length(gas_names)
      # 
      # Skip 3D fields above, also NO2 since RFMIP doesn't have this
      # 
      if string_in_array(gas_names[g], ["h2o", "o3", "no2"])
        continue
      end

      # Read the values as a function of experiment
#      gas_conc_temp_1d = read_field(ncid, trim(names_in_file(g)) // "_GM", nexp_l) *&
#        read_scaling(ncid, trim(names_in_file(g)) // "_GM")
      gas_conc_temp1d = ds[strip(names_in_file[g]) * "_GM"][:] * read_scaling(ds,strip(names_in_file[g]) * "_GM")

      for b = 1:nblocks
        # Does every value in this block belong to the same experiment?
        if all( exp_num[2:end,b] .- exp_num[1,b] .== 0 )
          # Provide a scalar value
          stop_on_err(set_vmr(gas_conc_array[b],gas_names[g], gas_conc_temp_1d[exp_num[1,b]]))
        else
          # Create 2D field, blocksize x nlay, with scalar values from each experiment
#          call stop_on_err(gas_conc_array(b)%set_vmr(gas_names(g), &
#          spread(gas_conc_temp_1d(exp_num(:,b)), 2, ncopies = nlay_l)))
          stop_on_err( set_vmr(gas_conc_array[b],gas_names[g], repeat(gas_conc_temp_1d, 1, nlay_l) ) )
        end 
      end

    end
  end
  #--------------------------------------------------------------------------------------------------------------------
#  function read_scaling(ncid, varName)
  function read_scaling(DS, varName) # modified to pass DS (dataset instead of ncid)
#    integer,          intent(in) :: ncid
#    character(len=*), intent(in) :: varName
#    real(wp)                     :: read_scaling

#    integer           :: varid
#    character(len=16) :: charUnits

    wp = Float64

    if haskey(DS,varName)
      scaling_str = DS[varName].attrib["units"]
    else
      stop_on_err("read_scaling: can't find variable " * strip(varName))
    end

#    if(nf90_inq_varid(ncid, trim(varName), varid) /= NF90_NOERR) &
#      call stop_on_err("read_scaling: can't find variable " // trim(varName))
#    if(nf90_get_att(ncid, varid, "units", charUnits)  /= NF90_NOERR) &
#      call stop_on_err("read_scaling: can't read attribute 'units' from variable " // trim(varName))
#    read(charUnits, *) read_scaling
    return parse(wp,scaling_str)

  end
#--------------------------------------------------------------------------------------------------------------------
  #
  # Reshape and reorder values (nominally fluxes) from RTE order (ncol, nlev, nblocks)
  #   to RFMIP order (nlev, ncol, nexp), then write them to a user-specified variable
  #   in a netCDF file.
  # 
  function unblock_and_write(fileName, varName, values)
#    character(len=*),           intent(in   ) :: fileName, varName
#    real(wp), dimension(:,:,:),  & ! [blocksize, nlay/+1, nblocks]
#                                intent(in   ) :: values
#    ! ---------------------------
#    integer :: ncid
#    integer :: b, blocksize, nlev, nblocks
#    real(wp), dimension(:,:), allocatable :: temp2d
#    ! ---------------------------
    wp = Float64

    if any([ncol_l, nlay_l, nexp_l] .== 0)
      stop_on_err("unblock_and_write: Haven't read problem size yet.")
    end
    blocksize = size(values,1)
    nlev      = size(values,2)
    nblocks   = size(values,3)
    if nlev ≠ nlay_l+1 
      stop_on_err("unblock_and_write: array values has the wrong number of levels")
    end
    if (blocksize*nblocks) ≠ (ncol_l*nexp_l)
      stop_on_err("unblock_and_write: array values has the wrong number of blocks/size")
    end

    temp2D = Array{wp}(undef,nlev,ncol_l*nexp_l)

    for b = 1:nblocks
      temp2D[1:nlev, ((b-1)*blocksize+1):(b*blocksize)] = transpose(values[1:blocksize,1:nlev,b])
    end
    # 
    # Check that output arrays are sized correctly : blocksize, nlay, (ncol * nexp)/blocksize
    #
# SK - commenting - assuming it is appending an existing file
#    if(nf90_open(trim(fileName), NF90_WRITE, ncid) /= NF90_NOERR) &
#      call stop_on_err("unblock_and_write: can't find file " // trim(fileName))
#    call stop_on_err(write_field(ncid, varName,  &
#                                 reshape(temp2d, shape = [nlev, ncol_l, nexp_l])))
#
#    ncid = nf90_close(ncid) 
    # this portion needs to be double checked
    ds = Dataset(fileName,"a")
    defVar(ds,varName, reshape(temp2d,nlev,ncol_l,nexp_l), ("nlev","ncol_l","nexp_l"))

  end  
#--------------------------------------------------------------------------------------------------------------------
  function stop_on_err(msg)
    #
    # Print error message and stop
    #
    if length(strip(msg)) > 0 
      error(strip(msg))
    end
  end
#-------------------------------------------
end
