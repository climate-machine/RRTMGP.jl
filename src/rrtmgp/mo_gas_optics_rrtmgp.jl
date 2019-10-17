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
# Class for computing spectrally-resolved gas optical properties and source functions
#   given atmopsheric physical properties (profiles of temperature, pressure, and gas concentrations)
#   The class must be initialized with data (provided as a netCDF file) before being used.
#
# Two variants apply to internal Planck sources (longwave radiation in the Earth's atmosphere) and to
#   external stellar radiation (shortwave radiation in the Earth's atmosphere).
#   The variant is chosen based on what information is supplied during initialization.
#   (It might make more sense to define two sub-classes)
#
# -------------------------------------------------------------------------------------------------
module mo_gas_optics_rrtmgp
  # use mo_rte_kind,           only: FT, wl
  # use mo_rrtmgp_constants,   only: avogad, m_dry, m_h2o, grav
  using OffsetArrays
  using ..fortran_intrinsics
  using ..mo_rrtmgp_constants
  # use mo_util_array,         only: zero_array, any_vals_less_than, any_vals_outside
  using ..mo_util_array
  # use mo_optical_props,      only: ty_optical_props
  using ..mo_optical_props
  # use mo_source_functions,   only: ty_source_func_lw
  using ..mo_source_functions
  # use mo_gas_optics_kernels, only: interpolation,
  #                                  compute_tau_absorption, compute_tau_rayleigh, compute_Planck_source,
  #                                  combine_and_reorder_2str, combine_and_reorder_nstr
  using ..mo_gas_optics_kernels

  using ..mo_util_string
  # use mo_gas_concentrations, only: ty_gas_concs
  using ..mo_gas_concentrations
  # use mo_optical_props,      only: ty_optical_props_arry, ty_optical_props_1scl, ty_optical_props_2str, ty_optical_props_nstr
  using ..mo_optical_props
  # use mo_gas_optics,         only: ty_gas_optics
  using ..mo_gas_optics # only defines abstract interfaces
  using ..mo_util_reorder
  export gas_optics!, ty_gas_optics_rrtmgp, load_totplnk, load_solar_source
  export source_is_internal, source_is_external, get_press_min

  # -------------------------------------------------------------------------------------------------
  # type, extends(ty_gas_optics), public :: ty_gas_optics_rrtmgp
  mutable struct ty_gas_optics_rrtmgp{T,I} <: ty_gas_optics{T,I}
    optical_props#::ty_optical_props
    press_ref#::Vector{T}
    press_ref_log#::Vector{T}
    temp_ref#::Vector{T}
    press_ref_min#::T
    press_ref_max#::T
    temp_ref_min#::T
    temp_ref_max#::T
    press_ref_log_delta#::T
    temp_ref_delta#::T
    press_ref_trop_log#::T
    gas_names#::Vector{String}     # gas names
    vmr_ref#::Array{T,3}       # vmr_ref(lower or upper atmosphere, gas, temp)
    flavor#::Array{I, 2}        # major species pair; (2,nflav)
    gpoint_flavor#::Array{I, 2} # flavor = gpoint_flavor(2, g-point)
    kmajor#::Array{T,4}        #  kmajor(g-point,eta,pressure,temperature)
    minor_limits_gpt_lower#::Array{I,2}
    minor_limits_gpt_upper#::Array{I,2}
    minor_scales_with_density_lower#::Vector{Bool}
    minor_scales_with_density_upper#::Vector{Bool}
    scale_by_complement_lower#::Vector{Bool}
    scale_by_complement_upper#::Vector{Bool}
    idx_minor_lower#::Vector{I}
    idx_minor_upper#::Vector{I}
    idx_minor_scaling_lower#::Vector{I}
    idx_minor_scaling_upper#::Vector{I}
    kminor_start_lower#::Vector{I}
    kminor_start_upper#::Vector{I}
    kminor_lower#::Array{T,3}
    kminor_upper#::Array{T,3} # kminor_lower(n_minor,eta,temperature)
    krayl#::Array{T, 4} # krayl(g-point,eta,temperature,upper/lower atmosphere)
    planck_frac#::Array{T, 4}   # stored fraction of Planck irradiance in band for given g-point
    totplnk#::Array{T,2}       # integrated Planck irradiance by band; (Planck temperatures,band)
    totplnk_delta#::T # temperature steps in totplnk
    solar_src#::Vector{T} # incoming solar irradiance(g-point)
    is_key#::Vector{Bool}
  end
  ty_gas_optics_rrtmgp(T,I) = ty_gas_optics_rrtmgp{T,I}(ty_optical_props_base(T,I), ntuple(i->nothing, 35)...)

  # -------------------------------------------------------------------------------------------------
  #
  # col_dry is the number of molecules per cm-2 of dry air
  #
  # public :: get_col_dry # Utility function, not type-bound
  export get_col_dry

#   interface check_range
#     module procedure check_range_1D, check_range_2D, check_range_3D
#   end interface check_range

#   interface check_extent
#     module procedure check_extent_1D, check_extent_2D, check_extent_3D
#     module procedure check_extent_4D, check_extent_5D, check_extent_6D
#   end interface check_extent
# contains
  # --------------------------------------------------------------------------------------
  #
  # Public procedures
  #
  # --------------------------------------------------------------------------------------
  #
  # Two functions to define array sizes needed by gas_optics()
  #
  function get_ngas(this::ty_gas_optics_rrtmgp)
    # return the number of gases registered in the spectral configuration
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # integer                                        :: get_ngas

    return length(this.gas_names)
  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # return the number of distinct major gas pairs in the spectral bands (referred to as
  # "flavors" - all bands have a flavor even if there is one or no major gas)
  #
  function get_nflav(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # integer                                 :: get_nflav

    return size(this.flavor, 2)
  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # Compute gas optical depth and Planck source functions,
  #  given temperature, pressure, and composition
  #
  function gas_optics!(this::ty_gas_optics_rrtmgp,
                       play,
                       plev,
                       tlay,
                       tsfc,
                       gas_desc::ty_gas_concs,
                       optical_props::ty_optical_props_arry,
                       sources::ty_source_func_lw,
                       col_dry=nothing,
                       tlev=nothing) # result(error_msg)
    # inputs
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # real(FT), dimension(:,:), intent(in   ) :: play,    # layer pressures [Pa, mb]; (ncol,nlay)
    #                                            plev,    # level pressures [Pa, mb]; (ncol,nlay+1)
    #                                            tlay      # layer temperatures [K]; (ncol,nlay)
    # real(FT), dimension(:),   intent(in   ) :: tsfc      # surface skin temperatures [K]; (ncol)
    # type(ty_gas_concs),       intent(in   ) :: gas_desc  # Gas volume mixing ratios
    # # output
    # class(ty_optical_props_arry),
    #                           intent(inout) :: optical_props # Optical properties
    # class(ty_source_func_lw    ),
    #                           intent(inout) :: sources       # Planck sources
    # character(len=128)                      :: error_msg
    # # Optional inputs
    # real(FT), dimension(:,:),   intent(in   ),
    #                        optional, target :: col_dry,   # Column dry amount; dim(ncol,nlay)
    #                                            tlev        # level temperatures [K]; (ncol,nlay+1)
    # ----------------------------------------------------------
    # Local variables
    # Interpolation coefficients for use in source function
    # integer,     dimension(size(play,dim=1), size(play,dim=2)) :: jtemp, jpress
    # logical(wl), dimension(size(play,dim=1), size(play,dim=2)) :: tropo
    # real(FT),    dimension(2,2,2,get_nflav(this),size(play,dim=1), size(play,dim=2)) :: fmajor
    # integer,     dimension(2,    get_nflav(this),size(play,dim=1), size(play,dim=2)) :: jeta

    # integer :: ncol, nlay, ngpt, nband, ngas, nflav
    # # ----------------------------------------------------------
println("entered gas_optics")
    FT = eltype(play)
println("FT = ", FT)

    jpress = Array{Int}(undef, size(play))
    jtemp = Array{Int}(undef, size(play))
    tropo = Array{Bool}(undef, size(play))
    fmajor = Array{FT}(undef, 2,2,2,get_nflav(this),size(play)...)
    jeta = Array{Int}(undef, 2,    get_nflav(this), size(play)...)

    ncol  = size(play, 1)
    nlay  = size(play, 2)
    ngpt  = get_ngpt(this.optical_props)
    nband = get_nband(this.optical_props)
    #
    # Gas optics
    #
    #$acc enter data create(jtemp, jpress, tropo, fmajor, jeta)
println("before compute_gas_taus")
    jtemp, jpress, jeta, tropo, fmajor = compute_gas_taus!(this,
                     ncol, nlay, ngpt, nband,
                     play, plev, tlay, gas_desc,
                     optical_props,
                     col_dry)

println("after compute_gas_taus")
    # ----------------------------------------------------------
    #
    # External source -- check arrays sizes and values
    # input data sizes and values
    #
println("before check_extent")
    check_extent(tsfc, ncol, "tsfc")
println("after check_extent")
    check_range(tsfc, this.temp_ref_min,  this.temp_ref_max,  "tsfc")
println("after check_range")
    if present(tlev)
      check_extent(tlev, (ncol, nlay+1), "tlev")
      check_range(tlev, this.temp_ref_min, this.temp_ref_max, "tlev")
    end
println("before output_extents")
@show ("get_ncol(sources) = ")
@show (get_ncol(sources))
@show ("get_nlay(sources) = ", get_nlay(sources))
@show ("get_ngpt(sources) = ", get_ngpt(sources))

    #
    #   output extents
    #
    if [get_ncol(sources), get_nlay(sources), get_ngpt(sources)] ≠ [ncol, nlay, ngpt]
      error("gas_optics%gas_optics: source function arrays inconsistently sized")
    end

    #
    # Interpolate source function
    #
println("before source")
    source(this,
           ncol, nlay, nband, ngpt,
           play, plev, tlay, tsfc,
           jtemp, jpress, jeta, tropo, fmajor,
           sources,
           tlev)
println("after source")
    #$acc exit data delete(jtemp, jpress, tropo, fmajor, jeta)
  end
  #------------------------------------------------------------------------------------------
  #
  # Compute gas optical depth given temperature, pressure, and composition
  #
  function gas_optics!(this::ty_gas_optics_rrtmgp,
                       play,
                       plev,
                       tlay,
                       gas_desc::ty_gas_concs,    # mandatory inputs
                       optical_props::ty_optical_props_arry,
                       toa_src,        # mandatory outputs
                       col_dry=nothing) # result(error_msg)      # optional input

    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # real(FT), dimension(:,:), intent(in   ) :: play,    # layer pressures [Pa, mb]; (ncol,nlay)
    #                                            plev,    # level pressures [Pa, mb]; (ncol,nlay+1)
    #                                            tlay      # layer temperatures [K]; (ncol,nlay)
    # type(ty_gas_concs),       intent(in   ) :: gas_desc  # Gas volume mixing ratios
    # # output
    # class(ty_optical_props_arry),
    #                           intent(inout) :: optical_props
    # real(FT), dimension(:,:), intent(  out) :: toa_src     # Incoming solar irradiance(ncol,ngpt)
    # character(len=128)                      :: error_msg

    # Optional inputs
    # real(FT), dimension(:,:), intent(in   ),
    #                        optional, target :: col_dry # Column dry amount; dim(ncol,nlay)
    # ----------------------------------------------------------
    # Local variables
    # Interpolation coefficients for use in source function
    # integer,     dimension(size(play,dim=1), size(play,dim=2)) :: jtemp, jpress
    # logical(wl), dimension(size(play,dim=1), size(play,dim=2)) :: tropo
    # real(FT),    dimension(2,2,2,get_nflav(this),size(play,dim=1), size(play,dim=2)) :: fmajor
    # integer,     dimension(2,    get_nflav(this),size(play,dim=1), size(play,dim=2)) :: jeta

    # integer :: ncol, nlay, ngpt, nband, ngas, nflav
    # integer :: igpt, icol
    # # ----------------------------------------------------------

    FT = eltype(play)

    jpress = Array{Int}( undef, size(play))
    jtemp  = Array{Int}( undef, size(play))
    tropo  = Array{Bool}(undef, size(play))
    fmajor = Array{FT}(  undef, 2,2,2, get_nflav(this), size(play)...)
    jeta   = Array{Int}( undef, 2,     get_nflav(this), size(play)...)

    ncol  = size(play, 1)
    nlay  = size(play, 2)
    ngpt  = get_ngpt(this.optical_props)
    nband = get_nband(this.optical_props)
    ngas  = get_ngas(this)
    nflav = get_nflav(this)
    #
    # Gas optics
    #
    #$acc enter data create(jtemp, jpress, tropo, fmajor, jeta)
    jtemp, jpress, jeta, tropo, fmajor = compute_gas_taus!(this,
                     ncol, nlay, ngpt, nband,
                     play, plev, tlay, gas_desc,
                     optical_props,
                     col_dry)

    # test_data(jeta, "jeta")
    # test_data(fmajor, "fmajor")
    #$acc exit data delete(jtemp, jpress, tropo, fmajor, jeta)

    # ----------------------------------------------------------
    #
    # External source function is constant
    #
    check_extent(toa_src,     (ncol,         ngpt), "toa_src")
    #$acc parallel loop collapse(2)
    for igpt in 1:ngpt
       for icol in 1:ncol
          toa_src[icol,igpt] = this.solar_src[igpt]
       end
    end
  end
  #------------------------------------------------------------------------------------------
  #
  # Returns optical properties and interpolation coefficients
  #
  function compute_gas_taus!(this::ty_gas_optics_rrtmgp,
                            ncol, nlay, ngpt, nband,
                            play, plev, tlay, gas_desc::ty_gas_concs,
                            optical_props::ty_optical_props_arry,
                            col_dry=nothing) # result(error_msg)

    # class(ty_gas_optics_rrtmgp),
    #                                   intent(in   ) :: this
    # integer,                          intent(in   ) :: ncol, nlay, ngpt, nband
    # real(FT), dimension(:,:),         intent(in   ) :: play,    # layer pressures [Pa, mb]; (ncol,nlay)
    #                                                    plev,    # level pressures [Pa, mb]; (ncol,nlay+1)
    #                                                    tlay      # layer temperatures [K]; (ncol,nlay)
    # type(ty_gas_concs),               intent(in   ) :: gas_desc  # Gas volume mixing ratios
    # class(ty_optical_props_arry),     intent(inout) :: optical_props #inout because components are allocated
    # # Interpolation coefficients for use in internal source function
    # integer,     dimension(                       ncol, nlay), intent(  out) :: jtemp, jpress
    # integer,     dimension(2,    get_nflav(this),ncol, nlay), intent(  out) :: jeta
    # logical(wl), dimension(                       ncol, nlay), intent(  out) :: tropo
    # real(FT),    dimension(2,2,2,get_nflav(this),ncol, nlay), intent(  out) :: fmajor
    # character(len=128)                                         :: error_msg

    # Optional inputs
    # real(FT), dimension(:,:), intent(in   ),
    #                        optional, target :: col_dry # Column dry amount; dim(ncol,nlay)
    # ----------------------------------------------------------
    # Local variables
    # real(FT), dimension(ngpt,nlay,ncol) :: tau, tau_rayleigh  # absorption, Rayleigh scattering optical depths
    # # integer :: igas, idx_h2o # index of some gases
    # # Number of molecules per cm^2
    # real(FT), dimension(ncol,nlay), target  :: col_dry_arr
    # real(FT), dimension(:,:),       pointer :: col_dry_wk
    # #
    # # Interpolation variables used in major gas but not elsewhere, so don't need exporting
    # #
    # real(FT), dimension(ncol,nlay,  this%get_ngas()) :: vmr     # volume mixing ratios
    # real(FT), dimension(ncol,nlay,0:this%get_ngas()) :: col_gas # column amounts for each gas, plus col_dry
    # real(FT), dimension(2,    get_nflav(this),ncol,nlay) :: col_mix # combination of major species's column amounts
    #                                                      # index(1) : reference temperature level
    #                                                      # index(2) : flavor
    #                                                      # index(3) : layer
    # real(FT), dimension(2,2,  get_nflav(this),ncol,nlay) :: fminor # interpolation fractions for minor species
    #                                                       # index(1) : reference eta level (temperature dependent)
    #                                                       # index(2) : reference temperature level
    #                                                       # index(3) : flavor
    #                                                       # index(4) : layer
    # integer :: ngas, nflav, neta, npres, ntemp
    # integer :: nminorlower, nminorklower,nminorupper, nminorkupper
    # logical :: use_rayl
    # ----------------------------------------------------------
@show "entered compute_gas_tau"
    FT = eltype(play)
@show ("FT = ", FT)
    tau = Array{FT}(undef, ngpt,nlay,ncol)          # absorption, Rayleigh scattering optical depths
    tau_rayleigh = Array{FT}(undef, ngpt,nlay,ncol) # absorption, Rayleigh scattering optical depths
    col_dry_arr = Array{FT}(undef, ncol, nlay)
    col_dry_wk = Array{FT}(undef, ncol, nlay)

    vmr     = Array{FT}(undef, ncol,nlay,  get_ngas(this)) # volume mixing ratios
    col_gas = OffsetArray{FT}(undef, 1:ncol,1:nlay,0:get_ngas(this)) # column amounts for each gas, plus col_dry
    col_mix = Array{FT}(undef, 2,    get_nflav(this),ncol,nlay) # combination of major species's column amounts
    fminor  = Array{FT}(undef, 2,2,  get_nflav(this),ncol,nlay) # interpolation fractions for minor species

@show "after assignments / compute_gas_tau"
    #
    # Error checking
    #
    use_rayl = allocated(this.krayl)
    # Check for initialization
    if !is_initialized(this.optical_props)
      error("ERROR: spectral configuration not loaded")
    end
    #
    # Check for presence of key species in ty_gas_concs; return error if any key species are not present
    #
    check_key_species_present(this, gas_desc)
@show "after check_key_species / compute_gas_tau"

    #
    # Check input data sizes and values
    #
    check_extent(play, (ncol, nlay  ),   "play")
    check_extent(plev, (ncol, nlay+1), "plev")
    check_extent(tlay, (ncol, nlay  ),   "tlay")
    check_range(play, this.press_ref_min,this.press_ref_max, "play")
    check_range(plev, this.press_ref_min, this.press_ref_max, "plev")
    check_range(tlay, this.temp_ref_min,  this.temp_ref_max,  "tlay")
    if present(col_dry)
      check_extent(col_dry, (ncol, nlay), "col_dry")
      check_range(col_dry, FT(0), floatmax(FT), "col_dry")
    end
@show "after checking extents, range / compute_gas_tau"

    # ----------------------------------------------------------
    ngas  = get_ngas(this)
    nflav = get_nflav(this)
    neta  = get_neta(this)
    npres = get_npres(this)
    ntemp = get_ntemp(this)
    # number of minor contributors, total num absorption coeffs
    nminorlower  = size(this.minor_scales_with_density_lower)
    nminorklower = size(this.kminor_lower, 1)
    nminorupper  = size(this.minor_scales_with_density_upper)
    nminorkupper = size(this.kminor_upper, 1)
@show ("ngas, nflav, neta, npres, ntemp, etc. range / compute_gas_tau",ngas,nflav,npres,ntemp)
    #
    # Fill out the array of volume mixing ratios
    #
    # error("Done")
    for igas in 1:ngas
      #
      # Get vmr if  gas is provided in ty_gas_concs
      #
      # if any(lower_case(this.gas_names[igas]) == gas_desc.gas_name[:])
      if lowercase(this.gas_names[igas]) in gas_desc.gas_name
         vmr[:,:,igas] = get_vmr(gas_desc, this.gas_names[igas])
      end
    end
    # test_data(vmr, "vmr_after_get")
@show ("after vmr, compute_gas_tau")

    #
    # Compute dry air column amounts (number of molecule per cm^2) if user hasn't provided them
    #
    idx_h2o = string_loc_in_array("h2o", this.gas_names)
    if present(col_dry)
      col_dry_wk = col_dry
    else
      col_dry_arr = get_col_dry(vmr[:,:,idx_h2o], plev, tlay) # dry air column amounts computation
      col_dry_wk = col_dry_arr
    end
    #
@show ("after col_dry_wk, compute_gas_tau")
    # compute column gas amounts [molec/cm^2]
    #
    col_gas[1:ncol,1:nlay,0] .= col_dry_wk[1:ncol,1:nlay]
    for igas = 1:ngas
      col_gas[1:ncol,1:nlay,igas] .= vmr[1:ncol,1:nlay,igas] .* col_dry_wk[1:ncol,1:nlay]
    end
@show ("after col_gas, compute_gas_tau")

    if present(col_dry)
      # test_data(col_dry, "col_dry")
    end
    # test_data(col_dry_arr, "col_dry_arr")
    # test_data(col_dry_wk, "col_dry_wk")
    # test_data(vmr, "vmr_local")
    # test_data(col_gas, "col_gas")


    #
    # ---- calculate gas optical depths ----
    #
    #$acc enter data create(jtemp, jpress, jeta, tropo, fmajor)
    #$acc enter data create(tau, tau_rayleigh)
    #$acc enter data create(col_mix, fminor)
    #$acc enter data copyin(play, tlay, col_gas)
    #$acc enter data copyin(this)
    #$acc enter data copyin(this%gpoint_flavor)
    zero_array!(tau)
@show ("before interpolation, compute_gas_tau")
    jtemp,fmajor,fminor,col_mix,tropo,jeta,jpress = interpolation(
            ncol,nlay,                        # problem dimensions
            ngas, nflav, neta, npres, ntemp,  # interpolation dimensions
            this.flavor,
            this.press_ref_log,
            this.temp_ref,
            this.press_ref_log_delta,
            this.temp_ref_min,
            this.temp_ref_delta,
            this.press_ref_trop_log,
            this.vmr_ref,
            play,
            tlay,
            col_gas)
@show ("after interpolation, compute_gas_tau")
    tau = compute_tau_absorption!(
            ncol,nlay,nband,ngpt,                      # dimensions
            ngas,nflav,neta,npres,ntemp,
            nminorlower, nminorklower,                # number of minor contributors, total num absorption coeffs
            nminorupper, nminorkupper,
            idx_h2o,
            this.gpoint_flavor,
            get_band_lims_gpoint(this.optical_props),
            this.kmajor,
            this.kminor_lower,
            this.kminor_upper,
            this.minor_limits_gpt_lower,
            this.minor_limits_gpt_upper,
            this.minor_scales_with_density_lower,
            this.minor_scales_with_density_upper,
            this.scale_by_complement_lower,
            this.scale_by_complement_upper,
            this.idx_minor_lower,
            this.idx_minor_upper,
            this.idx_minor_scaling_lower,
            this.idx_minor_scaling_upper,
            this.kminor_start_lower,
            this.kminor_start_upper,
            tropo,
            col_mix,fmajor,fminor,
            play,tlay,col_gas,
            jeta,jtemp,jpress)
@show ("after compute_tau_absorption, compute_gas_tau")
@show this.krayl
@show allocated(this.krayl)
    if allocated(this.krayl)
      #$acc enter data attach(col_dry_wk) copyin(this%krayl)
      compute_tau_rayleigh!(          #Rayleigh scattering optical depths
            ncol,nlay,nband,ngpt,
            ngas,nflav,neta,npres,ntemp,  # dimensions
            this.gpoint_flavor,
            get_band_lims_gpoint(this.optical_props),
            this.krayl,                   # inputs from object
            idx_h2o, col_dry_wk,col_gas,
            fminor,jeta,tropo,jtemp,      # local input
            tau_rayleigh)
      @show ("after compute_tau_rayleigh, compute_gas_tau")
      #$acc exit data detach(col_dry_wk) delete(this%krayl)
    end
@show ("before combine_and_reorder, compute_gas_tau")

    # Combine optical depths and reorder for radiative transfer solver.
    # test_data(tau_rayleigh, "tau_rayleigh")
    # test_data(tau, "tau_before_CAR")
    combine_and_reorder!(tau, tau_rayleigh, allocated(this.krayl), optical_props)
@show ("after combine_and_reorder, compute_gas_tau")
    #$acc exit data delete(tau, tau_rayleigh)
    #$acc exit data delete(play, tlay, col_gas)
    #$acc exit data delete(col_mix, fminor)
    #$acc exit data delete(this%gpoint_flavor)
    #$acc exit data copyout(jtemp, jpress, jeta, tropo, fmajor)
    return jtemp, jpress, jeta, tropo, fmajor
  end
  #------------------------------------------------------------------------------------------
  #
  # Compute Planck source functions at layer centers and levels
  #
  function source(this::ty_gas_optics_rrtmgp,
                  ncol, nlay, nbnd, ngpt,
                  play, plev, tlay, tsfc,
                  jtemp, jpress, jeta, tropo, fmajor,
                  sources::ty_source_func_lw,          # Planck sources
                  tlev)                                # optional input
                  #result(error_msg)
    # # inputs
    # class(ty_gas_optics_rrtmgp),    intent(in ) :: this
    # integer,                               intent(in   ) :: ncol, nlay, nbnd, ngpt
    # real(FT), dimension(ncol,nlay),        intent(in   ) :: play   # layer pressures [Pa, mb]
    # real(FT), dimension(ncol,nlay+1),      intent(in   ) :: plev   # level pressures [Pa, mb]
    # real(FT), dimension(ncol,nlay),        intent(in   ) :: tlay   # layer temperatures [K]
    # real(FT), dimension(ncol),             intent(in   ) :: tsfc   # surface skin temperatures [K]
    # # Interplation coefficients
    # integer,     dimension(ncol,nlay),     intent(in   ) :: jtemp, jpress
    # logical(wl), dimension(ncol,nlay),     intent(in   ) :: tropo
    # real(FT),    dimension(2,2,2,get_nflav(this),ncol,nlay),
    #                                        intent(in   ) :: fmajor
    # integer,     dimension(2,    get_nflav(this),ncol,nlay),
    #                                        intent(in   ) :: jeta
    # class(ty_source_func_lw    ),          intent(inout) :: sources
    # real(FT), dimension(ncol,nlay+1),      intent(in   ),
    #                                   optional, target :: tlev          # level temperatures [K]
    # character(len=128)                                 :: error_msg
    # ----------------------------------------------------------
    # integer                                      :: icol, ilay, igpt
    # real(FT), dimension(ngpt,nlay,ncol)          :: lay_source_t, lev_source_inc_t, lev_source_dec_t
    # real(FT), dimension(ngpt,     ncol)          :: sfc_source_t
    # # Variables for temperature at layer edges [K] (ncol, nlay+1)
    # real(FT), dimension(   ncol,nlay+1), target  :: tlev_arr
    # real(FT), dimension(:,:),            pointer :: tlev_wk
    FT = eltype(this.vmr_ref) # Float64

    lay_source_t = Array{FT}(undef, ngpt,nlay,ncol)
    lev_source_inc_t = Array{FT}(undef, ngpt,nlay,ncol)
    lev_source_dec_t = Array{FT}(undef, ngpt,nlay,ncol)
    sfc_source_t = Array{FT}(undef, ngpt,ncol)
    tlev_arr = Array{FT}(undef, ncol,nlay+1)

    # ----------------------------------------------------------
    #
    # Source function needs temperature at interfaces/levels and at layer centers
    #
    if present(tlev)
      #   Users might have provided these
      tlev_wk = tlev
    else
      tlev_wk = tlev_arr
      #
      # Interpolate temperature to levels if not provided
      #   Interpolation and extrapolation at boundaries is weighted by pressure
      #
      for icol = 1:ncol
         tlev_arr[icol,1] = tlay[icol,1] +
                             (plev[icol,1]-play[icol,1])*(tlay[icol,2]-tlay[icol,1]) /
                                                         (play[icol,2]-play[icol,1])
      end
      for ilay in 2:nlay
        for icol in 1:ncol
           tlev_arr[icol,ilay] = (play[icol,ilay-1]*tlay[icol,ilay-1]*(plev[icol,ilay]-play[icol,ilay]) +
                                  play[icol,ilay]*tlay[icol,ilay]*(play[icol,ilay-1]-plev[icol,ilay])) /
                                  (plev[icol,ilay]*(play[icol,ilay-1] - play[icol,ilay]))
        end
      end
      for icol = 1:ncol
         tlev_arr[icol,nlay+1] = tlay[icol,nlay] +
                                 (plev[icol,nlay+1]-play[icol,nlay])*(tlay[icol,nlay]-tlay(icol,nlay-1)) /
                                                                       (play(icol,nlay)-play(icol,nlay-1))
      end
    end

    #-------------------------------------------------------------------
    # Compute internal (Planck) source functions at layers and levels,
    #  which depend on mapping from spectral space that creates k-distribution.
    #$acc enter data copyin(sources)
    #$acc enter data create(sources%lay_source, sources%lev_source_inc, sources%lev_source_dec, sources%sfc_source)
    #$acc enter data create(sfc_source_t, lay_source_t, lev_source_inc_t, lev_source_dec_t) attach(tlev_wk)
    compute_Planck_source!(ncol, nlay, nbnd, ngpt,
                get_nflav(this), get_neta(this), get_npres(this), get_ntemp(this), get_nPlanckTemp(this),
                tlay, tlev_wk, tsfc, fmerge(1,nlay,play[1,1] > play[1,nlay]),
                fmajor, jeta, tropo, jtemp, jpress,
                get_gpoint_bands(this.optical_props), get_band_lims_gpoint(this.optical_props), this.planck_frac, this.temp_ref_min,
                this.totplnk_delta, this.totplnk, this.gpoint_flavor,
                sfc_source_t, lay_source_t, lev_source_inc_t, lev_source_dec_t)
    #$acc parallel loop collapse(2)
    for igpt in 1:ngpt
      for icol in 1:ncol
        sources.sfc_source[icol,igpt] = sfc_source_t[igpt,icol]
      end
    end
    reorder123x321!(lay_source_t, sources.lay_source)
    reorder123x321!(lev_source_inc_t, sources.lev_source_inc)
    reorder123x321!(lev_source_dec_t, sources.lev_source_dec)
    #$acc exit data delete(sfc_source_t, lay_source_t, lev_source_inc_t, lev_source_dec_t) detach(tlev_wk)
    #$acc exit data copyout(sources%lay_source, sources%lev_source_inc, sources%lev_source_dec, sources%sfc_source)
    #$acc exit data copyout(sources)
  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # Initialization
  #
  #--------------------------------------------------------------------------------------------------------------------
  # Initialize object based on data read from netCDF file however the user desires.
  #  Rayleigh scattering tables may or may not be present; this is indicated with allocation status
  # This interface is for the internal-sources object -- includes Plank functions and fractions
  #
  function load_totplnk(totplnk, planck_frac, rayl_lower, rayl_upper, args...) #result(err_message)
    # class(ty_gas_optics_rrtmgp),     intent(inout) :: this
    # class(ty_gas_concs),                    intent(in   ) :: available_gases # Which gases does the host model have available?
    # character(len=*),   dimension(:),       intent(in   ) :: gas_names
    # integer,            dimension(:,:,:),   intent(in   ) :: key_species
    # integer,            dimension(:,:),     intent(in   ) :: band2gpt
    # real(FT),           dimension(:,:),     intent(in   ) :: band_lims_wavenum
    # real(FT),           dimension(:),       intent(in   ) :: press_ref, temp_ref
    # real(FT),                               intent(in   ) :: press_ref_trop, temp_ref_p, temp_ref_t
    # real(FT),           dimension(:,:,:),   intent(in   ) :: vmr_ref
    # real(FT),           dimension(:,:,:,:), intent(in   ) :: kmajor
    # real(FT),           dimension(:,:,:),   intent(in   ) :: kminor_lower, kminor_upper
    # real(FT),           dimension(:,:),     intent(in   ) :: totplnk
    # real(FT),           dimension(:,:,:,:), intent(in   ) :: planck_frac
    # real(FT),           dimension(:,:,:),   intent(in   ),
    #                                           allocatable :: rayl_lower, rayl_upper
    # character(len=*),   dimension(:),       intent(in   ) :: gas_minor,identifier_minor
    # character(len=*),   dimension(:),       intent(in   ) :: minor_gases_lower,
    #                                                          minor_gases_upper
    # integer,            dimension(:,:),     intent(in   ) :: minor_limits_gpt_lower,
    #                                                          minor_limits_gpt_upper
    # logical(wl),        dimension(:),       intent(in   ) :: minor_scales_with_density_lower,
    #                                                          minor_scales_with_density_upper
    # character(len=*),   dimension(:),       intent(in   ) :: scaling_gas_lower,
    #                                                          scaling_gas_upper
    # logical(wl),        dimension(:),       intent(in   ) :: scale_by_complement_lower,
    #                                                          scale_by_complement_upper
    # integer,            dimension(:),       intent(in   ) :: kminor_start_lower,
    #                                                          kminor_start_upper
    # character(len = 128) :: err_message
    # # ----
    this = init_abs_coeffs(rayl_lower, rayl_upper, args...)
    # Planck function tables
    #
    this.totplnk = totplnk
    this.planck_frac = planck_frac
    # Temperature steps for Planck function interpolation
    #   Assumes that temperature minimum and max are the same for the absorption coefficient grid and the
    #   Planck grid and the Planck grid is equally spaced
    this.totplnk_delta =  (this.temp_ref_max-this.temp_ref_min) / (size(this.totplnk, 1)-1)
    return this
  end

  #--------------------------------------------------------------------------------------------------------------------
  #
  # Initialize object based on data read from netCDF file however the user desires.
  #  Rayleigh scattering tables may or may not be present; this is indicated with allocation status
  # This interface is for the external-sources object -- includes TOA source function table
  #
  function load_solar_source(solar_src, rayl_lower, rayl_upper, args...)
    # class(ty_gas_optics_rrtmgp), intent(inout) :: this
    # class(ty_gas_concs),                intent(in   ) :: available_gases # Which gases does the host model have available?
    # character(len=*),
    #           dimension(:),       intent(in) :: gas_names
    # integer,  dimension(:,:,:),   intent(in) :: key_species
    # integer,  dimension(:,:),     intent(in) :: band2gpt
    # real(FT), dimension(:,:),     intent(in) :: band_lims_wavenum
    # real(FT), dimension(:),       intent(in) :: press_ref, temp_ref
    # real(FT),                     intent(in) :: press_ref_trop, temp_ref_p, temp_ref_t
    # real(FT), dimension(:,:,:),   intent(in) :: vmr_ref
    # real(FT), dimension(:,:,:,:), intent(in) :: kmajor
    # real(FT), dimension(:,:,:),   intent(in) :: kminor_lower, kminor_upper
    # character(len=*),   dimension(:),
    #                               intent(in) :: gas_minor,
    #                                             identifier_minor
    # character(len=*),   dimension(:),
    #                               intent(in) :: minor_gases_lower,
    #                                             minor_gases_upper
    # integer,  dimension(:,:),     intent(in) ::
    #                                             minor_limits_gpt_lower,
    #                                             minor_limits_gpt_upper
    # logical(wl), dimension(:),    intent(in) ::
    #                                             minor_scales_with_density_lower,
    #                                             minor_scales_with_density_upper
    # character(len=*),   dimension(:),intent(in) ::
    #                                             scaling_gas_lower,
    #                                             scaling_gas_upper
    # logical(wl), dimension(:),    intent(in) ::
    #                                             scale_by_complement_lower,
    #                                             scale_by_complement_upper
    # integer,  dimension(:),       intent(in) ::
    #                                             kminor_start_lower,
    #                                             kminor_start_upper
    # real(FT), dimension(:),       intent(in), allocatable :: solar_src
    #                                                         # allocatable status to change when solar source is present in file
    # real(FT), dimension(:,:,:), intent(in), allocatable :: rayl_lower, rayl_upper
    # character(len = 128) err_message
    # ----

    this = init_abs_coeffs(rayl_lower, rayl_upper, args...)
    #
    # Solar source table init
    #
    this.solar_src = solar_src
    return this

  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # Initialize absorption coefficient arrays,
  #   including Rayleigh scattering tables if provided (allocated)
  #
  function init_abs_coeffs(rayl_lower, rayl_upper,
available_gases::ty_gas_concs,
gas_names,
key_species,
band2gpt,
band_lims_wavenum,
press_ref,
press_ref_trop,
temp_ref,
temp_ref_p,
temp_ref_t,
vmr_ref,
kmajor,
kminor_lower,
kminor_upper,
gas_minor,
identifier_minor,
minor_gases_lower,
minor_gases_upper,
minor_limits_gpt_lower,
minor_limits_gpt_upper,
minor_scales_with_density_lower,
minor_scales_with_density_upper,
scaling_gas_lower,
scaling_gas_upper,
scale_by_complement_lower,
scale_by_complement_upper,
kminor_start_lower,
kminor_start_upper)
    # class(ty_gas_optics_rrtmgp), intent(inout) :: this
    # class(ty_gas_concs),                intent(in   ) :: available_gases
    # character(len=*),
    #           dimension(:),       intent(in) :: gas_names
    # integer,  dimension(:,:,:),   intent(in) :: key_species
    # integer,  dimension(:,:),     intent(in) :: band2gpt
    # real(FT), dimension(:,:),     intent(in) :: band_lims_wavenum
    # real(FT), dimension(:),       intent(in) :: press_ref, temp_ref
    # real(FT),                     intent(in) :: press_ref_trop, temp_ref_p, temp_ref_t
    # real(FT), dimension(:,:,:),   intent(in) :: vmr_ref
    # real(FT), dimension(:,:,:,:), intent(in) :: kmajor
    # real(FT), dimension(:,:,:),   intent(in) :: kminor_lower, kminor_upper
    # character(len=*),   dimension(:),
    #                               intent(in) :: gas_minor,
    #                                             identifier_minor
    # character(len=*),   dimension(:),
    #                               intent(in) :: minor_gases_lower,
    #                                             minor_gases_upper
    # integer,  dimension(:,:),     intent(in) :: minor_limits_gpt_lower,
    #                                             minor_limits_gpt_upper
    # logical(wl), dimension(:),    intent(in) :: minor_scales_with_density_lower,
    #                                             minor_scales_with_density_upper
    # character(len=*),   dimension(:),
    #                               intent(in) :: scaling_gas_lower,
    #                                             scaling_gas_upper
    # logical(wl), dimension(:),    intent(in) :: scale_by_complement_lower,
    #                                             scale_by_complement_upper
    # integer,  dimension(:),       intent(in) :: kminor_start_lower,
    #                                             kminor_start_upper
    # real(FT), dimension(:,:,:),   intent(in),
    #                              allocatable :: rayl_lower, rayl_upper
    # character(len=128)                       :: err_message
    # --------------------------------------------------------------------------
    # logical,  dimension(:),     allocatable :: gas_is_present
    # logical,  dimension(:),     allocatable :: key_species_present_init
    # integer,  dimension(:,:,:), allocatable :: key_species_red
    # real(FT), dimension(:,:,:), allocatable :: vmr_ref_red
    # character(len=256),
    #           dimension(:),     allocatable :: minor_gases_lower_red,
    #                                            minor_gases_upper_red
    # character(len=256),
    #           dimension(:),     allocatable :: scaling_gas_lower_red,
    #                                            scaling_gas_upper_red
    # integer :: i, j, idx
    # integer :: ngas
    # # --------------------------------------
    FT = eltype(kmajor)

    this = ty_gas_optics_rrtmgp(FT,Int)
    init!(this.optical_props, "ty_gas_optics_rrtmgp optical props", band_lims_wavenum, band2gpt)
    #
    # Which gases known to the gas optics are present in the host model (available_gases)?
    #
    ngas = length(gas_names)
    gas_is_present = Vector{Bool}(undef, ngas...)

    for i in 1:ngas
      gas_is_present[i] = string_in_array(gas_names[i], available_gases.gas_name)
    end
    #
    # Now the number of gases is the union of those known to the k-distribution and provided
    #   by the host model
    #
    ngas = count(gas_is_present)
    #
    # Initialize the gas optics object, keeping only those gases known to the
    #   gas optics and also present in the host model
    #
    this.gas_names = pack(gas_names, gas_is_present)

    # vmr_ref_red = Array{FT}(undef, size(vmr_ref, 1),0:ngas, size(vmr_ref, 3))
    vmr_ref_red = OffsetArray{FT}(undef, 1:size(vmr_ref, 1),0:ngas, 1:size(vmr_ref, 3))

    # Gas 0 is used in single-key species method, set to 1.0 (col_dry)
    vmr_ref_red[:,0,:] = vmr_ref[:,1,:]
    for i = 1:ngas
      idx = string_loc_in_array(this.gas_names[i], gas_names)
      vmr_ref_red[:,i,:] = vmr_ref[:,idx+1,:]
    end
    this.vmr_ref = vmr_ref_red
    # test_data(this.vmr_ref, "vmr_ref")
    #
    # Reduce minor arrays so variables only contain minor gases that are available
    # Reduce size of minor Arrays
    #

    this.kminor_lower,
    minor_gases_lower_red,
    this.minor_limits_gpt_lower,
    this.minor_scales_with_density_lower,
    scaling_gas_lower_red,
    this.scale_by_complement_lower,
    this.kminor_start_lower              = reduce_minor_arrays(available_gases,
gas_names,                            # gas_names,
gas_minor,                            # gas_minor,
identifier_minor,                     # identifier_minor,
kminor_lower,                         # kminor_atm,
minor_gases_lower,                    # minor_gases_atm,
minor_limits_gpt_lower,               # minor_limits_gpt_atm,
minor_scales_with_density_lower,      # minor_scales_with_density_atm,
scaling_gas_lower,                    # scaling_gas_atm,
scale_by_complement_lower,            # scale_by_complement_atm,
kminor_start_lower                   # kminor_start_atm,
# this.kminor_lower,                    # kminor_atm_red,
# minor_gases_lower_red,                # minor_gases_atm_red,
# this.minor_limits_gpt_lower,          # minor_limits_gpt_atm_red,
# this.minor_scales_with_density_lower, # minor_scales_with_density_atm_red,
# scaling_gas_lower_red,                # scaling_gas_atm_red,
# this.scale_by_complement_lower,       # scale_by_complement_atm_red,
# this.kminor_start_lower               # kminor_start_atm_red
)

    # test_data(this.kminor_lower, "kminor_lower")
    # test_data(this.minor_limits_gpt_lower, "minor_limits_gpt_lower")
    # test_data(this.kminor_start_lower, "kminor_start_lower")

    this.kminor_upper,
    minor_gases_upper_red,
    this.minor_limits_gpt_upper,
    this.minor_scales_with_density_upper,
    scaling_gas_upper_red,
    this.scale_by_complement_upper,
    this.kminor_start_upper = reduce_minor_arrays(available_gases,
                             gas_names,
                             gas_minor,
                             identifier_minor,
                             kminor_upper,
                             minor_gases_upper,
                             minor_limits_gpt_upper,
                             minor_scales_with_density_upper,
                             scaling_gas_upper,
                             scale_by_complement_upper,
                             kminor_start_upper
                             # this.kminor_upper,
                             # minor_gases_upper_red,
                             # this.minor_limits_gpt_upper,
                             # this.minor_scales_with_density_upper,
                             # scaling_gas_upper_red,
                             # this.scale_by_complement_upper,
                             # this.kminor_start_upper
                             )

    # test_data(this.kminor_upper,                    "kminor_upper")
    # test_data(this.minor_limits_gpt_upper,          "minor_limits_gpt_upper")
    # test_data(this.kminor_start_upper,              "kminor_start_upper")

    # Arrays not reduced by the presence, or lack thereof, of a gas
    this.press_ref = press_ref
    this.temp_ref  = temp_ref
    this.kmajor    = kmajor
    FT = eltype(kmajor)
    # TODO: Check if .neqv. is the same as ≠
    if allocated(rayl_lower) ≠ allocated(rayl_upper)
      error("rayl_lower and rayl_upper must have the same allocation status")
    end
    if allocated(rayl_lower)
      this.krayl = Array{FT}(undef, size(rayl_lower,1),size(rayl_lower,2),size(rayl_lower,3),2)
      this.krayl[:,:,:,1] = rayl_lower
      this.krayl[:,:,:,2] = rayl_upper
    end

    # ---- post processing ----
    # Incoming coefficients file has units of Pa
    this.press_ref .= this.press_ref

    # creates log reference pressure
    this.press_ref_log = log.(this.press_ref)

    # log scale of reference pressure
    this.press_ref_trop_log = log(press_ref_trop)

    # Get index of gas (if present) for determining col_gas
    this.idx_minor_lower = create_idx_minor(this.gas_names, gas_minor, identifier_minor, minor_gases_lower_red)
    this.idx_minor_upper = create_idx_minor(this.gas_names, gas_minor, identifier_minor, minor_gases_upper_red)
    # test_data(this.idx_minor_lower, "idx_minor_lower")
    # test_data(this.idx_minor_lower, "idx_minor_lower")
    # Get index of gas (if present) that has special treatment in density scaling
    this.idx_minor_scaling_lower = create_idx_minor_scaling(this.gas_names, scaling_gas_lower_red)
    this.idx_minor_scaling_upper = create_idx_minor_scaling(this.gas_names, scaling_gas_upper_red)

    # create flavor list
    # Reduce (remap) key_species list; checks that all key gases are present in incoming
    key_species_red,key_species_present_init = create_key_species_reduce(gas_names,this.gas_names, key_species)
    check_key_species_present_init(gas_names,key_species_present_init)
    # test_data(key_species_red, "key_species_red")

    # create flavor list
    this.flavor = create_flavor(key_species_red)
    # test_data(this.flavor, "flavor")
    # create gpoint_flavor list
    this.gpoint_flavor = create_gpoint_flavor(key_species_red, get_gpoint_bands(this.optical_props), this.flavor)
    # test_data(this.gpoint_flavor, "gpoint_flavor")

    # minimum, maximum reference temperature, pressure -- assumes low-to-high ordering
    #   for T, high-to-low ordering for p
    this.temp_ref_min  = this.temp_ref[1]
    this.temp_ref_max  = this.temp_ref[length(this.temp_ref)]
    this.press_ref_min = this.press_ref[length(this.press_ref)]
    this.press_ref_max = this.press_ref[1]

    # creates press_ref_log, temp_ref_delta
    this.press_ref_log_delta = (log(this.press_ref_min)-log(this.press_ref_max))/(length(this.press_ref)-1)
    this.temp_ref_delta      = (this.temp_ref_max-this.temp_ref_min)/(length(this.temp_ref)-1)

    # Which species are key in one or more bands?
    #   this%flavor is an index into this%gas_names
    #
    this.is_key = [false for i in 1:get_ngas(this)]
    for j in 1:size(this.flavor, 2)
      for i in 1:size(this.flavor, 1) # should be 2
        if this.flavor[i,j] ≠ 0
          this.is_key[this.flavor[i,j]] = true
        end
      end
    end
    # test_data(this.is_key, "is_key")
    return this

  end
  # ----------------------------------------------------------------------------------------------------
  function check_key_species_present_init(gas_names, key_species_present_init) # result(err_message)
    # logical,          dimension(:), intent(in) :: key_species_present_init
    # character(len=*), dimension(:), intent(in) :: gas_names
    # character(len=128)                             :: err_message

    # integer :: i

    for i in 1:length(key_species_present_init)
      if !key_species_present_init[i]
        error("gas_optics: required gases" * trim(gas_names[i]) * " are not provided")
      end
    end
  end
  #------------------------------------------------------------------------------------------
  #
  # Ensure that every key gas required by the k-distribution is
  #    present in the gas concentration object
  #
  function check_key_species_present(this::ty_gas_optics_rrtmgp, gas_desc::ty_gas_concs) # result(error_msg)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # class(ty_gas_concs),                intent(in) :: gas_desc
    # character(len=128)                             :: error_msg

    # # Local variables
    # character(len=32), dimension(count(this%is_key(:)  )) :: key_gas_names
    # integer                                               :: igas
    # # --------------------------------------
    key_gas_names = pack(this.gas_names, this.is_key)
    for igas = 1:length(key_gas_names)
      if !string_in_array(key_gas_names[igas], gas_desc.gas_name)
        error("gas_optics: required gases" * trim(key_gas_names[igas]) * " are not provided")
      end
    end
  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # Function to define names of key and minor gases to be used by gas_optics().
  # The final list gases includes those that are defined in gas_optics_specification
  # and are provided in ty_gas_concs.
  #
  function get_minor_list(this::ty_gas_optics_rrtmgp, gas_desc::ty_gas_concs, ngas, names_spec)
    # class(ty_gas_optics_rrtmgp), intent(in)       :: this
    # class(ty_gas_concs), intent(in)                      :: gas_desc
    # integer, intent(in)                                  :: ngas
    # character(32), dimension(ngas), intent(in)           :: names_spec

    # # List of minor gases to be used in gas_optics()
    # character(len=32), dimension(:), allocatable         :: get_minor_list
    # # Logical flag for minor species in specification (T = minor; F = not minor)
    # logical, dimension(size(names_spec))                 :: gas_is_present
    # integer                                              :: igas, icnt

    allocated(get_minor_list) && deallocate!(get_minor_list)
    for igas = 1:get_ngas(this)
      gas_is_present[igas] = string_in_array(names_spec[igas], gas_desc.gas_name)
    end
    icnt = count(gas_is_present)
    get_minor_list = Vector{String}(undef, icnt)
    get_minor_list[:] .= pack(this.gas_names, gas_is_present)
  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # Inquiry functions
  #
  #--------------------------------------------------------------------------------------------------------------------
  #
  # return true if initialized for internal sources, false otherwise
  #
  function source_is_internal(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # logical                          :: source_is_internal
    return allocated(this.totplnk) && allocated(this.planck_frac)

#    return size(this.totplnk)>0 && size(this.planck_frac)>0

  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # return true if initialized for external sources, false otherwise
  #
  function source_is_external(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # logical                          :: source_is_external
    return allocated(this.solar_src)
  end

  #--------------------------------------------------------------------------------------------------------------------
  #
  # return the gas names
  #
  function get_gases(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # character(32), dimension(get_ngas(this))     :: get_gases

    return this.gas_names
  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # return the minimum pressure on the interpolation grids
  #
  function get_press_min(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # real(FT)                                       :: get_press_min

    return this.press_ref_min
  end

  #--------------------------------------------------------------------------------------------------------------------
  #
  # return the maximum pressure on the interpolation grids
  #
  function get_press_max(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # real(FT)                                       :: get_press_max

    return this.press_ref_max
  end

  #--------------------------------------------------------------------------------------------------------------------
  #
  # return the minimum temparature on the interpolation grids
  #
  function get_temp_min(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # real(FT)                                       :: get_temp_min

    return this.temp_ref_min
  end

  #--------------------------------------------------------------------------------------------------------------------
  #
  # return the maximum temparature on the interpolation grids
  #
  function get_temp_max(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # real(FT)                                       :: get_temp_max

    return this.temp_ref_max
  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # Utility function, provided for user convenience
  # computes column amounts of dry air using hydrostatic equation
  #
  function get_col_dry(vmr_h2o, plev, tlay, latitude=nothing) # result(col_dry)
    # # input
    # real(FT), dimension(:,:), intent(in) :: vmr_h2o  # volume mixing ratio of water vapor to dry air; (ncol,nlay)
    # real(FT), dimension(:,:), intent(in) :: plev     # Layer boundary pressures [Pa] (ncol,nlay+1)
    # real(FT), dimension(:,:), intent(in) :: tlay     # Layer temperatures [K] (ncol,nlay)
    # real(FT), dimension(:),   optional,
    #                           intent(in) :: latitude # Latitude [degrees] (ncol)
    # # output
    # real(FT), dimension(size(tlay,dim=1),size(tlay,dim=2)) :: col_dry # Column dry amount (ncol,nlay)
    # ------------------------------------------------
    # # first and second term of Helmert formula
    # real(FT), parameter :: helmert1 = 9.80665_wp
    # real(FT), parameter :: helmert2 = 0.02586_wp
    # # local variables
    # real(FT), dimension(size(tlay,dim=1)                 ) :: g0 # (ncol)
    # real(FT), dimension(size(tlay,dim=1),size(tlay,dim=2)) :: delta_plev # (ncol,nlay)
    # real(FT), dimension(size(tlay,dim=1),size(tlay,dim=2)) :: m_air # average mass of air; (ncol,nlay)
    # integer :: nlev, nlay
    # # ------------------------------------------------
    FT = eltype(plev)

    # first and second term of Helmert formula
    helmert1 = FT(9.80665)
    helmert2 = FT(0.02586)
    # local variables
    g0         = Array{FT}(undef, size(tlay,1)             ) # (ncol)
    delta_plev = Array{FT}(undef, size(tlay,1),size(tlay,2)) # (ncol,nlay)
    m_air      = Array{FT}(undef, size(tlay,1),size(tlay,2)) # average mass of air; (ncol,nlay)
    # integer :: nlev, nlay
    # ------------------------------------------------
    nlay = size(tlay, 2)
    nlev = size(plev, 2)

    if present(latitude)
      g0[:] .= helmert1 - helmert2 * cos(FT(2) * π * latitude[:] / FT(180)) # acceleration due to gravity [m/s^2]
    else
      g0[:] .= grav(FT)
    end
    delta_plev[:,:] .= abs.(plev[:,1:nlev-1] .- plev[:,2:nlev])

    # Get average mass of moist air per mole of moist air
    m_air[:,:] .= (m_dry(FT) .+ m_h2o(FT) .* vmr_h2o[:,:]) ./ (1 .+ vmr_h2o[:,:])

    # Hydrostatic equation
    col_dry = Array{FT}(undef, size(tlay,1),size(tlay,2))
    col_dry[:,:] .= FT(10) .* delta_plev[:,:] .* avogad(FT) ./ (FT(1000)*m_air[:,:] .* FT(100) .* spread(g0[:], 2, nlay))
    col_dry[:,:] .= col_dry[:,:] ./ (FT(1) .+ vmr_h2o[:,:])
    return col_dry
  end
  #--------------------------------------------------------------------------------------------------------------------
  #
  # Internal procedures
  #
  #--------------------------------------------------------------------------------------------------------------------
  function rewrite_key_species_pair(key_species_pair)
    # (0,0) becomes (2,2) -- because absorption coefficients for these g-points will be 0.
    # integer, dimension(2) :: rewrite_key_species_pair
    # integer, dimension(2), intent(in) :: key_species_pair
    result = key_species_pair
    if all(key_species_pair[:] .== [0,0])
      result[:] .= [2,2]
    end
    return result
  end

  # ---------------------------------------------------------------------------------------
  # true is key_species_pair exists in key_species_list
  function key_species_pair_exists(key_species_list, key_species_pair)
    # logical                             :: key_species_pair_exists
    # integer, dimension(:,:), intent(in) :: key_species_list
    # integer, dimension(2),   intent(in) :: key_species_pair
    # integer :: i
    for i=1:size(key_species_list,2)
      if all(key_species_list[:,i] .== key_species_pair[:])
        result = true
        return result
      end
    end
    result = false
    return result
  end
  # ---------------------------------------------------------------------------------------
  # create flavor list --
  #   an unordered array of extent (2,:) containing all possible pairs of key species
  #   used in either upper or lower atmos
  #
  function create_flavor(key_species)
    # integer, dimension(:,:,:), intent(in) :: key_species
    # integer, dimension(:,:), allocatable, intent(out) :: flavor
    # integer, dimension(2,size(key_species,3)*2) :: key_species_list
    # integer :: ibnd, iatm, i, iflavor

    key_species_list = Array{Int}(undef, 2,size(key_species,3)*2)

    # prepare list of key_species
    i = 1
    for ibnd=1:size(key_species,3)
      for iatm=1:size(key_species,1)
        key_species_list[:,i] .= key_species[:,iatm,ibnd]
        i = i + 1
      end
    end
    # rewrite single key_species pairs
    for i=1:size(key_species_list,2)
        key_species_list[:,i] = rewrite_key_species_pair(key_species_list[:,i])
    end
    # count unique key species pairs
    iflavor = 0
    for i=1:size(key_species_list,2)
      if !key_species_pair_exists(key_species_list[:,1:i-1],key_species_list[:,i])
        iflavor = iflavor + 1
      end
    end
    # fill flavors
    flavor = Array{Int}(undef, 2,iflavor)
    iflavor = 0
    for i=1:size(key_species_list,2)
      if !key_species_pair_exists(key_species_list[:,1:i-1],key_species_list[:,i])
        iflavor = iflavor + 1
        flavor[:,iflavor] = key_species_list[:,i]
      end
    end
    return flavor
  end
  # ---------------------------------------------------------------------------------------
  #
  # create index list for extracting col_gas needed for minor gas optical depth calculations
  #
  function create_idx_minor(gas_names, gas_minor, identifier_minor, minor_gases_atm)
    # character(len=*), dimension(:), intent(in) :: gas_names
    # character(len=*), dimension(:), intent(in) ::
    #                                               gas_minor,
    #                                               identifier_minor
    # character(len=*), dimension(:), intent(in) :: minor_gases_atm
    # integer, dimension(:), allocatable,
    #                                intent(out) :: idx_minor_atm

    # # local
    # integer :: imnr
    # integer :: idx_mnr
    idx_minor_atm = Vector{Int}(undef, size(minor_gases_atm,1))
    for imnr = 1:size(minor_gases_atm,1) # loop over minor absorbers in each band
          # Find identifying string for minor species in list of possible identifiers (e.g. h2o_slf)
          idx_mnr     = string_loc_in_array(minor_gases_atm[imnr], identifier_minor)
          # Find name of gas associated with minor species identifier (e.g. h2o)
          idx_minor_atm[imnr] = string_loc_in_array(gas_minor[idx_mnr],    gas_names)
    end
    return idx_minor_atm

  end

  # ---------------------------------------------------------------------------------------
  #
  # create index for special treatment in density scaling of minor gases
  #
  function create_idx_minor_scaling(gas_names, scaling_gas_atm)
    # character(len=*), dimension(:), intent(in) :: gas_names
    # character(len=*), dimension(:), intent(in) :: scaling_gas_atm
    # integer, dimension(:), allocatable,
    #                                intent(out) :: idx_minor_scaling_atm

    # # local
    # integer :: imnr
    idx_minor_scaling_atm = Vector{Int}(undef, size(scaling_gas_atm,1))
    for imnr = 1:size(scaling_gas_atm,1) # loop over minor absorbers in each band
          # This will be -1 if there's no interacting gas
          idx_minor_scaling_atm[imnr] = string_loc_in_array(scaling_gas_atm[imnr], gas_names)
    end
    return idx_minor_scaling_atm

  end
  # ---------------------------------------------------------------------------------------
  function create_key_species_reduce(gas_names, gas_names_red, key_species)
    # character(len=*),
    #           dimension(:),       intent(in) :: gas_names
    # character(len=*),
    #           dimension(:),       intent(in) :: gas_names_red
    # integer,  dimension(:,:,:),   intent(in) :: key_species
    # integer,  dimension(:,:,:), allocatable, intent(out) :: key_species_red

    # logical, dimension(:), allocatable, intent(out) :: key_species_present_init
    # integer :: ip, ia, it, np, na, nt

    np = size(key_species,1)
    na = size(key_species,2)
    nt = size(key_species,3)
    key_species_red = Array{Int}(undef, size(key_species,1),
                                        size(key_species,2),
                                        size(key_species,3))
    key_species_present_init = Vector{Bool}(undef, size(gas_names))
    key_species_present_init = true

    for ip = 1:np
      for ia = 1:na
        for it = 1:nt
          if key_species[ip,ia,it] ≠ 0
            key_species_red[ip,ia,it] = string_loc_in_array(gas_names[key_species[ip,ia,it]],gas_names_red)
            if key_species_red[ip,ia,it] == -1
              key_species_present_init[key_species[ip,ia,it]] = false
            end
          else
            key_species_red[ip,ia,it] = key_species[ip,ia,it]
          end
        end
      end
    end
    return key_species_red,key_species_present_init

  end

# ---------------------------------------------------------------------------------------
  function reduce_minor_arrays(available_gases::ty_gas_concs,
gas_names,
gas_minor,
identifier_minor,
kminor_atm,
minor_gases_atm,
minor_limits_gpt_atm,
minor_scales_with_density_atm,
scaling_gas_atm,
scale_by_complement_atm,
kminor_start_atm
# kminor_atm_red,
# minor_gases_atm_red,
# minor_limits_gpt_atm_red,
# minor_scales_with_density_atm_red,
# scaling_gas_atm_red,
# scale_by_complement_atm_red,
# kminor_start_atm_red
)

    # class(ty_gas_concs),                intent(in   ) :: available_gases
    # character(len=*), dimension(:),     intent(in) :: gas_names
    # real(FT),         dimension(:,:,:), intent(in) :: kminor_atm
    # character(len=*), dimension(:),     intent(in) :: gas_minor,
    #                                                   identifier_minor
    # character(len=*), dimension(:),     intent(in) :: minor_gases_atm
    # integer,          dimension(:,:),   intent(in) :: minor_limits_gpt_atm
    # logical(wl),      dimension(:),     intent(in) :: minor_scales_with_density_atm
    # character(len=*), dimension(:),     intent(in) :: scaling_gas_atm
    # logical(wl),      dimension(:),     intent(in) :: scale_by_complement_atm
    # integer,          dimension(:),     intent(in) :: kminor_start_atm
    # real(FT),         dimension(:,:,:), allocatable, intent(out) :: kminor_atm_red
    # character(len=*), dimension(:),     allocatable, intent(out) :: minor_gases_atm_red
    # integer,          dimension(:,:),   allocatable, intent(out) :: minor_limits_gpt_atm_red
    # logical(wl),      dimension(:),     allocatable, intent(out) :: minor_scales_with_density_atm_red
    # character(len=*), dimension(:),     allocatable, intent(out) :: scaling_gas_atm_red
    # logical(wl),      dimension(:),     allocatable, intent(out) :: scale_by_complement_atm_red
    # integer,          dimension(:),     allocatable, intent(out) :: kminor_start_atm_red

    # # Local variables
    # integer :: i, j
    # integer :: idx_mnr, nm, tot_g, red_nm
    # integer :: icnt, n_elim, ng
    # logical, dimension(:), allocatable :: gas_is_present
    FT = eltype(kminor_atm)

    nm = length(minor_gases_atm)
    tot_g=0
    gas_is_present = Vector{Bool}(undef, nm)
    for i = 1:length(minor_gases_atm)
      idx_mnr = string_loc_in_array(minor_gases_atm[i], identifier_minor)
      gas_is_present[i] = string_in_array(gas_minor[idx_mnr],available_gases.gas_name)
      if gas_is_present[i]
        tot_g = tot_g + (minor_limits_gpt_atm[2,i]-minor_limits_gpt_atm[1,i]+1)
      end
    end
    red_nm = count(gas_is_present)

    if red_nm == nm
      kminor_atm_red = kminor_atm
      minor_gases_atm_red = minor_gases_atm
      minor_limits_gpt_atm_red = minor_limits_gpt_atm
      minor_scales_with_density_atm_red = minor_scales_with_density_atm
      scaling_gas_atm_red = scaling_gas_atm
      scale_by_complement_atm_red = scale_by_complement_atm
      kminor_start_atm_red = kminor_start_atm
    else
      minor_gases_atm_red= pack(minor_gases_atm, gas_is_present)
      minor_scales_with_density_atm_red = pack(minor_scales_with_density_atm,
        gas_is_present)
      scaling_gas_atm_red = pack(scaling_gas_atm,
        gas_is_present)
      scale_by_complement_atm_red = pack(scale_by_complement_atm,
        gas_is_present)
      kminor_start_atm_red = pack(kminor_start_atm,
        gas_is_present)

      minor_limits_gpt_atm_red = Array{Int}(undef, 2, red_nm)
      kminor_atm_red = Array{FT}(undef, tot_g, size(kminor_atm,2), size(kminor_atm,3))

      icnt = 0
      n_elim = 0
      for i = 1:nm
        ng = minor_limits_gpt_atm[2,i]-minor_limits_gpt_atm[1,i]+1
        if gas_is_present[i]
          icnt = icnt + 1
          minor_limits_gpt_atm_red[1:2,icnt] = minor_limits_gpt_atm[1:2,i]
          kminor_start_atm_red[icnt] = kminor_start_atm[i]-n_elim
          for j = 1:ng
            kminor_atm_red[kminor_start_atm_red[icnt]+j-1,:,:] = kminor_atm[kminor_start_atm[i]+j-1,:,:]
          end
        else
          n_elim = n_elim + ng
        end
      end
    end
    return kminor_atm_red,
           minor_gases_atm_red,
           minor_limits_gpt_atm_red,
           minor_scales_with_density_atm_red,
           scaling_gas_atm_red,
           scale_by_complement_atm_red,
           kminor_start_atm_red

  end

# ---------------------------------------------------------------------------------------
  # returns flavor index; -1 if not found
  function key_species_pair2flavor(flavor, key_species_pair)
    # integer :: key_species_pair2flavor
    # integer, dimension(:,:), intent(in) :: flavor
    # integer, dimension(2), intent(in) :: key_species_pair
    # integer :: iflav
    for iflav=1:size(flavor,2)
      if all(key_species_pair[:] == flavor[:,iflav])
        return iflav
      end
    end
    return -1
  end

  # ---------------------------------------------------------------------------------------
  #
  # create gpoint_flavor list
  #   a map pointing from each g-point to the corresponding entry in the "flavor list"
  #
  function create_gpoint_flavor(key_species, gpt2band, flavor)
    # integer, dimension(:,:,:), intent(in) :: key_species
    # integer, dimension(:), intent(in) :: gpt2band
    # integer, dimension(:,:), intent(in) :: flavor
    # integer, dimension(:,:), intent(out), allocatable :: gpoint_flavor
    # integer :: ngpt, igpt, iatm
    ngpt = length(gpt2band)
    gpoint_flavor = Array{Int}(undef, 2,ngpt)
    for igpt=1:ngpt
      for iatm=1:2
        gpoint_flavor[iatm,igpt] = key_species_pair2flavor(
          flavor,
          rewrite_key_species_pair(key_species[:,iatm,gpt2band[igpt]])
        )
      end
    end
    return gpoint_flavor
  end

 #--------------------------------------------------------------------------------------------------------------------
 #
 # Utility function to combine optical depths from gas absorption and Rayleigh scattering
 #   (and reorder them for convenience, while we're at it)
 #
 function combine_and_reorder!(tau, tau_rayleigh, has_rayleigh, optical_props::ty_optical_props_arry)
    # real(FT), dimension(:,:,:),   intent(in) :: tau
    # real(FT), dimension(:,:,:),   intent(in) :: tau_rayleigh
    # logical,                      intent(in) :: has_rayleigh
    # class(ty_optical_props_arry), intent(inout) :: optical_props

    # integer :: ncol, nlay, ngpt, nmom

    ncol = size(tau, 3)
    nlay = size(tau, 2)
    ngpt = size(tau, 1)
    #$acc enter data copyin(optical_props)
    if !has_rayleigh
      # index reorder (ngpt, nlay, ncol) -> (ncol,nlay,gpt)
      #$acc enter data copyin(tau)
      #$acc enter data create(optical_props%tau)
      reorder123x321!(tau, optical_props.tau)
        if optical_props isa ty_optical_props_2str
          #$acc enter data create(optical_props%ssa, optical_props%g)
          zero_array!(optical_props.ssa)
          zero_array!(optical_props.g  )
          #$acc exit data copyout(optical_props%ssa, optical_props%g)
        elseif optical_props isa ty_optical_props_nstr # We ought to be able to combine this with above
          nmom = size(optical_props.p, 1)
          #$acc enter data create(optical_props%ssa, optical_props%p)
          zero_array!(optical_props.ssa)
          zero_array!(optical_props.p  )
          #$acc exit data copyout(optical_props%ssa, optical_props%p)
        end
      #$acc exit data copyout(optical_props%tau)
      #$acc exit data delete(tau)
    else
      # combine optical depth and rayleigh scattering
      #$acc enter data copyin(tau, tau_rayleigh)
        if optical_props isa ty_optical_props_1scl
          # User is asking for absorption optical depth
          #$acc enter data create(optical_props%tau)
          reorder123x321!(tau, optical_props.tau)
          #$acc exit data copyout(optical_props%tau)
        elseif optical_props isa ty_optical_props_2str
          #$acc enter data create(optical_props%tau, optical_props%ssa, optical_props%g)
          optical_props.tau, optical_props.ssa, optical_props.g =
            combine_and_reorder_2str(ncol, nlay, ngpt,       tau, tau_rayleigh)
          #$acc exit data copyout(optical_props%tau, optical_props%ssa, optical_props%g)
        elseif optical_props isa ty_optical_props_nstr # We ought to be able to combine this with above
          nmom = size(optical_props.p, 1)
          #$acc enter data create(optical_props%tau, optical_props%ssa, optical_props%p)
          combine_and_reorder_nstr!(ncol, nlay, ngpt, nmom, tau, tau_rayleigh,
                                        optical_props.tau, optical_props.ssa, optical_props.p)
          #$acc exit data copyout(optical_props%tau, optical_props%ssa, optical_props%p)
        end
      #$acc exit data delete(tau, tau_rayleigh)
    end
    #$acc exit data copyout(optical_props)
  end

  #--------------------------------------------------------------------------------------------------------------------
  # Sizes of tables: pressure, temperate, eta (mixing fraction)
  #   Equivalent routines for the number of gases and flavors (get_ngas(), get_nflav()) are defined above because they're
  #   used in function defintions
  # Table kmajor has dimensions (ngpt, neta, npres, ntemp)
  #--------------------------------------------------------------------------------------------------------------------
  #
  # return extent of eta dimension
  #
  function get_neta(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # integer                          :: get_neta

    return size(this.kmajor,2)
  end
  # --------------------------------------------------------------------------------------
  #
  # return the number of pressures in reference profile
  #   absorption coefficient table is one bigger since a pressure is repeated in upper/lower atmos
  #
  function get_npres(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # integer                          :: get_npres

    return size(this.kmajor,3)-1
  end
  # --------------------------------------------------------------------------------------
  #
  # return the number of temperatures
  #
  function get_ntemp(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # integer                          :: get_ntemp

    return size(this.kmajor,4)
  end
  # --------------------------------------------------------------------------------------
  #
  # return the number of temperatures for Planck function
  #
  function get_nPlanckTemp(this::ty_gas_optics_rrtmgp)
    # class(ty_gas_optics_rrtmgp), intent(in) :: this
    # integer                          :: get_nPlanckTemp

    return size(this.totplnk,1) # dimensions are Planck-temperature, band
  end
  #--------------------------------------------------------------------------------------------------------------------
  # Generic procedures for checking sizes, limits
  #--------------------------------------------------------------------------------------------------------------------
  #
  # Extents
  #
  # --------------------------------------------------------------------------------------
  function check_extent(array, s, label)
    # real(FT), dimension(:          ), intent(in) :: array
    # integer,                          intent(in) :: n1
    # character(len=*),                 intent(in) :: label
    # character(len=128)                           :: check_extent_1d

    @assert all(size(array).==s)
  end
  # --------------------------------------------------------------------------------------
  #
  # Values
  #
  # --------------------------------------------------------------------------------------

  function check_range(val, minV, maxV, label)
    # real(FT), dimension(:),     intent(in) :: val
    # real(FT),                   intent(in) :: minV, maxV
    # character(len=*),           intent(in) :: label
    # character(len=128)                     :: check_range_1D

    s = ""
    if(any(val .< minV) || any(val .> maxV))
      s = trim(label) * " values out of range."
    end
    return s
  end
  #------------------------------------------------------------------------------------------
end
