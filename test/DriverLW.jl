using Test
using JRRTMGP
using NCDatasets
using BenchmarkTools
using JRRTMGP.mo_optical_props
using JRRTMGP.mo_rte_solver_kernels
using JRRTMGP.fortran_intrinsics
using JRRTMGP.mo_util_array
using JRRTMGP.mo_gas_optics_rrtmgp
using JRRTMGP.mo_gas_concentrations
using JRRTMGP.mo_rte_lw
using JRRTMGP.mo_fluxes
using JRRTMGP.mo_load_coefficients
using JRRTMGP.mo_rfmip_io
using JRRTMGP.mo_source_functions

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
# Example program to demonstrate the calculation of shortwave radiative fluxes in clear, aerosol-free skies.
#   The example files come from the Radiative Forcing MIP (https://www.earthsystemcog.org/projects/rfmip/)
#   The large problem (1800 profiles) is divided into blocks
#
# Program is invoked as rrtmgp_rfmip_sw [block_size input_file  coefficient_file upflux_file downflux_file]
#   All arguments are optional but need to be specified in order.
#
# -------------------------------------------------------------------------------------------------
#
# Error checking: Procedures in rte+rrtmgp return strings which are empty if no errors occured
#   Check the incoming string, print it out and stop execution if non-empty
#

# -------------------------------------------------------------------------------------------------
#
# Main program
#
# -------------------------------------------------------------------------------------------------
function run_driver(nblocks_iterations=nothing)
  # --------------------------------------------------
  #
  # Modules for working with rte and rrtmgp
  #
  # Working precision for real variables
  #
  # use mo_rte_kind,           only: FT
  #
  # Array utilities
  #
  # use mo_util_array,         only: zero_array
  #
  # Optical properties of the atmosphere as array of values
  #   In the longwave we include only absorption optical depth (_1scl)
  #   Shortwave calculations use optical depth, single-scattering albedo, asymmetry parameter (_2str)
  #
  # use mo_optical_props,      only: ty_optical_props_2str
  #
  # Gas optics: maps physical state of the atmosphere to optical properties
  #
  # use mo_gas_optics_rrtmgp,  only: ty_gas_optics_rrtmgp
  #
  # Gas optics uses a derived type to represent gas concentrations compactly
  #
  # use mo_gas_concentrations, only: ty_gas_concs
  #
  # RTE shortwave driver
  #
  # use mo_rte_sw,             only: rte_sw
  #
  # RTE driver uses a derived type to reduce spectral fluxes to whatever the user wants
  #   Here we're just reporting broadband fluxes
  #
  # use mo_fluxes,             only: ty_fluxes_broadband
  # --------------------------------------------------
  #
  # modules for reading and writing files
  #
  # RRTMGP's gas optics class needs to be initialized with data read from a netCDF files
  #
  # use mo_load_coefficients,  only: load_and_init
  # use mo_rfmip_io,           only: read_size, read_and_block_pt, read_and_block_gases_ty, unblock_and_write, read_and_block_sw_bc, determine_gas_names

#ifdef USE_TIMING
  #
  # Timing library
  #
  # use gptl,                  only: gptlstart, gptlstop, gptlinitialize, gptlpr, gptlfinalize, gptlsetoption,
  #                                  gptlpercent, gptloverhead
#endif
  # implicit none
  # --------------------------------------------------
  #
  # Local variables
  #


  # character(len=132) :: flxdn_file, flxup_file
  # integer            :: nargs, ncol, nlay, nbnd, ngpt, nexp, nblocks, block_size, forcing_index
  # logical            :: top_at_1
  # integer            :: b, icol, ibnd, igpt
  # character(len=4)   :: block_size_char, forcing_index_char = '1'

  # character(len=32 ), dimension(:),   allocatable :: kdist_gas_names, rfmip_gas_games
  # real(FT), dimension(:,:,:),         allocatable :: p_lay, p_lev, t_lay, t_lev # block_size, nlay, nblocks
  # real(FT), dimension(:,:,:), target, allocatable :: flux_up, flux_dn
  # real(FT), dimension(:,:  ),         allocatable :: surface_albedo, total_solar_irradiance, solar_zenith_angle
  #                                                    # block_size, nblocks
  # real(FT), dimension(:,:  ),         allocatable :: sfc_alb_spec # nbnd, block_size; spectrally-resolved surface albedo
  #
  # Classes used by rte+rrtmgp
  #
  # type(ty_gas_optics_rrtmgp)                     :: k_dist
  # type(ty_optical_props_2str)                    :: optical_props
  # type(ty_fluxes_broadband)                      :: fluxes

  # real(FT), dimension(:,:), allocatable          :: toa_flux # block_size, ngpt
  # real(FT), dimension(:  ), allocatable          :: def_tsi, mu0    # block_size
  # logical , dimension(:,:), allocatable          :: usecol # block_size, nblocks
  #
  # ty_gas_concentration holds multiple columns; we make an array of these objects to
  #   leverage what we know about the input file
  #
  # type(ty_gas_concs), dimension(:), allocatable  :: gas_conc_array
  FT = Float64
  I  = Integer
  deg_to_rad = acos(-FT(1))/FT(180)
  n_quad_angles = Integer.(1)

  # -------------------------------------------------------------------------------------------------
  #
  # Code starts
  #   all arguments are optional
  #
  pwd()
  clear_sky_dir = joinpath("..", "src", "examples","rfmip-clear-sky")
  data_dir = joinpath("..", "src", "rrtmgp","data")
  rfmip_file = joinpath(clear_sky_dir, "multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc")
#  kdist_file = joinpath(data_dir, "rrtmgp-data-sw-g224-2018-12-04.nc")
  kdist_file = joinpath(data_dir, "rrtmgp-data-lw-g256-2018-12-04.nc")
  ds = Dataset(rfmip_file, "r") # reading the NetCDF file in read only mode
  ds_k_dist = Dataset(kdist_file, "r") # reading the NetCDF file in read only mode

  ncol, nlay, nexp = read_size(ds)

  forcing_index = 1
  block_size = 8
  # error("Done")

  #
  # How big is the problem? Does it fit into blocks of the size we've specified?
  #
  if mod(ncol*nexp, block_size) ≠ 0
    error("rrtmgp_rfmip_sw: number of columns does not fit evenly into blocks.")
  end
  nblocks = Int((ncol*nexp)/block_size)
  println("Doing $(nblocks) blocks of size $(block_size)")

  # TODO: Fix readme
  # read(forcing_index_char, "(i4)") forcing_index
  if (forcing_index < 1 || forcing_index > 3)
    error("Forcing index is invalid (must be 1,2 or 3)")
  end
  flxdn_file = "rld_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p" * string(forcing_index) * "_gn.nc"
  flxup_file = "rlu_Efx_RTE-RRTMGP-181204_rad-irf_r1i1p" * string(forcing_index) * "_gn.nc"

  #
  # Identify the set of gases used in the calculation based on the forcing index
  #   A gas might have a different name in the k-distribution than in the files
  #   provided by RFMIP (e.g. 'co2' and 'carbon_dioxide')
  #
  # names_in_kdist, names_in_file = determine_gas_names(ds_k_dist, forcing_index)
  kdist_gas_names, rfmip_gas_games = determine_gas_names(ds_k_dist, forcing_index)
  print("Calculation uses RFMIP gases: ")
  @show rfmip_gas_games

  # --------------------------------------------------
  #
  # Prepare data for use in rte+rrtmgp
  #
  #
  # Allocation on assignment within reading routines
  #
  p_lay, p_lev, t_lay, t_lev = read_and_block_pt(ds, block_size)
  #
  # Are the arrays ordered in the vertical with 1 at the top or the bottom of the domain?
  #

  # test_data(p_lay, "p_lay")
  # test_data(p_lev, "p_lev")
  # test_data(t_lay, "t_lay")
  # test_data(t_lev, "t_lev")

  top_at_1 = p_lay[1, 1, 1] < p_lay[1, nlay, 1]

  #
  # Read the gas concentrations and surface properties
  #
  gas_conc_array = read_and_block_gases_ty(ds, block_size, kdist_gas_names, rfmip_gas_games)
println("before read_and_block_lw_bc")
  sfc_emis, sfc_t = read_and_block_lw_bc(ds, block_size)
println("after read_and_block_lw_bc")

  # test_data(surface_albedo, "surface_albedo")
  # test_data(total_solar_irradiance, "total_solar_irradiance")
  # test_data(solar_zenith_angle, "solar_zenith_angle")

  #
  # Read k-distribution information. load_and_init() reads data from netCDF and calls
  #   k_dist%init(); users might want to use their own reading methods
  #
  k_dist = load_and_init(ds_k_dist, gas_conc_array[1])
println("After load_and_init" )
println("source_is_internal = ", source_is_internal(k_dist) )
println("size(kdist.totplnk) = ", size(k_dist.totplnk) )
  if !source_is_internal(k_dist) 
    error("rrtmgp_rfmip_lw: k-distribution file is not LW")
  end
println("After source_is_internal" )

  # test_data(k_dist.optical_props.band_lims_wvn, "k_dist_op_band_lims_wvn")
  # test_data(k_dist.optical_props.gpt2band, "k_dist_op_gpt2band")
  # test_data(k_dist.optical_props.band2gpt, "k_dist_op_band2gpt")

  nbnd = get_nband(k_dist.optical_props)
println("After get_nband" )
  ngpt = get_ngpt(k_dist.optical_props)
println("After get_ngpt" )

#  toa_flux = Array{FT}(undef, block_size, get_ngpt(k_dist.optical_props))
#  def_tsi = Array{FT}(undef, block_size)
#  usecol = Array{Bool}(undef, block_size, nblocks)
  #
  # RRTMGP won't run with pressure less than its minimum. The top level in the RFMIP file
  #   is set to 10^-3 Pa. Here we pretend the layer is just a bit less deep.
  #   This introduces an error but shows input sanitizing.
  #
  if top_at_1
    p_lev[:,1,:] .= get_press_min(k_dist) + eps(FT)
  else
    p_lev[:,nlay+1,:] .= get_press_min(k_dist) + eps(FT)
  end
println("After p_lev adjustment" )
  # test_data(p_lev, "p_lev_2")

  #
  # RTE will fail if passed solar zenith angles greater than 90 degree. We replace any with
  #   nighttime columns with a default solar zenith angle. We'll mask these out later, of
  #   course, but this gives us more work and so a better measure of timing.
  #
  # test_data(usecol, "usecol")

  #
  # Allocate space for output fluxes (accessed via pointers in ty_fluxes_broadband),
  #   gas optical properties, and source functions. The %alloc() routines carry along
  #   the spectral discretization from the k-distribution.
  #
  flux_up = Array{FT}(undef, block_size, nlay+1, nblocks)
  flux_dn = Array{FT}(undef, block_size, nlay+1, nblocks)
#  mu0 = Array{FT}(undef, block_size)
  sfc_emis_spec = Array{FT}(undef, nbnd,block_size)

println("After initializing flux_up, flux_dn, and sfc_emis_spec" )
  optical_props = ty_optical_props_1scl(FT, Int)
println("initialized optical_props")
  copy_and_alloc!(optical_props, block_size, nlay, k_dist.optical_props)
println("copied optical_props")
  source = ty_source_func_lw(block_size,nlay,k_dist.optical_props)
println("initialized source function")
#----curr - SK

  # test_data(optical_props.band_lims_wvn, "DriverSW_op_band_lims_wvn")
  # test_data(optical_props.gpt2band, "DriverSW_op_gpt2band")
  # test_data(optical_props.band2gpt, "DriverSW_op_band2gpt")

  #$acc enter data create(optical_props, optical_props%tau, optical_props%ssa, optical_props%g)
  #$acc enter data create (toa_flux, def_tsi)
  #$acc enter data create (sfc_alb_spec, mu0)
  # --------------------------------------------------
#ifdef USE_TIMING
  #
  # Initialize timers
  #
  # ret = gptlsetoption (gptlpercent, 1)        # Turn on "% of" print
  # ret = gptlsetoption (gptloverhead, 0)       # Turn off overhead estimate
  # ret =  gptlinitialize()
#endif
  #
  # Loop over blocks
  #
  fluxes = ty_fluxes_broadband(FT)

  for b = 1:1#nblocks_iterations
    fluxes.flux_up = flux_up[:,:,b]
    fluxes.flux_dn = flux_dn[:,:,b] 
println("before gas_optics")
println(" size(p_lay[:,:,b] = ", size(p_lay[:,:,b]), eltype(p_lay[:,:,b]))
println(" size(p_lev[:,:,b] = ", size(p_lev[:,:,b]), eltype(p_lev[:,:,b]))
println(" size(t_lay[:,:,b] = ", size(t_lay[:,:,b]), eltype(t_lay[:,:,b]))
println(" size(sfc_t[:,b]   = ", size(sfc_t[:,b]), eltype(sfc_t[:,b]))

#@show sfc_t[:,b]
    gas_optics!(k_dist,
                p_lay[:,:,b],
                p_lev[:,:,b],
                t_lay[:,:,b],
                sfc_t[:,  b],
                gas_conc_array[b],
                optical_props,
                source,
                nothing,
                t_lev[:,:,b]) 
println("after gas_optics")

    rte_lw!(optical_props,top_at_1,source,sfc_emis_spec,fluxes,nothing,n_quad_angles)

#rte_sw!(optical_props, top_at_1, mu0, toa_flux, sfc_alb_spec, sfc_alb_spec, fluxes)

  end

#  nblocks_iterations==nothing && (nblocks_iterations = nblocks)
#  for b = 1:nblocks_iterations
#    @show b/nblocks
#    fluxes.flux_up = flux_up[:,:,b]
#    fluxes.flux_dn = flux_dn[:,:,b]
    #
    # Compute the optical properties of the atmosphere and the Planck source functions
    #    from pressures, temperatures, and gas concentrations...
    #
#ifdef USE_TIMING
    # ret =  gptlstart('gas_optics (SW)')
#endif
#    gas_optics!(k_dist,
#                p_lay[:,:,b],
#                p_lev[:,:,b],
#                t_lay[:,:,b],
#                gas_conc_array[b],
#                optical_props,
#                toa_flux)
    # test_data(optical_props.tau, "DriverSW_op_tau")
#ifdef USE_TIMING
    # ret =  gptlstop('gas_optics (SW)')
#endif
    # Boundary conditions
    #   (This is partly to show how to keep work on GPUs using OpenACC in a host application)
    # What's the total solar irradiance assumed by RRTMGP?
    #
#ifdef _OPENACC
#    zero_array!(def_tsi)
    #$acc parallel loop collapse(2) copy(def_tsi) copyin(toa_flux)
    # for igpt = 1:ngpt
    #   for icol = 1:block_size
    #     #$acc atomic update
    #     def_tsi[icol] = def_tsi[icol] + toa_flux[icol, igpt]
    #   end
    # end
#else
    #
    # More compactly...
    #
#    def_tsi[1:block_size] = sum(toa_flux, dims=2)
    # test_data(def_tsi, "def_tsi")
#endif
    #
    # Normalize incoming solar flux to match RFMIP specification
    #
    #$acc parallel loop collapse(2) copyin(total_solar_irradiance, def_tsi) copy(toa_flux)
#    for igpt = 1:ngpt
#      for icol = 1:block_size
#        toa_flux[icol,igpt] = toa_flux[icol,igpt] * total_solar_irradiance[icol,b]/def_tsi[icol]
#      end
#    end
    # test_data(toa_flux, "toa_flux")
    #
    # Expand the spectrally-constant surface albedo to a per-band albedo for each column
    #
    #$acc parallel loop collapse(2) copyin(surface_albedo)
#    for icol = 1:block_size
#      for ibnd = 1:nbnd
#        sfc_alb_spec[ibnd,icol] = surface_albedo[icol,b]
#      end
#    end
    # test_data(sfc_alb_spec, "sfc_alb_spec")
    #
    # Cosine of the solar zenith angle
    #
    #$acc parallel loop copyin(solar_zenith_angle, usecol)
#    for icol = 1:block_size
#      mu0[icol] = fmerge(cos(solar_zenith_angle[icol,b]*deg_to_rad), FT(1), usecol[icol,b])
#    end
    # test_data(mu0, "mu0")

    #
    # ... and compute the spectrally-resolved fluxes, providing reduced values
    #    via ty_fluxes_broadband
    #
#ifdef USE_TIMING
    # ret =  gptlstart('rte_sw')
#endif

#    rte_sw!(optical_props, top_at_1, mu0, toa_flux, sfc_alb_spec, sfc_alb_spec, fluxes)


#ifdef USE_TIMING
    # ret =  gptlstop('rte_sw')
#endif
    #
    # Zero out fluxes for which the original solar zenith angle is > 90 degrees.
    #
#    for icol = 1:block_size
#      if !usecol[icol,b]
#        flux_up[icol,:,b] .= FT(0)
#        flux_dn[icol,:,b] .= FT(0)
#      end
#    end
#  end
  # test_data(fluxes.flux_up, "fluxes_flux_up")
  # test_data(fluxes.flux_dn, "fluxes_flux_dn")

  #
  # End timers
  #

  #$acc exit data delete(optical_props%tau, optical_props%ssa, optical_props%g, optical_props)
  #$acc exit data delete(sfc_alb_spec, mu0)
  #$acc exit data delete(toa_flux, def_tsi)
  # --------------------------------------------------
  # unblock_and_write!(trim(flxup_file), "rsu", flux_up)
  # unblock_and_write!(trim(flxdn_file), "rsd", flux_dn)

end


@testset "Longwave driver" begin
  run_driver(1)
#  run_driver()
end
