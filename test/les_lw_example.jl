using Test
using RRTMGP.Device: array_type, array_device
using KernelAbstractions
using CUDA
using RRTMGP
using RRTMGP.GrayBCs
using RRTMGP.GrayAngularDiscretizations
using RRTMGP.GrayAtmosphericStates
using RRTMGP.GrayFluxes
using RRTMGP.GraySources
using RRTMGP.GrayAtmos
using RRTMGP.GrayRTESolver
using RRTMGP.GrayOptics
using RRTMGP.GrayUtils

using NCDatasets

using CLIMAParameters
import CLIMAParameters.Planet: cp_d, grav, R_d
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

#using Plots

DA = array_type()

"""
Example program to demonstrate the calculation of longwave radiative fluxes in a model gray atmosphere.
"""
function les_lw_example(
    ::Type{OPC},
    ::Type{FT},
    ::Type{I},
    ::Type{DA},
) where {OPC<:AbstractGrayOpticalProps,FT<:AbstractFloat,I<:Int,DA}

    nbnd, ngpt = 1, 1  # # of nbands/g-points (=1 for gray radiation)

    max_threads = 256
    # Reading LES data from NetCDF file
    ds =
        Dataset("test/HadGEM2-A_amip.2004-2008.07_site23_AtmosLESDefault_2020-10-07T12.46.14.887.nc")

    nlev = I(ds.dim["z"])
    nlay = nlev - 1
    ncol = I(ds.dim["time"])
    t_lev = DA{FT}(ds["temp"][:])
    p_lev = DA{FT}(ds["pres"][:])
    z_lev = DA{FT}(repeat(ds["z"][:], 1, ncol))

    close(ds)
    #-------------------------------------------
    lat = DA{FT}((FT(π) * FT(17) / FT(180)) * ones(FT, ncol)) # latitude
    sfc_emis = DA{FT}(undef, ncol)                               # surface emissivity
    sfc_emis .= FT(1.0)

    hr_lay = DA{FT}(undef, nlay, ncol)
    n_gauss_angles = I(1)                                  # only one supported at this point (only used for OneScalar LW)

    p0 = FT(maximum(p_lev[1, :])) # surface pressure (Pa)
    pe = FT(9000)                # TOA pressure (Pa)
    #-------------------------------------------

    as = GrayAtmosphericState(nlay, ncol, p_lev, t_lev, z_lev, lat, DA)
    optical_props = OPC(FT, ncol, nlay, DA)
    sf = source_func_longwave_gray_atmos(FT, ncol, nlay, ngpt, OPC, DA)
    bcs = GrayLwBCs(DA, sfc_emis)
    gray_flux = GrayFlux(ncol, nlay, nlev, FT, DA)
    ang_disc = AngularDiscretization(FT, n_gauss_angles, DA)


    gray_rrtmgp = GrayRRTMGP{
        FT,
        I,
        DA{FT,1},
        DA{FT,2},
        DA{FT,3},
        Bool,
        typeof(optical_props),
        typeof(sf),
        typeof(bcs),
    }(
        as,
        optical_props,
        sf,
        bcs,
        gray_flux,
        ang_disc,
    )

    # calling the long wave gray radiation solver
    gray_atmos_lw!(gray_rrtmgp, max_threads = max_threads)
    # computing heating rate
    compute_gray_heating_rate!(
        gray_rrtmgp.as,
        gray_rrtmgp.flux,
        hr_lay,
        param_set,
    )

end

if DA == CuArray
    @time les_lw_example(GrayOneScalar, Float64, Int, DA)
    @time les_lw_example(GrayTwoStream, Float64, Int, DA)
else
    @time les_lw_example(GrayOneScalar, Float64, Int, DA)
    @time les_lw_example(GrayTwoStream, Float64, Int, DA)
end
