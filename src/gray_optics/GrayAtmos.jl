module GrayAtmos
using KernelAbstractions
using CUDA
using ..Device: array_type, array_device
using ..GrayAtmosphericStates
using DocStringExtensions
import GaussQuadrature
using ..GraySources
using ..GrayFluxes
using ..GrayAngularDiscretizations
using ..GrayOptics
using ..GrayBCs

using CLIMAParameters
using CLIMAParameters.Planet: grav, R_d, cp_d

export gas_optics_gray_atmos!,
    compute_gas_τs_gray_atmos!,
    update_profile_lw_kernel!,
    GrayRRTMGP,
    compute_gray_heating_rate!

function gas_optics_gray_atmos!(
    as::GrayAtmosphericState{FT,FTA1D,FTA2D,I},
    optical_props::AbstractGrayOpticalProps{FT,FTA2D},
    source::AbstractGraySource{FT},
) where {
    FT<:AbstractFloat,
    I<:Int,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    FTA3D<:AbstractArray{FT,3},
    B<:Bool,
}
    compute_gas_τs_gray_atmos!(as, optical_props)
    compute_Planck_source!(source, as)

    if typeof(optical_props) == GrayTwoStream{FT,FTA2D}
        optical_props.ssa .= FT(0)
        optical_props.g .= FT(0)
    end
    return nothing
end

function compute_gas_τs_gray_atmos!(
    as::GrayAtmosphericState{FT,FTA1D,FTA2D,I},
    optical_props::AbstractGrayOpticalProps{FT,FTA2D},
) where {
    FT<:AbstractFloat,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    I<:Int,
    B<:Bool,
}
    ncol = as.ncol
    nlay = as.nlay
    p_lay = as.p_lay
    p_lev = as.p_lev
    α = as.α
    d0 = as.d0
    τ = optical_props.τ
    #----Launcing KA Kernel---------------------------
    max_threads = 256
    thr_y = min(32, ncol)
    thr_x = min(Int(floor(FT(max_threads / thr_y))), nlay)

    device = array_device(τ)
    workgroup = (thr_x, thr_y)
    ndrange = (nlay, ncol)
    #----------------------------------
    kernel = compute_gas_τs_gray_atmos_kernel!(device, workgroup)
    event =
        kernel(p_lay, p_lev, d0, α, τ, Val(nlay), Val(ncol), ndrange = ndrange)
    wait(event)
    #----------------------------------
    return nothing
end

# This functions calculates the optical thickness based on pressure 
# and lapse rate for a gray atmosphere. 
# See Schneider 2004, J. Atmos. Sci. (2004) 61 (12): 1317–1340.
# https://doi.org/10.1175/1520-0469(2004)061<1317:TTATTS>2.0.CO;2

@kernel function compute_gas_τs_gray_atmos_kernel!(
    p_lay::FTA2D,
    p_lev::FTA2D,
    d0::FTA1D,
    α::FT,
    τ::FTA2D,
    ::Val{nlay},
    ::Val{ncol},
) where {
    FT<:AbstractFloat,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    nlay,
    ncol,
}
    glay, gcol = @index(Global, NTuple)  # global col & lay ids

    p0 = p_lev[1, gcol]

    τ[glay, gcol] = abs(
        (α * d0[gcol] * (p_lay[glay, gcol] ./ p0) .^ α ./ p_lay[glay, gcol]) * (p_lev[glay+1, gcol] - p_lev[glay, gcol]),
    )

    @synchronize
end

function compute_Planck_source!(
    source::AbstractGraySource{FT},
    as::GrayAtmosphericState{FT,FTA1D,FTA2D,I},
) where {
    FT<:AbstractFloat,
    I<:Int,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    FTA3D<:AbstractArray{FT,3},
    B<:Bool,
}
    ncol = as.ncol
    nlay = as.nlay
    t_lay = as.t_lay
    t_lev = as.t_lev
    t_sfc = as.t_sfc
    sfc_source = source.sfc_source
    lay_source = source.lay_source
    lev_source_inc = source.lev_source_inc
    lev_source_dec = source.lev_source_dec
    sbc = FT(Stefan())
    #----Launcing KA Kernel---------------------------
    max_threads = 256
    thr_y = min(32, ncol)
    thr_x = min(Int(floor(FT(max_threads / thr_y))), nlay)

    device = array_device(lay_source)
    workgroup = (thr_x, thr_y)
    ndrange = (nlay, ncol)
    #----------------------------------
    kernel = compute_Planck_source_kernel!(device, workgroup)
    event = kernel(
        t_sfc,
        t_lay,
        t_lev,
        sfc_source,
        lay_source,
        lev_source_inc,
        lev_source_dec,
        sbc,
        Val(nlay),
        Val(ncol),
        ndrange = ndrange,
    )
    wait(event)
    #----------------------------------
    return nothing
end

@kernel function compute_Planck_source_kernel!(
    t_sfc::FTA1D,
    t_lay::FTA2D,
    t_lev::FTA2D,
    sfc_source::FTA2D,
    lay_source::FTA3D,
    lev_source_inc::FTA3D,
    lev_source_dec::FTA3D,
    sbc::FT,
    ::Val{nlay},
    ::Val{ncol},
) where {
    FT<:AbstractFloat,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    FTA3D<:AbstractArray{FT,3},
    nlay,
    ncol,
}
    glay, gcol = @index(Global, NTuple)  # global col & lay ids

    sfc_source[gcol, 1] = sbc * t_sfc[gcol]^FT(4) / FT(π)   # computing sfc_source
    lay_source[glay, gcol, 1] = sbc * t_lay[glay, gcol]^FT(4) / FT(π)   # computing lay_source
    lev_source_inc[glay, gcol, 1] = sbc * t_lev[glay+1, gcol]^FT(4) / FT(π)
    lev_source_dec[glay, gcol, 1] = sbc * t_lev[glay, gcol]^FT(4) / FT(π)

    @synchronize
end

struct GrayRRTMGP{
    FT<:AbstractFloat,
    I<:Int,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    FTA3D<:AbstractArray{FT,3},
    B<:Bool,
    OP<:AbstractGrayOpticalProps{FT,FTA2D},
    GS<:AbstractGraySource{FT},
    BC<:AbstractGrayBCs{FT,FTA1D},
}
    as::GrayAtmosphericState{FT,FTA1D,FTA2D,I}    # atmoshperic state
    op::OP                                        # optical properties
    src::GS                                       # source functions
    bcs::BC                                       # boundary conditions
    flux::GrayFlux{FT,FTA2D}                      # fluxes
    angle_disc::AngularDiscretization{FT,FTA1D,I} # angular discretization
end

function GrayRRTMGP(
    nlay::Int,
    lat::FTA1D,
    p0::FT,
    pe::FT,
    optical_props_constructor,
    sfc_emis::Array{FT,1},
    flux::GrayFlux{FT,FTA2D},
    n_gauss_angles::Int,
    param_set,
    stype,
    ::Type{DA},
) where {
    FT<:AbstractFloat,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    DA,
}
    ncol, ngpt, nbnd = length(lat), 1, 1
    if stype == "lw"
        as = setup_gray_as_pr_grid(nlay, lat, p0, pe, param_set, DA)
        optical_props = optical_props_constructor(FT, ncol, nlay, DA)
        sf = source_func_longwave_gray_atmos(
            FT,
            ncol,
            nlay,
            ngpt,
            optical_props_constructor,
            DA,
        )
        bc = GrayLwBCs(DA, sfc_emis)
        ang_disc = AngularDiscretization(FT, n_gauss_angles, DA)
        return GrayRRTMGP{
            FT,
            Int,
            DA{FT,1},
            DA{FT,2},
            DA{FT,3},
            Bool,
            typeof(optical_props),
            typeof(sf),
            typeof(bc),
        }(
            as,
            optical_props,
            sf,
            bc,
            flux,
            ang_disc,
        )
    else
        error("Shortwave solver not yet supported")
    end
end

function GrayRRTMGP(
    lat::FTA1D,
    p0::FT,
    zend::FT,
    ncls::Int,
    poly_order::Int,
    optical_props_constructor,
    sfc_emis::Array{FT,1},
    flux::GrayFlux{FT,FTA2D},
    n_gauss_angles::Int,
    param_set,
    stype,
    ::Type{DA},
) where {
    FT<:AbstractFloat,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    DA,
}
    nlev = Int(ncls + 1 + (poly_order - 1) * ncls)
    nlay = Int(nlev - 1)
    ncol, ngpt, nbnd = length(lat), 1, 1
    if stype == "lw"
        as = setup_gray_as_alt_grid(
            lat,
            p0,
            zend,
            ncls,
            poly_order,
            param_set,
            DA,
        )
        optical_props = optical_props_constructor(FT, ncol, nlay, DA)
        sf = source_func_longwave_gray_atmos(
            FT,
            ncol,
            nlay,
            ngpt,
            optical_props_constructor,
            DA,
        )
        bc = GrayLwBCs(DA, sfc_emis)
        ang_disc = AngularDiscretization(FT, n_gauss_angles, DA)
        return GrayRRTMGP{
            FT,
            Int,
            DA{FT,1},
            DA{FT,2},
            DA{FT,3},
            Bool,
            typeof(optical_props),
            typeof(sf),
            typeof(bc),
        }(
            as,
            optical_props,
            sf,
            bc,
            flux,
            ang_disc,
        )
    else
        error("Shortwave solver not yet supported")
    end
end


end
