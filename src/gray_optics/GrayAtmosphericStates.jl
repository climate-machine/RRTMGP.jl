module GrayAtmosphericStates
using KernelAbstractions
using CUDA
using ..Device: array_type, array_device
using DocStringExtensions

using GaussQuadrature
using CLIMAParameters
using CLIMAParameters.Planet: grav, R_d

export GrayAtmosphericState, setup_gray_as_pr_grid, setup_gray_as_alt_grid

"""
    GrayAtmosphericState{FT}

Atmospheric conditions, used to compute optical properties with the gray atmosphere approximation

# Fields
$(DocStringExtensions.FIELDS)
"""
struct GrayAtmosphericState{
    FT<:AbstractFloat,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    I<:Int,
}
    "Layer pressures `[Pa, mb]`; (`nlay,ncol`)"
    p_lay::FTA2D
    "Level pressures `[Pa, mb]`; (`nlay+1,ncol`)"
    p_lev::FTA2D
    "Layer temperatures `[K]`; (`nlay,ncol`)"
    t_lay::FTA2D
    "Level temperatures `[K]`; (`nlay+1,ncol`)"
    t_lev::FTA2D
    "Level Altitude `[m]`; (`nlay+1,ncol`)"
    z_lev::FTA2D
    "Surface temperatures `[K]`; (`ncol`)"
    t_sfc::FTA1D
    "lapse rate"
    α::FT
    "optical thickness parameter"
    d0::FTA1D
    "Number of layers."
    nlay::I
    "Number of columns."
    ncol::I
end

# This functions sets up a model temperature and pressure 
# distributions for a gray atmosphere based on a pressure grid
# see Schneider 2004, J. Atmos. Sci. (2004) 61 (12): 1317–1340.
# https://doi.org/10.1175/1520-0469(2004)061<1317:TTATTS>2.0.CO;2

function setup_gray_as_pr_grid(
    nlay::Int,
    lat::FTA1D,
    p0::FT,
    pe::FT,
    param_set::AbstractEarthParameterSet,
    ::Type{DA},
    step = "linear",
) where {FT<:AbstractFloat,FTA1D<:AbstractArray{FT,1},DA}
    nlev = Int(nlay + 1)
    ncol = length(lat)
    p_lay = DA{FT}(undef, nlay, ncol) # layer mean pressure
    p_lev = DA{FT}(undef, nlev, ncol) # level pressure
    t_lay = DA{FT}(undef, nlay, ncol) # layer mean temperature
    t_lev = DA{FT}(undef, nlev, ncol) # level temperature
    z_lev = DA{FT}(undef, nlev, ncol) # level altitude
    t_sfc = DA{FT}(undef, ncol)       # surface temperature
    efac = log(p0 / pe) / nlay       # multiplication factor for each pressure step
    d0 = DA{FT}(undef, ncol)       # optical depth (function of latitude)
    Δp = (p0 - pe) / nlay          # Δp for linear pressure distribution
    te = FT(300)                   # global mean surface temperature (K)
    tt = FT(200)                   # skin temp at top of atmosphere (K)
    Δt = FT(60)
    α = FT(3.5)                   # lapse rate of radiative equillibrium
    r_d = FT(R_d(param_set))
    grav_ = FT((grav(param_set)))
    #----launcing KA kernel
    thr_x = 256

    device = array_device(p_lay)
    workgroup = (thr_x)
    ndrange = (ncol)

    kernel = setup_gray_as_pr_grid_kernel!(device, workgroup)

    event = kernel(
        p_lev,
        p_lay,
        t_lev,
        t_lay,
        z_lev,
        t_sfc,
        lat,
        d0,
        efac,
        p0,
        pe,
        Δp,
        te,
        tt,
        Δt,
        α,
        r_d,
        grav_,
        Val(nlay),
        Val(ncol),
        ndrange = ndrange,
    )
    wait(event)
    #------------------------------------------------
    return GrayAtmosphericState{FT,DA{FT,1},DA{FT,2},Int}(
        p_lay,
        p_lev,
        t_lay,
        t_lev,
        z_lev,
        t_sfc,
        α,
        d0,
        nlay,
        ncol,
    )
end

# This functions sets up a model temperature and pressure 
# distributions for a gray atmosphere based on an altitude grid
# with internal GLL point distribution within each cell
# see Schneider 2004, J. Atmos. Sci. (2004) 61 (12): 1317–1340.
# https://doi.org/10.1175/1520-0469(2004)061<1317:TTATTS>2.0.CO;2

function setup_gray_as_alt_grid(
    lat::FTA1D,
    p0::FT,
    zend::FT,
    ncls::Int,
    poly_order::Int,
    param_set::AbstractEarthParameterSet,
    ::Type{DA},
) where {FT<:AbstractFloat,FTA1D<:AbstractArray{FT,1},DA}
    ncol = length(lat)
    nlev = Int(ncls + 1 + (poly_order - 1) * ncls)
    nlay = Int(nlev - 1)
    qm1 = poly_order + 1
    zst = FT(0)
    te = FT(300)                   # global mean surface temperature (K)
    tt = FT(200)                   # skin temp at top of atmosphere (K)
    Δt = FT(60)
    α = FT(3.5)           # lapse rate of radiative equillibrium

    p_lay = DA{FT}(undef, nlay, ncol)
    p_lev = DA{FT}(undef, nlev, ncol)
    t_lay = DA{FT}(undef, nlay, ncol)
    t_lev = DA{FT}(undef, nlev, ncol)
    z_lev = DA{FT}(undef, nlev, ncol)
    t_sfc = DA{FT}(undef, ncol)       # surface temperature
    d0 = DA{FT}(undef, ncol)       # optical depth (function of latitude)

    pts, _ = GaussQuadrature.legendre(FT, poly_order + 1, GaussQuadrature.both)
    pts = (pts .+ FT(1)) ./ FT(2)
    Δz_cell = zend / ncls

    for icol = 1:ncol
        #---------bot_at_1---------------------------------------
        z_lev[1, icol] = zst
        z_lev[nlev, icol] = zend

        for icls = 1:ncls
            for j = 1:poly_order-1
                z_lev[(icls - 1) * poly_order+1+j, icol] =
                    z_lev[(icls-1)*poly_order+1, icol] + pts[1+j] * Δz_cell
            end
            z_lev[icls*poly_order+1, icol] = zst + icls * Δz_cell
        end

        ts = te + Δt * (FT(1 / 3) - sin(lat[icol])^2) # surface temp at a given latitude (K)
        d0[icol] = FT((ts / tt)^4 - 1) # optical depth

        p_lev[1, icol] = p0
        t_lev[1, icol] = tt * (1 + d0[icol] * (p_lev[icol, 1] / p0)^α)^FT(0.25)

        for ilay = 1:nlay
            H = FT(R_d(param_set)) * t_lev[ilay, icol] / FT(grav(param_set))
            Δz_lay = abs(z_lev[ilay+1, icol] - z_lev[ilay, icol])
            p_lev[ilay+1, icol] = p_lev[ilay, icol] * exp(-Δz_lay / H)
            t_lev[ilay+1, icol] =
                tt * (1 + d0[icol] * (p_lev[ilay+1, icol] / p0)^α)^FT(0.25)

            t_lay[ilay, icol] =
                FT(0.5) * (t_lev[ilay, icol] + t_lev[ilay+1, icol])
            p_lay[ilay, icol] =
                FT(0.5) * (p_lev[ilay, icol] + p_lev[ilay+1, icol])
        end
        #--------------------------------------------------------
        t_sfc[icol] = t_lev[1, icol]
    end
    return GrayAtmosphericState{FT,DA{FT,1},DA{FT,2},Int}(
        p_lay,
        p_lev,
        t_lay,
        t_lev,
        z_lev,
        t_sfc,
        α,
        d0,
        nlay,
        ncol,
    )
end

#-------------------------------------------------------------------------
@kernel function setup_gray_as_pr_grid_kernel!(
    p_lev::FTA2D,
    p_lay::FTA2D,
    t_lev::FTA2D,
    t_lay::FTA2D,
    z_lev::FTA2D,
    t_sfc::FTA1D,
    lat::FTA1D,
    d0::FTA1D,
    efac::FT,
    p0::FT,
    pe::FT,
    Δp::FT,
    te::FT,
    tt::FT,
    Δt::FT,
    α::FT,
    r_d::FT,
    grav_::FT,
    ::Val{nlay},
    ::Val{ncol},
) where {
    FT<:AbstractFloat,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    nlay,
    ncol,
}
    gcol = @index(Global, Linear)  # global col & lay ids

    ts = te + Δt * (FT(1) / FT(3) - sin(lat[gcol])^2) # surface temp at a given latitude (K)
    d0[gcol] = FT((ts / tt)^FT(4) - FT(1)) # optical depth
    nlev = nlay + 1

    #---bot_at_1------------------------------
    p_lev[1, gcol] = p0
    t_lev[1, gcol] = tt * (FT(1) + d0[gcol] * (p_lev[1, gcol] / p0)^α)^FT(0.25)
    z_lev[1, gcol] = FT(0)

    for ilay = 1:nlay
        #                if step == "linear"
        p_lev[ilay+1, gcol] = p_lev[ilay, gcol] - Δp
        #                else
        #                    p_lev[ilay+1, gcol] = p_lev[ilay, gcol] * exp(-efac)
        #                end
        p_lay[ilay, gcol] = (p_lev[ilay, gcol] + p_lev[ilay+1, gcol]) * FT(0.5)

        t_lev[ilay+1, gcol] =
            tt * (FT(1) + d0[gcol] * (p_lev[ilay+1, gcol] / p0)^α)^FT(0.25)
        t_lay[ilay, gcol] =
            tt * (FT(1) + d0[gcol] * (p_lay[ilay, gcol] / p0)^α)^FT(0.25)

        H = r_d * t_lay[ilay, gcol] / grav_
        Δz_lay = H * log(p_lev[ilay, gcol] / p_lev[ilay+1, gcol])
        z_lev[ilay+1, gcol] = Δz_lay + z_lev[ilay, gcol]
    end
    t_sfc[gcol] = t_lev[1, gcol]

    #---------------------------------

end
#-------------------------------------------------------------------------
# Gray Atmospheric State setup for LES data
function GrayAtmosphericState(
    nlay::Int,
    ncol::Int,
    p_lev::FTA2D,
    t_lev::FTA2D,
    z_lev::FTA2D,
    lat::FTA1D,
    ::Type{DA},
) where {
    I<:Int,
    FT<:AbstractFloat,
    FTA1D<:AbstractArray{FT,1},
    FTA2D<:AbstractArray{FT,2},
    DA,
}

    te = FT(300)                   # global mean surface temperature (K)
    tt = FT(200)                   # skin temp at top of atmosphere (K)
    Δt = FT(60)
    α = FT(3.5)                   # lapse rate of radiative equillibrium

    p_lay = DA{FT}(undef, nlay, ncol)
    t_lay = DA{FT}(undef, nlay, ncol)
    t_sfc = DA{FT}(undef, ncol)       # surface temperature
    d0 = DA{FT}(undef, ncol)       # optical depth (function of latitude)

    #------------------
    max_threads = 256
    thr_y = min(32, ncol)
    thr_x = min(Int(floor(FT(max_threads / thr_y))), nlay)

    device = array_device(p_lev)
    comp_stream = Event(device)
    workgroup = (thr_x, thr_y)

    comp_stream = compute_play_tlay_tsfc_kernel!(device, workgroup)(
        p_lev,
        p_lay,
        t_lev,
        t_lay,
        t_sfc,
        d0,
        lat,
        te,
        tt,
        Δt,
        Val(nlay),
        Val(ncol),
        ndrange = (nlay, ncol),
        dependencies = (comp_stream,),
    )
    wait(comp_stream)
    #------------------

    return GrayAtmosphericState{FT,DA{FT,1},DA{FT,2},Int}(
        p_lay,
        p_lev,
        t_lay,
        t_lev,
        z_lev,
        t_sfc,
        α,
        d0,
        nlay,
        ncol,
    )
end
#-------------------------------------------------------------------------
@kernel function compute_play_tlay_tsfc_kernel!(
    p_lev::FTA2D,
    p_lay::FTA2D,
    t_lev::FTA2D,
    t_lay::FTA2D,
    t_sfc::FTA1D,
    d0::FTA1D,
    lat::FTA1D,
    te::FT,
    tt::FT,
    Δt::FT,
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

    p_lay[glay, gcol] = (p_lev[glay, gcol] + p_lev[glay+1, gcol]) * FT(0.5)
    t_lay[glay, gcol] = (t_lev[glay, gcol] + t_lev[glay+1, gcol]) * FT(0.5)

    if glay == 1
        ts = te + Δt * (FT(1) / FT(3) - sin(lat[gcol])^2) # surface temp at a given latitude (K)
        d0[gcol] = FT((ts / tt)^FT(4) - FT(1)) # optical depth
        t_sfc[gcol] = t_lev[1, gcol]
    end
end
#-------------------------------------------------------------------------
end
