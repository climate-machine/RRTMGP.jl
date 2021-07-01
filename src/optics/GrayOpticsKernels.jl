
# This functions calculates the optical thickness based on pressure 
# and lapse rate for a gray atmosphere. 
# See Schneider 2004, J. Atmos. Sci. (2004) 61 (12): 1317–1340.
# https://doi.org/10.1175/1520-0469(2004)061<1317:TTATTS>2.0.CO;2

function compute_optical_props_kernel!(
    op::AbstractOpticalProps{FT},
    as::GrayAtmosphericState{FT},
    glaycol,
    source::AbstractSourceLW{FT},
) where {FT<:AbstractFloat}

    compute_optical_props_kernel!(op, as, glaycol)     # computing optical thickness
    compute_sources_gray_kernel!(source, as, glaycol) # computing Planck sources
end

function compute_optical_props_kernel!(
    op::AbstractOpticalProps{FT},
    as::GrayAtmosphericState{FT},
    glaycol,
) where {FT<:AbstractFloat}
    # setting references
    glay, gcol = glaycol
    @unpack p_lay, p_lev, d0, α = as
    @inbounds p0 = p_lev[1, gcol]

    @inbounds op.τ[glaycol...] = abs(
        (α * d0[gcol] * (p_lay[glaycol...] ./ p0) .^ α ./ p_lay[glaycol...]) * (p_lev[glay+1, gcol] - p_lev[glaycol...]),
    )

    if op isa TwoStream
        op.ssa[glaycol...] = FT(0)
        op.g[glaycol...] = FT(0)
    end
end

function compute_sources_gray_kernel!(
    source::AbstractSourceLW{FT},
    as::GrayAtmosphericState{FT},
    glaycol,
) where {FT<:AbstractFloat}
    # computing Planck sources
    glay, gcol = glaycol
    @unpack t_lay, t_lev = as
    @unpack lay_source, lev_source_inc, lev_source_dec, sfc_source = source

    sbc = FT(Stefan())
    @inbounds lay_source[glaycol...] = sbc * t_lay[glaycol...]^FT(4) / FT(π)   # computing lay_source
    @inbounds lev_source_inc[glaycol...] =
        sbc * t_lev[glay+1, gcol]^FT(4) / FT(π)
    @inbounds lev_source_dec[glaycol...] = sbc * t_lev[glaycol...]^FT(4) / FT(π)
    if glay == 1
        @inbounds sfc_source[gcol] = sbc * as.t_sfc[gcol]^FT(4) / FT(π)   # computing sfc_source
    end
end