
function add_cloud_optics_2stream(op::OneScalar, args...)
    return nothing
end

function add_cloud_optics_2stream(
    op::TwoStream,
    as::AtmosphericState,
    lkp::LookUpLW,
    lkp_cld,
    glaycol,
    ibnd,
    igpt,
)
    if as.cld_mask isa AbstractArray
        τ_cl, τ_cl_ssa, τ_cl_ssag =
            compute_cld_props(lkp_cld, as, glaycol..., ibnd, igpt)
        increment!(op, τ_cl, τ_cl_ssa, τ_cl_ssag, glaycol..., igpt)
    end
    return nothing
end

function add_cloud_optics_2stream(
    op::TwoStream,
    as::AtmosphericState,
    lkp::LookUpSW,
    lkp_cld,
    glaycol,
    ibnd,
    igpt,
)
    if as.cld_mask isa AbstractArray
        τ_cl, τ_cl_ssa, τ_cl_ssag =
            compute_cld_props(lkp_cld, as, glaycol..., ibnd, igpt)
        τ_cl, τ_cl_ssa, τ_cl_ssag = delta_scale(τ_cl, τ_cl_ssa, τ_cl_ssag)
        increment!(op, τ_cl, τ_cl_ssa, τ_cl_ssag, glaycol..., igpt)
    end
    return nothing
end

function compute_cld_props(lkp_cld, as, glay, gcol, ibnd, igpt)
    use_lut = lkp_cld.use_lut
    cld_mask = as.cld_mask[glay, gcol]
    re_liq = as.cld_r_eff_liq[glay, gcol]
    re_ice = as.cld_r_eff_ice[glay, gcol]
    ice_rgh = as.ice_rgh
    cld_path_liq = as.cld_path_liq[glay, gcol]
    cld_path_ice = as.cld_path_ice[glay, gcol]
    FT = eltype(re_liq)
    τl, τl_ssa, τl_ssag = FT(0), FT(0), FT(0)
    τi, τi_ssa, τi_ssag = FT(0), FT(0), FT(0)
    if use_lut # use LUT interpolation
        @unpack lut_extliq,
        lut_ssaliq,
        lut_asyliq,
        lut_extice,
        lut_ssaice,
        lut_asyice,
        radliq_lwr,
        radliq_upr,
        radice_lwr,
        radice_upr,
        nsize_liq,
        nsize_ice = lkp_cld
        Δr_liq = (radliq_upr - radliq_lwr) / FT(nsize_liq - 1)
        Δr_ice = (radice_upr - radice_lwr) / FT(nsize_ice - 1)
        # cloud liquid particles
        if cld_path_liq > eps(FT)
            loc = Int(max(
                min(
                    unsafe_trunc(Int, (re_liq - radliq_lwr) / Δr_liq) + 1,
                    nsize_liq - 1,
                ),
                1,
            ))
            fac = (re_liq - radliq_lwr - (loc - 1) * Δr_liq) / Δr_liq
            fc1 = FT(1) - fac
            τl =
                (fc1 * lut_extliq[loc, ibnd] + fac * lut_extliq[loc+1, ibnd]) *
                cld_path_liq
            τl_ssa =
                (fc1 * lut_ssaliq[loc, ibnd] + fac * lut_ssaliq[loc+1, ibnd]) *
                τl
            τl_ssag =
                (fc1 * lut_asyliq[loc, ibnd] + fac * lut_asyliq[loc+1, ibnd]) *
                τl_ssa
        end
        # cloud ice particles
        if cld_path_ice > eps(FT)
            loc = Int(max(
                min(
                    unsafe_trunc(Int, (re_ice - radice_lwr) / Δr_ice) + 1,
                    nsize_ice - 1,
                ),
                1,
            ))
            fac = (re_ice - radice_lwr - (loc - 1) * Δr_ice) / Δr_ice
            fc1 = FT(1) - fac
            τi =
                (
                    fc1 * lut_extice[loc, ibnd, ice_rgh] +
                    fac * lut_extice[loc+1, ibnd, ice_rgh]
                ) * cld_path_ice
            τi_ssa =
                (
                    fc1 * lut_ssaice[loc, ibnd, ice_rgh] +
                    fac * lut_ssaice[loc+1, ibnd, ice_rgh]
                ) * τi
            τi_ssag =
                (
                    fc1 * lut_asyice[loc, ibnd, ice_rgh] +
                    fac * lut_asyice[loc+1, ibnd, ice_rgh]
                ) * τi_ssa
        end
    else # use pade interpolation
        @unpack pade_extliq,
        pade_ssaliq,
        pade_asyliq,
        pade_extice,
        pade_ssaice,
        pade_asyice,
        pade_sizreg_extliq,
        pade_sizreg_ssaliq,
        pade_sizreg_asyliq,
        pade_sizreg_extice,
        pade_sizreg_ssaice,
        pade_sizreg_asyice = lkp_cld
        m_ext, m_ssa_g = 3, 3
        n_ext, n_ssa_g = 3, 2
        # Finds index into size regime table
        # This works only if there are precisely three size regimes (four bounds) and it's
        # previously guaranteed that size_bounds(1) <= size <= size_bounds(4)
        if cld_path_liq > eps(FT)
            irad = Int(min(
                floor(
                    (re_liq - pade_sizreg_extliq[2]) / pade_sizreg_extliq[3],
                ) + 2,
                3,
            ))
            τl =
                pade_eval(ibnd, re_liq, irad, m_ext, n_ext, pade_extliq) *
                cld_path_liq

            irad = Int(min(
                floor(
                    (re_liq - pade_sizreg_ssaliq[2]) / pade_sizreg_ssaliq[3],
                ) + 2,
                3,
            ))
            τl_ssa =
                (
                    FT(1) - max(
                        FT(0),
                        pade_eval(
                            ibnd,
                            re_liq,
                            irad,
                            m_ssa_g,
                            n_ssa_g,
                            pade_ssaliq,
                        ),
                    )
                ) * τl

            irad = Int(min(
                floor(
                    (re_liq - pade_sizreg_asyliq[2]) / pade_sizreg_asyliq[3],
                ) + 2,
                3,
            ))
            τl_ssag =
                pade_eval(ibnd, re_liq, irad, m_ssa_g, n_ssa_g, pade_asyliq) *
                τl_ssa
        end

        if cld_path_ice > eps(FT)
            irad = Int(min(
                floor(
                    (re_ice - pade_sizreg_extice[2]) / pade_sizreg_extice[3],
                ) + 2,
                3,
            ))

            τi =
                pade_eval(
                    ibnd,
                    re_ice,
                    irad,
                    m_ext,
                    n_ext,
                    pade_extice,
                    ice_rgh,
                ) * cld_path_ice

            irad = Int(min(
                floor(
                    (re_ice - pade_sizreg_ssaice[2]) / pade_sizreg_ssaice[3],
                ) + 2,
                3,
            ))
            τi_ssa =
                (
                    FT(1) - max(
                        FT(0),
                        pade_eval(
                            ibnd,
                            re_ice,
                            irad,
                            m_ssa_g,
                            n_ssa_g,
                            pade_ssaice,
                            ice_rgh,
                        ),
                    )
                ) * τi

            irad = Int(min(
                floor(
                    (re_ice - pade_sizreg_asyice[2]) / pade_sizreg_asyice[3],
                ) + 2,
                3,
            ))
            τi_ssag =
                pade_eval(
                    ibnd,
                    re_ice,
                    irad,
                    m_ssa_g,
                    n_ssa_g,
                    pade_asyice,
                    ice_rgh,
                ) * τi_ssa
        end
    end

    τ = τl + τi
    τ_ssa = τl_ssa + τi_ssa
    τ_ssag = (τl_ssag + τi_ssag) / max(eps(FT), τ_ssa)
    τ_ssa /= max(eps(FT), τ)

    return (τ, τ_ssa, τ_ssag)
end

function pade_eval(
    ibnd,
    re,
    irad,
    m,
    n,
    pade_coeffs,
    irgh::Union{Int,Nothing} = nothing,
)
    FT = eltype(re)
    if irgh isa Int
        coeffs = view(pade_coeffs, :, :, :, irgh)

    else
        coeffs = pade_coeffs
    end

    denom = coeffs[ibnd, irad, n+m]
    for i = (n+m-1):-1:(1+m)
        denom = coeffs[ibnd, irad, i] + re * denom
    end
    denom = FT(1) + re * denom

    numer = coeffs[ibnd, irad, m]
    for i = (m-1):-1:2
        numer = coeffs[ibnd, irad, i] + re * numer
    end
    numer = coeffs[ibnd, irad, 1] + re * numer

    return (numer / denom)
end