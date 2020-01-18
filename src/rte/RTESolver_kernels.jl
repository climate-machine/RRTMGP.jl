#=
Numeric calculations for radiative transfer solvers.
 Emission/absorption (no-scattering) calculations
   solver for multi-angle Gaussian quadrature
   solver for a single angle, calling
     source function computation (linear-in-τ)
     transport
 Extinction-only calculation (direct solar beam)
 Two-stream calculations
   solvers for LW and SW with different boundary conditions and source functions
     source function calculation for LW, SW
     two-stream calculations for LW, SW (using different assumptions about phase function)
     transport (adding)
 Application of boundary conditions
=#


LW_diff_sec(::Type{FT}) where FT = FT(1.66)  # 1./cos(diffusivity angle)


#####
##### Top-level long-wave kernels
#####

#
# LW fluxes, no scattering, μ (cosine of integration angle) specified by column
#  Does radiation calculation at user-supplied angles; converts radiances to flux
#  using user-supplied weights
#

"""
    lw_solver_noscat!(ncol::I, nlay::I, ngpt::I,
                      mo::MeshOrientation{I},
                      D::Array{FT,2},
                      w_μ::FT,
                      τ::Array{FT,3},
                      source::SourceFuncLongWave{FT,I},
                      sfc_emis::Array{FT,2},
                      radn_up::Array{FT,3},
                      radn_dn::Array{FT,3}) where {FT<:AbstractFloat,I<:Int}

 - `source` longwave source function, see [`SourceFuncLongWave`](@ref)
 - `mo` mesh orientation, see [`MeshOrientation`](@ref)
 - `optical_props` 2-stream optical properties, see [`TwoStream`](@ref)
 - `sfc_emis` - surface emissivity
 - `radn_up` - upward radiance [W/m2-str]
 - `radn_dn` - downward radiance [W/m2-str] Top level must contain incident flux boundary condition

real(FT), dimension(ncol,       ngpt), intent(in   ) :: D            # secant of propagation angle  []
real(FT),                              intent(in   ) :: w_μ       # quadrature weight
real(FT), dimension(ncol,nlay,  ngpt), intent(in   ) :: τ          # Absorption optical thickness []

# Local variables, no g-point dependency
real(FT), dimension(ncol,nlay) :: τ_loc,   # path length (τ/μ)
                                  trans    # transmissivity  = exp(-τ)
real(FT), dimension(ncol,nlay) :: source_dn, source_up
real(FT), dimension(ncol     ) :: source_sfc, sfc_albedo

real(FT), dimension(:,:,:), pointer :: lev_source_up, lev_source_dn # Mapping increasing/decreasing indicies to up/down
"""
function lw_solver_noscat!(ncol::I, nlay::I, ngpt::I,
                           mo::MeshOrientation{I},
                           D::Array{FT,2},
                           w_μ::FT,
                           τ::Array{FT,3},
                           source::SourceFuncLongWave{FT,I},
                           sfc_emis::Array{FT,2},
                           radn_up::Array{FT,3},
                           radn_dn::Array{FT,3}) where {FT<:AbstractFloat,I<:Int}
  τ_loc    = Array{FT}(undef,ncol,nlay)
  trans      = Array{FT}(undef,ncol,nlay)
  source_up  = Array{FT}(undef,ncol,nlay)
  source_dn  = Array{FT}(undef,ncol,nlay)
  source_sfc = Array{FT}(undef,ncol)
  sfc_albedo = Array{FT}(undef,ncol)
  # Which way is up?
  # Level Planck sources for upward and downward radiation
  # When top_at_1, lev_source_up => lev_source_dec
  #                lev_source_dn => lev_source_inc, and vice-versa
  i_lev_top = ilev_top(mo)
  lev_source_up = mo.top_at_1 ? source.lev_source_dec : source.lev_source_inc
  lev_source_dn = mo.top_at_1 ? source.lev_source_inc : source.lev_source_dec

  @inbounds for igpt in 1:ngpt
    #
    # Transport is for intensity
    #   convert flux at top of domain to intensity assuming azimuthal isotropy
    #
    radn_dn[:,i_lev_top,igpt] ./= FT(2 * π) * w_μ

    #
    # Optical path and transmission, used in source function and transport calculations
    #
    @inbounds for ilev in 1:nlay
      τ_loc[:,ilev] .= τ[:,ilev,igpt] .* D[:,igpt]
      trans[:,ilev] .= exp.(-τ_loc[:,ilev])
    end
    #
    # Source function for diffuse radiation
    #
    lw_source_noscat!(ncol, nlay,
                      source.lay_source[:,:,igpt],
                      lev_source_up[:,:,igpt],
                      lev_source_dn[:,:,igpt],
                      τ_loc,
                      trans,
                      source_dn,
                      source_up)
    #
    # Surface albedo, surface source function
    #
    sfc_albedo .= FT(1) .- sfc_emis[:,igpt]
    source_sfc .= sfc_emis[:,igpt] .* source.sfc_source[:,igpt]
    #
    # Transport
    #
    lw_transport_noscat!(ncol, nlay, mo,
                         τ_loc,
                         trans,
                         sfc_albedo,
                         source_dn,
                         source_up,
                         source_sfc,
                         @view(radn_up[:,:,igpt]),
                         @view(radn_dn[:,:,igpt]))
    #
    # Convert intensity to flux assuming azimuthal isotropy and quadrature weight
    #
    radn_dn[:,:,igpt] .*= FT(2 * π) * w_μ
    radn_up[:,:,igpt] .*= FT(2 * π) * w_μ
  end  # g point loop

end

"""
    lw_solver_noscat_GaussQuad!(ncol::I, nlay::I, ngpt::I,
                                mo::MeshOrientation{I},
                                angle_disc::GaussQuadrature{FT,I},
                                τ::Array{FT,3},
                                source::SourceFuncLongWave{FT, I},
                                sfc_emis::Array{FT,2},
                                flux_up::Array{FT,3},
                                flux_dn::Array{FT,3}) where {I<:Int,FT<:AbstractFloat}

LW transport, no scattering, multi-angle quadrature
Users provide a set of weights and quadrature angles
Routine sums over single-angle solutions for each sets of angles/weights

given

 - `source` longwave source function, see [`SourceFuncLongWave`](@ref)
 - `mo` mesh orientation, see [`MeshOrientation`](@ref)
 - `optical_props` 2-stream optical properties, see [`TwoStream`](@ref)
 - `sfc_emis` - surface emissivity
 - `angle_disc` - angular discretization, see [`GaussQuadrature`](@ref)
 - `flux_up` upward radiance [W/m2-str]
 - `flux_dn` downward radiance, Top level must contain incident flux boundary condition
 - `τ` Absorption optical thickness [ncol,nlay,  ngpt]

 - `radn_dn` Fluxes per quad angle [ncol,nlay+1,ngpt]
 - `radn_up` Fluxes per quad angle [ncol,nlay+1,ngpt]

# Local variables
real(FT), dimension(ncol,       ngpt) :: Ds_ncol
"""
function lw_solver_noscat_GaussQuad!(ncol::I, nlay::I, ngpt::I,
                                     mo::MeshOrientation{I},
                                     angle_disc::GaussQuadrature{FT,I},
                                     τ::Array{FT,3},
                                     source::SourceFuncLongWave{FT, I},
                                     sfc_emis::Array{FT,2},
                                     flux_up::Array{FT,3},
                                     flux_dn::Array{FT,3}) where {I<:Int,FT<:AbstractFloat}

  n_μ = angle_disc.n_gauss_angles
  Ds = angle_disc.gauss_Ds[1:n_μ,n_μ]
  w_μ = angle_disc.gauss_wts[1:n_μ,n_μ]
  #
  # For the first angle output arrays store total flux
  #
  Ds_ncol = Array{FT}(undef, ncol,ngpt)
  radn_up, radn_dn = ntuple(i->Array{FT}(undef, ncol,nlay+1,ngpt), 2)

  Ds_ncol .= Ds[1]
  lw_solver_noscat!(ncol, nlay, ngpt,
                    mo,
                    Ds_ncol,
                    w_μ[1],
                    τ,
                    source,
                    sfc_emis,
                    flux_up,
                    flux_dn)
  #
  # For more than one angle use local arrays
  #
  i_lev_top = ilev_top(mo)
  apply_BC!(radn_dn, i_lev_top, flux_dn[:,i_lev_top,:])

  @inbounds for i_μ in 2:n_μ
    Ds_ncol .= Ds[i_μ]
    lw_solver_noscat!(ncol, nlay, ngpt,
                      mo,
                      Ds_ncol,
                      w_μ[i_μ],
                      τ,
                      source,
                      sfc_emis,
                      radn_up,
                      radn_dn)
    flux_up .+= radn_up
    flux_dn .+= radn_dn
  end
end

"""
    lw_solver!(ncol::I, nlay::I, ngpt::I,
               mo::MeshOrientation{I},
               optical_props::TwoStream{FT},
               source::SourceFuncLongWave{FT},
               sfc_emis::Array{FT},
               flux_up::Array{FT},
               flux_dn::Array{FT}) where {I<:Int,FT<:AbstractFloat}

Longwave calculation:
 - combine RRTMGP-specific sources at levels
 - compute layer reflectance, transmittance
 - compute total source function at levels using linear-in-τ
 - transport

given

 - `source` longwave source function, see [`SourceFuncLongWave`](@ref)
 - `mo` mesh orientation, see [`MeshOrientation`](@ref)
 - `optical_props` 2-stream optical properties, see [`TwoStream`](@ref)
 - `sfc_emis` - surface emissivity
 - `flux_up` - upward flux [W/m2]
 - `flux_dn` - downward flux [W/m2] Top level must contain incident flux boundary condition

real(FT), dimension(ncol,nlay  ) :: Rdif, Tdif, γ_1, γ_2
real(FT), dimension(ncol       ) :: sfc_albedo
real(FT), dimension(ncol,nlay+1) :: lev_source
real(FT), dimension(ncol,nlay  ) :: source_dn, source_up
real(FT), dimension(ncol       ) :: source_sfc
"""
function lw_solver!(ncol::I, nlay::I, ngpt::I,
                    mo::MeshOrientation{I},
                    optical_props::TwoStream{FT,I},
                    source::SourceFuncLongWave{FT},
                    sfc_emis::Array{FT,2},
                    flux_up::Array{FT,3},
                    flux_dn::Array{FT,3}) where {I<:Int,FT<:AbstractFloat}

  lev_source = Array{FT}(undef, ncol,nlay+1)
  source_sfc = Array{FT}(undef, ncol)
  source_dn = Array{FT}(undef, ncol, nlay)
  source_up = Array{FT}(undef, ncol, nlay)
  sfc_albedo = Array{FT}(undef, ncol)
  Rdif, Tdif, γ_1, γ_2 = ntuple(i->Array{FT}(undef, ncol, nlay),4)
  @inbounds for igpt in 1:ngpt
    #
    # RRTMGP provides source functions at each level using the spectral mapping
    #   of each adjacent layer. Combine these for two-stream calculations
    #
    lw_combine_sources!(ncol, nlay, mo,
                        source.lev_source_inc[:,:,igpt],
                        source.lev_source_dec[:,:,igpt],
                        lev_source)
    #
    # Cell properties: reflection, transmission for diffuse radiation
    #   Coupling coefficients needed for source function
    #
    lw_two_stream!(ncol, nlay,
                   optical_props.τ[:,:,igpt],
                   optical_props.ssa[:,:,igpt],
                   optical_props.g[:,:,igpt],
                   γ_1,
                   γ_2,
                   Rdif,
                   Tdif)
    #
    # Source function for diffuse radiation
    #
    lw_source_2str!(ncol, nlay, mo,
                    sfc_emis[:,igpt],
                    source.sfc_source[:,igpt],
                    source.lay_source[:,:,igpt],
                    lev_source,
                    γ_1,
                    γ_2,
                    Rdif,
                    Tdif,
                    optical_props.τ[:,:,igpt],
                    source_dn,
                    source_up,
                    source_sfc)
    #
    # Transport
    #
    sfc_albedo .= FT(1) .- sfc_emis[:,igpt]
    adding!(ncol, nlay, mo,
            sfc_albedo,
            Rdif,
            Tdif,
            source_dn,
            source_up,
            source_sfc,
            @view(flux_up[:,:,igpt]),
            @view(flux_dn[:,:,igpt]))
  end

end

#####
#####   Top-level shortwave kernels
#####

#####
#####   Extinction-only i.e. solar direct beam
#####

"""
    sw_solver_noscat!(ncol::I, nlay::I, ngpt::I,
                      mo::MeshOrientation{I},
                      τ::Array{FT},
                      μ_0::Array{FT,1},
                      flux_dir::Array{FT,3}) where {FT<:AbstractFloat,I<:Int}

 - `mo` mesh orientation, see [`MeshOrientation`](@ref)

integer,                    intent( in) :: ncol, nlay, ngpt # Number of columns, layers, g-points
real(FT), dimension(ncol,nlay,  ngpt), intent( in) :: τ          # Absorption optical thickness []
real(FT), dimension(ncol            ), intent( in) :: μ_0          # cosine of solar zenith angle
real(FT), dimension(ncol,nlay+1,ngpt), intent(inout) :: flux_dir     # Direct-beam flux, spectral [W/m2]
                                                                   # Top level must contain incident flux boundary condition
integer :: icol, ilev, igpt
real(FT) :: μ_0_inv(ncol)
"""
function sw_solver_noscat!(ncol::I, nlay::I, ngpt::I,
                           mo::MeshOrientation{I},
                           τ::Array{FT},
                           μ_0::Array{FT,1},
                           flux_dir::Array{FT,3}) where {FT<:AbstractFloat,I<:Int}

  μ_0_inv = 1 ./ μ_0
  # Indexing into arrays for upward and downward propagation depends on the vertical
  #   orientation of the arrays (whether the domain top is at the first or last index)
  # We write the loops out explicitly so compilers will have no trouble optimizing them.
  n̂ = nhat(mo)
  b = binary(mo)

  # Downward propagation
  # For the flux at this level, what was the previous level, and which layer has the
  #   radiation just passed through?
  @inbounds for igpt in 1:ngpt
    @inbounds for ilev in lev_range(mo)
      flux_dir[:,ilev,igpt] .= flux_dir[:,ilev-n̂,igpt] .* exp.(-τ[:,ilev-b,igpt].*μ_0_inv)
    end
  end
end

# Shortwave two-stream calculation:
#   compute layer reflectance, transmittance
#   compute solar source function for diffuse radiation
#   transport

"""
    sw_solver!(ncol::I, nlay::I, ngpt::I,
               mo::MeshOrientation{I},
               optical_props::TwoStream{FT,I},
               μ_0::Array{FT},
               sfc_alb_dir::Array{FT,2},
               sfc_alb_dif::Array{FT,2},
               flux_up::Array{FT,3},
               flux_dn::Array{FT,3},
               flux_dir::Array{FT,3}) where {FT<:AbstractFloat,I<:Int}

 - `mo` mesh orientation, see [`MeshOrientation`](@ref)
 - `optical_props` 2-stream optical properties, see [`TwoStream`](@ref)
 - `flux_up` - upward flux [W/m2]
 - `flux_dn` - downward flux [W/m2] Top level must contain incident flux boundary condition

real(FT), dimension(ncol            ), intent(in   ) :: μ_0     # cosine of solar zenith angle
real(FT), dimension(ncol,       ngpt), intent(in   ) :: sfc_alb_dir, sfc_alb_dif # Spectral albedo of surface to direct and diffuse radiation
# -------------------------------------------
real(FT), dimension(ncol,nlay) :: Rdif, Tdif, Rdir, Tdir, Tnoscat
real(FT), dimension(ncol,nlay) :: source_up, source_dn
real(FT), dimension(ncol     ) :: source_srf
# ------------------------------------
"""
function sw_solver!(ncol::I, nlay::I, ngpt::I,
                    mo::MeshOrientation{I},
                    optical_props::TwoStream{FT,I},
                    μ_0::Array{FT},
                    sfc_alb_dir::Array{FT,2},
                    sfc_alb_dif::Array{FT,2},
                    flux_up::Array{FT,3},
                    flux_dn::Array{FT,3},
                    flux_dir::Array{FT,3}) where {FT<:AbstractFloat,I<:Int}

  Rdif, Tdif, Rdir, Tdir, Tnoscat, source_up, source_dn = ntuple(i->zeros(FT, ncol,nlay), 7)
  source_srf = zeros(FT, ncol)

  @inbounds for igpt in 1:ngpt
    #
    # Cell properties: transmittance and reflectance for direct and diffuse radiation
    #

    sw_two_stream!(Rdif, Tdif, Rdir, Tdir, Tnoscat,
                   ncol, nlay, μ_0,
                   optical_props.τ[:,:,igpt],
                   optical_props.ssa[:,:,igpt],
                   optical_props.g[:,:,igpt])
    #
    # Direct-beam and source for diffuse radiation
    #
    sw_source_2str!(ncol, nlay, mo, Rdir, Tdir, Tnoscat, sfc_alb_dir[:,igpt],
                    source_up, source_dn, source_srf, @view(flux_dir[:,:,igpt]))

    #
    # Transport
    #
    adding!(ncol, nlay, mo,
            sfc_alb_dif[:,igpt], Rdif, Tdif,
            source_dn, source_up, source_srf,
            @view(flux_up[:,:,igpt]),
            @view(flux_dn[:,:,igpt]))

    #
    # adding computes only diffuse flux; flux_dn is total
    #
    flux_dn[:,:,igpt] .= flux_dn[:,:,igpt] .+ flux_dir[:,:,igpt]
  end
end

#####
#####   Lower-level longwave kernels
#####

#
# Compute LW source function for upward and downward emission at levels using linear-in-τ assumption
# See Clough et al., 1992, doi: 10.1029/92JD01419, Eq 13
#

"""
    lw_source_noscat!(ncol::I, nlay::I,
                      lay_source::Array{FT,2},
                      lev_source_up::Array{FT,2},
                      lev_source_dn::Array{FT,2},
                      τ::Array{FT,2},
                      trans::Array{FT,2},
                      source_dn::Array{FT,2},
                      source_up::Array{FT,2}) where {I<:Int,FT<:AbstractFloat}

integer,                         intent(in) :: ncol, nlay
real(FT), dimension(ncol, nlay), intent(in) :: lay_source,  # Planck source at layer center
                                               lev_source_up,  # Planck source at levels (layer edges),
                                               lev_source_dn,  #   increasing/decreasing layer index
                                               τ,         # Optical path (τ/μ)
                                               trans         # Transmissivity (exp(-τ))
real(FT), dimension(ncol, nlay), intent(out):: source_dn, source_up
                                                               # Source function at layer edges
                                                               # Down at the bottom of the layer, up at the top
real(FT), parameter :: τ_thresh = sqrt(epsilon(τ))

"""
function lw_source_noscat!(ncol::I, nlay::I,
                           lay_source::Array{FT,2},
                           lev_source_up::Array{FT,2},
                           lev_source_dn::Array{FT,2},
                           τ::Array{FT,2},
                           trans::Array{FT,2},
                           source_dn::Array{FT,2},
                           source_up::Array{FT,2}) where {I<:Int,FT<:AbstractFloat}

  τ_thresh = sqrt(eps(eltype(τ)))
#    τ_thresh = abs(eps(eltype(τ)))
  @inbounds for ilay in 1:nlay
    @inbounds for icol in 1:ncol
    #
    # Weighting factor. Use 2nd order series expansion when rounding error (~τ^2)
    #   is of order epsilon (smallest difference from 1. in working precision)
    #   Thanks to Peter Blossey
    #
    fact = fmerge((FT(1) - trans[icol,ilay])/τ[icol,ilay] - trans[icol,ilay],
                  τ[icol, ilay] * ( FT(0.5) - FT(1)/FT(3)*τ[icol, ilay]   ),
                  τ[icol, ilay] > τ_thresh)
    #
    # Equation below is developed in Clough et al., 1992, doi:10.1029/92JD01419, Eq 13
    #
    source_dn[icol,ilay] = (FT(1) - trans[icol,ilay]) * lev_source_dn[icol,ilay] +
                            FT(2) * fact * (lay_source[icol,ilay] - lev_source_dn[icol,ilay])
    source_up[icol,ilay] = (FT(1) - trans[icol,ilay]) * lev_source_up[icol,ilay] +
                            FT(2) * fact * (lay_source[icol,ilay] - lev_source_up[icol,ilay])
    end
  end
end

#####
##### Longwave no-scattering transport
#####

"""
    lw_transport_noscat!(ncol::I, nlay::I,
                         mo::MeshOrientation{I},
                         τ::Array{FT,2},
                         trans::Array{FT,2},
                         sfc_albedo::Array{FT,1},
                         source_dn::Array{FT,2},
                         source_up::Array{FT,2},
                         source_sfc::Array{FT,1},
                         radn_up::AbstractArray{FT,2},
                         radn_dn::AbstractArray{FT,2}) where {I<:Int, FT<:AbstractFloat}

 - `mo` mesh orientation, see [`MeshOrientation`](@ref)

integer,                          intent(in   ) :: ncol, nlay # Number of columns, layers, g-points
real(FT), dimension(ncol,nlay  ), intent(in   ) :: τ,         # Absorption optical thickness, pre-divided by μ []
                                                   trans      # transmissivity = exp(-τ)
real(FT), dimension(ncol       ), intent(in   ) :: sfc_albedo # Surface albedo
real(FT), dimension(ncol,nlay  ), intent(in   ) :: source_dn,
                                                   source_up  # Diffuse radiation emitted by the layer
real(FT), dimension(ncol       ), intent(in   ) :: source_sfc # Surface source function [W/m2]
real(FT), dimension(ncol,nlay+1), intent(  out) :: radn_up    # Radiances [W/m2-str]
real(FT), dimension(ncol,nlay+1), intent(inout) :: radn_dn    # Top level must contain incident flux boundary condition
# Local variables
integer :: ilev
"""
function lw_transport_noscat!(ncol::I, nlay::I,
                              mo::MeshOrientation{I},
                              τ::Array{FT,2},
                              trans::Array{FT,2},
                              sfc_albedo::Array{FT,1},
                              source_dn::Array{FT,2},
                              source_up::Array{FT,2},
                              source_sfc::Array{FT,1},
                              radn_up::AbstractArray{FT,2},
                              radn_dn::AbstractArray{FT,2}) where {I<:Int, FT<:AbstractFloat}
  n̂ = nhat(mo)
  b = binary(mo)
  n_sfc = ilev_bot(mo)

  # Downward propagation
  @inbounds for ilev in lev_range(mo)
    radn_dn[:,ilev] .= trans[:,ilev-b].*radn_dn[:,ilev-n̂] .+ source_dn[:,ilev-b]
  end

  # Surface reflection and emission
  radn_up[:,n_sfc] .= radn_dn[:,n_sfc].*sfc_albedo .+ source_sfc

  # Upward propagation
  @inbounds for ilev in lev_range_reversed(mo)
    radn_up[:,ilev] .= trans[:,ilev-1+b].*radn_up[:,ilev+n̂] .+ source_up[:,ilev-1+b]
  end
end

# -------------------------------------------------------------------------------------------------
#
# Longwave two-stream solutions to diffuse reflectance and transmittance for a layer
#    with optical depth τ, single scattering albedo w0, and asymmetery parameter g.
#
# Equations are developed in Meador and Weaver, 1980,
#    doi:10.1175/1520-0469(1980)037<0630:TSATRT>2.0.CO;2
#
# -------------------------------------------------------------------------------------------------

"""
    lw_two_stream!(ncol::I, nlay::I,
                   τ::Array{FT,2},
                   w0::Array{FT,2},
                   g::Array{FT,2},
                   γ_1::Array{FT,2},
                   γ_2::Array{FT,2},
                   Rdif::Array{FT,2},
                   Tdif::Array{FT,2}) where {I<:Int, FT<:AbstractFloat}

integer,                        intent(in)  :: ncol, nlay
real(FT), dimension(ncol,nlay), intent(in)  :: τ, w0, g
real(FT), dimension(ncol,nlay), intent(out) :: γ_1, γ_2, Rdif, Tdif

# -----------------------
integer  :: i, j

# Variables used in Meador and Weaver
real(FT) :: k(ncol)

# Ancillary variables
real(FT) :: RT_term(ncol)
real(FT) :: exp_minuskτ(ncol), exp_minus2kτ(ncol)
"""
function lw_two_stream!(ncol::I, nlay::I,
                        τ::Array{FT,2},
                        w0::Array{FT,2},
                        g::Array{FT,2},
                        γ_1::Array{FT,2},
                        γ_2::Array{FT,2},
                        Rdif::Array{FT,2},
                        Tdif::Array{FT,2}) where {I<:Int, FT<:AbstractFloat}
  k = Vector{FT}(undef, ncol)
  RT_term = Vector{FT}(undef, ncol)
  exp_minuskτ = Vector{FT}(undef, ncol)
  exp_minus2kτ = Vector{FT}(undef, ncol)
  @inbounds for j in 1:nlay
    @inbounds for i in 1:ncol
      #
      # Coefficients differ from SW implementation because the phase function is more isotropic
      #   Here we follow Fu et al. 1997, doi:10.1175/1520-0469(1997)054<2799:MSPITI>2.0.CO;2
      #
      γ_1[i,j] = LW_diff_sec(FT) * (FT(1) - FT(0.5) * w0[i,j] * (FT(1) + g[i,j])) # Fu et al. Eq 2.9
      γ_2[i,j] = LW_diff_sec(FT) *          FT(0.5) * w0[i,j] * (FT(1) - g[i,j])  # Fu et al. Eq 2.10
    end

    # Written to encourage vectorization of exponential, square root
    # Eq 18;  k = SQRT(γ_1**2 - γ_2**2), limited below to avoid div by 0.
    #   k = 0 for isotropic, conservative scattering; this lower limit on k
    #   gives relative error with respect to conservative solution
    #   of < 0.1% in Rdif down to τ = 10^-9
    temp1 = γ_1[1:ncol,j] .- γ_2[1:ncol,j] .*
            γ_1[1:ncol,j] .+ γ_2[1:ncol,j]

    temp2 = max.(temp1, FT(1.e-12))
    k .= sqrt.(temp2)
    exp_minuskτ .= exp.(-τ[1:ncol,j].*k)

    #
    # Diffuse reflection and transmission
    #
    @inbounds for i in 1:ncol
      exp_minus2kτ[i] = exp_minuskτ[i] * exp_minuskτ[i]

      # Refactored to avoid rounding errors when k, γ_1 are of very different magnitudes
      RT_term[i] = FT(1) / (k[i] * (FT(1) + exp_minus2kτ[i])  +
                            γ_1[i,j] * (FT(1) - exp_minus2kτ[i]) )

      # Equation 25
      Rdif[i,j] = RT_term[i] * γ_2[i,j] * (FT(1) - exp_minus2kτ[i])

      # Equation 26
      Tdif[i,j] = RT_term[i] * FT(2) * k[i] * exp_minuskτ[i]
    end

  end
end

# -------------------------------------------------------------------------------------------------
#
# Source function combination
# RRTMGP provides two source functions at each level
#   using the spectral mapping from each of the adjacent layers.
#   Need to combine these for use in two-stream calculation.
#
# -------------------------------------------------------------------------------------------------

"""
    lw_combine_sources!(ncol::I, nlay::I,
                        mo::MeshOrientation{I},
                        lev_src_inc::Array{FT,2},
                        lev_src_dec::Array{FT,2},
                        lev_source::Array{FT,2}) where {I<:Int,FT<:AbstractFloat}

 - `mo` mesh orientation, see [`MeshOrientation`](@ref)

integer,                           intent(in ) :: ncol, nlay
real(FT), dimension(ncol, nlay  ), intent(in ) :: lev_src_inc, lev_src_dec
real(FT), dimension(ncol, nlay+1), intent(out) :: lev_source
"""
function lw_combine_sources!(ncol::I, nlay::I,
                             mo::MeshOrientation{I},
                             lev_src_inc::Array{FT,2},
                             lev_src_dec::Array{FT,2},
                             lev_source::Array{FT,2}) where {I<:Int,FT<:AbstractFloat}
  ilay = 1
  @inbounds for icol in 1:ncol
    lev_source[icol, ilay] =        lev_src_dec[icol, ilay]
  end
  @inbounds for ilay in 2:nlay
    @inbounds for icol in 1:ncol
      lev_source[icol, ilay] = sqrt(lev_src_dec[icol, ilay] * lev_src_inc[icol, ilay-1])
    end
  end
  ilay = nlay+1
  @inbounds for icol in 1:ncol
    lev_source[icol, ilay] = lev_src_inc[icol, ilay-1]
  end

end

# ---------------------------------------------------------------
#
# Compute LW source function for upward and downward emission at levels using linear-in-τ assumption
#   This version straight from ECRAD
#   Source is provided as W/m2-str; factor of π converts to flux units
#
# ---------------------------------------------------------------

"""
    lw_source_2str!(ncol::I, nlay::I,
                    mo::MeshOrientation{I},
                    sfc_emis::Array{FT,1},
                    sfc_src::Array{FT,1},
                    lay_source::Array{FT,2},
                    lev_source::Array{FT,2},
                    γ_1::Array{FT,2},
                    γ_2::Array{FT,2},
                    rdif::Array{FT,2},
                    tdif::Array{FT,2},
                    τ::Array{FT,2},
                    source_dn::Array{FT,2},
                    source_up::Array{FT,2},
                    source_sfc::Array{FT,1}) where {I<:Int,FT<:AbstractFloat}

 - `mo` mesh orientation, see [`MeshOrientation`](@ref)

integer,                         intent(in) :: ncol, nlay
real(FT), dimension(ncol      ), intent(in) :: sfc_emis, sfc_src
real(FT), dimension(ncol, nlay), intent(in) :: lay_source,     # Planck source at layer center
                                             τ,            # Optical depth (τ)
                                             γ_1, γ_2, # Coupling coefficients
                                             rdif, tdif       # Layer reflectance and transmittance
real(FT), dimension(ncol, nlay+1), target,
                               intent(in)  :: lev_source       # Planck source at layer edges
real(FT), dimension(ncol, nlay), intent(out) :: source_dn, source_up
real(FT), dimension(ncol      ), intent(out) :: source_sfc      # Source function for upward radation at surface

integer             :: icol, ilay
real(FT)            :: Z, Zup_top, Zup_bottom, Zdn_top, Zdn_bottom
real(FT), dimension(:), pointer :: lev_source_bot, lev_source_top
"""
function lw_source_2str!(ncol::I, nlay::I,
                         mo::MeshOrientation{I},
                         sfc_emis::Array{FT,1},
                         sfc_src::Array{FT,1},
                         lay_source::Array{FT,2},
                         lev_source::Array{FT,2},
                         γ_1::Array{FT,2},
                         γ_2::Array{FT,2},
                         rdif::Array{FT,2},
                         tdif::Array{FT,2},
                         τ::Array{FT,2},
                         source_dn::Array{FT,2},
                         source_up::Array{FT,2},
                         source_sfc::Array{FT,1}) where {I<:Int,FT<:AbstractFloat}
  b = binary(mo)
  @inbounds for ilay in 1:nlay
    lev_source_top = lev_source[:,ilay+1-b]
    lev_source_bot = lev_source[:,ilay+b]
    @inbounds for icol in 1:ncol
      if τ[icol,ilay] > FT(1.0e-8)
        #
        # Toon et al. (JGR 1989) Eqs 26-27
        #
        Z = (lev_source_bot[icol]-lev_source_top[icol]) / (τ[icol,ilay]*(γ_1[icol,ilay]+γ_2[icol,ilay]))
        Zup_top        =  Z + lev_source_top[icol]
        Zup_bottom     =  Z + lev_source_bot[icol]
        Zdn_top        = -Z + lev_source_top[icol]
        Zdn_bottom     = -Z + lev_source_bot[icol]
        source_up[icol,ilay] = π * (Zup_top    - rdif[icol,ilay] * Zdn_top    - tdif[icol,ilay] * Zup_bottom)
        source_dn[icol,ilay] = π * (Zdn_bottom - rdif[icol,ilay] * Zup_bottom - tdif[icol,ilay] * Zdn_top)
      else
        source_up[icol,ilay] = FT(0)
        source_dn[icol,ilay] = FT(0)
      end
    end
  end
  @inbounds for icol in 1:ncol
    source_sfc[icol] = π * sfc_emis[icol] * sfc_src[icol]
  end
end

# -------------------------------------------------------------------------------------------------
#
#   Lower-level shortwave kernels
#
# -------------------------------------------------------------------------------------------------
#
# Two-stream solutions to direct and diffuse reflectance and transmittance for a layer
#    with optical depth τ, single scattering albedo w0, and asymmetry parameter g.
#
# Equations are developed in Meador and Weaver, 1980,
#    doi:10.1175/1520-0469(1980)037<0630:TSATRT>2.0.CO;2
#
# -------------------------------------------------------------------------------------------------

"""
    sw_two_stream!(Rdif::Array{FT,2},
                   Tdif::Array{FT,2},
                   Rdir::Array{FT,2},
                   Tdir::Array{FT,2},
                   Tnoscat::Array{FT,2},
                   ncol::I, nlay::I,
                   μ_0::Array{FT,1},
                   τ::Array{FT,2},
                   w0::Array{FT,2},
                   g::Array{FT,2}) where {I<:Int,FT<:AbstractFloat}

integer,                        intent(in)  :: ncol, nlay
real(FT), dimension(ncol),      intent(in)  :: μ_0
real(FT), dimension(ncol,nlay), intent(in)  :: τ, w0, g
real(FT), dimension(ncol,nlay), intent(out) :: Rdif, Tdif, Rdir, Tdir, Tnoscat

# -----------------------
integer  :: i, j

# Variables used in Meador and Weaver
real(FT) :: γ_1(ncol), γ_2(ncol), γ_3(ncol), γ_4(ncol)
real(FT) :: α_1(ncol), α_2(ncol), k(ncol)

# Ancillary variables
real(FT) :: RT_term(ncol)
real(FT) :: exp_minuskτ(ncol), exp_minus2kτ(ncol)
real(FT) :: k_μ, k_γ_3, k_γ_4
real(FT) :: μ_0_inv(ncol)
# ---------------------------------
"""
function sw_two_stream!(Rdif::Array{FT,2},
                        Tdif::Array{FT,2},
                        Rdir::Array{FT,2},
                        Tdir::Array{FT,2},
                        Tnoscat::Array{FT,2},
                        ncol::I, nlay::I,
                        μ_0::Array{FT,1},
                        τ::Array{FT,2},
                        w0::Array{FT,2},
                        g::Array{FT,2}) where {I<:Int,FT<:AbstractFloat}

  μ_0_inv = 1 ./ μ_0
  exp_minuskτ  = Array{FT}(undef, ncol)
  exp_minus2kτ = Array{FT}(undef, ncol)
  RT_term        = Array{FT}(undef, ncol)

  γ_1, γ_2, γ_3, γ_4, α_1, α_2, k = ntuple(i->zeros(FT, ncol), 7)

  @inbounds for j in 1:nlay
    @inbounds for i in 1:ncol
      # Zdunkowski Practical Improved Flux Method "PIFM"
      #  (Zdunkowski et al., 1980;  Contributions to Atmospheric Physics 53, 147-66)
      #
      γ_1[i]= (FT(8) - w0[i,j] * (FT(5) + FT(3) * g[i,j])) * FT(.25)
      γ_2[i]=  FT(3) *(w0[i,j] * (FT(1) -         g[i,j])) * FT(.25)
      γ_3[i]= (FT(2) - FT(3) * μ_0[i] *           g[i,j] ) * FT(.25)
      γ_4[i]=  FT(1) - γ_3[i]

      α_1[i] = γ_1[i] * γ_4[i] + γ_2[i] * γ_3[i]           # Eq. 16
      α_2[i] = γ_1[i] * γ_3[i] + γ_2[i] * γ_4[i]           # Eq. 17
    end

    # Written to encourage vectorization of exponential, square root
    # Eq 18;  k = SQRT(γ_1**2 - γ_2**2), limited below to avoid div by 0.
    #   k = 0 for isotropic, conservative scattering; this lower limit on k
    #   gives relative error with respect to conservative solution
    #   of < 0.1% in Rdif down to τ = 10^-9
    temp = (γ_1 .- γ_2) .* (γ_1 .+ γ_2)
    k .= sqrt.(max( temp..., FT(1.e-12)))
    exp_minuskτ .= exp.(-τ[1:ncol,j] .* k)

    #
    # Diffuse reflection and transmission
    #
    @inbounds for i in 1:ncol
      exp_minus2kτ[i] = exp_minuskτ[i] * exp_minuskτ[i]

      # Refactored to avoid rounding errors when k, γ_1 are of very different magnitudes
      RT_term[i] = FT(1) / (     k[i] * (FT(1) + exp_minus2kτ[i])  +
                            γ_1[i] * (FT(1) - exp_minus2kτ[i]) )

      # Equation 25
      Rdif[i,j] = RT_term[i] * γ_2[i] * (FT(1) - exp_minus2kτ[i])

      # Equation 26
      Tdif[i,j] = RT_term[i] * FT(2) * k[i] * exp_minuskτ[i]
    end

    #
    # Transmittance of direct, non-scattered beam. Also used below
    #
    Tnoscat[1:ncol,j] .= exp.(-τ[1:ncol,j] .* μ_0_inv)

    #
    # Direct reflect and transmission
    #
    @inbounds for i in 1:ncol
      k_μ     = k[i] * μ_0[i]
      k_γ_3 = k[i] * γ_3[i]
      k_γ_4 = k[i] * γ_4[i]

      #
      # Equation 14, multiplying top and bottom by exp(-k*τ)
      #   and rearranging to avoid div by 0.
      #
      RT_term[i] =  w0[i,j] * RT_term[i]/fmerge(FT(1) - k_μ*k_μ,
                                                eps(FT),
                                                abs(FT(1) - k_μ*k_μ) >= eps(FT))

      Rdir[i,j] = RT_term[i]  *
          ((FT(1) - k_μ) * (α_2[i] + k_γ_3)                     -
           (FT(1) + k_μ) * (α_2[i] - k_γ_3) * exp_minus2kτ[i] -
            FT(2) * (k_γ_3 - α_2[i] * k_μ)  * exp_minuskτ[i] * Tnoscat[i,j])

      #
      # Equation 15, multiplying top and bottom by exp(-k*τ),
      #   multiplying through by exp(-τ/μ_0) to
      #   prefer underflow to overflow
      # Omitting direct transmittance
      #
      Tdir[i,j] = -RT_term[i] *
                  ((FT(1) + k_μ) * (α_1[i] + k_γ_4)                     * Tnoscat[i,j] -
                   (FT(1) - k_μ) * (α_1[i] - k_γ_4) * exp_minus2kτ[i] * Tnoscat[i,j] -
                    FT(2) * (k_γ_4 + α_1[i] * k_μ)  * exp_minuskτ[i])

    end
  end
  return nothing
end

#####
##### Direct beam source for diffuse radiation in layers and at surface;
#####   report direct beam as a byproduct
#####

"""
    sw_source_2str!(ncol::I, nlay::I,
                    mo::MeshOrientation{I},
                    Rdir::Array{FT,2},
                    Tdir::Array{FT,2},
                    Tnoscat::Array{FT,2},
                    sfc_albedo::Array{FT,1},
                    source_up::Array{FT,2},
                    source_dn::Array{FT,2},
                    source_sfc::Array{FT,1},
                    flux_dn_dir::AbstractArray{FT,2}) where {I<:Int,FT<:AbstractFloat}

 - `mo` mesh orientation, see [`MeshOrientation`](@ref)

integer,                           intent(in   ) :: ncol, nlay
real(FT), dimension(ncol, nlay  ), intent(in   ) :: Rdir, Tdir, Tnoscat # Layer reflectance, transmittance for diffuse radiation
real(FT), dimension(ncol        ), intent(in   ) :: sfc_albedo          # surface albedo for direct radiation
real(FT), dimension(ncol, nlay  ), intent(  out) :: source_dn, source_up
real(FT), dimension(ncol        ), intent(  out) :: source_sfc          # Source function for upward radation at surface
real(FT), dimension(ncol, nlay+1), intent(inout) :: flux_dn_dir # Direct beam flux
                                                                # intent(inout) because top layer includes incident flux
"""
function sw_source_2str!(ncol::I, nlay::I,
                         mo::MeshOrientation{I},
                         Rdir::Array{FT,2},
                         Tdir::Array{FT,2},
                         Tnoscat::Array{FT,2},
                         sfc_albedo::Array{FT,1},
                         source_up::Array{FT,2},
                         source_dn::Array{FT,2},
                         source_sfc::Array{FT,1},
                         flux_dn_dir::AbstractArray{FT,2}) where {I<:Int,FT<:AbstractFloat}

  b = binary(mo)
  i_lev_sfc = ilev_bot(mo)

  @inbounds for ilev in lay_range(mo)
    source_up[:,ilev]     .=    Rdir[:,ilev] .* flux_dn_dir[:,ilev+1-b]
    source_dn[:,ilev]     .=    Tdir[:,ilev] .* flux_dn_dir[:,ilev+1-b]
    flux_dn_dir[:,ilev+b] .= Tnoscat[:,ilev] .* flux_dn_dir[:,ilev+1-b]
  end
  source_sfc .= flux_dn_dir[:,i_lev_sfc] .* sfc_albedo
  return nothing
end

#####
##### Transport of diffuse radiation through a vertically layered atmosphere.
#####   Equations are after Shonk and Hogan 2008, doi:10.1175/2007JCLI1940.1 (SH08)
#####   This routine is shared by longwave and shortwave
#####

"""
    adding!(ncol::I, nlay::I,
            mo::MeshOrientation{I},
            albedo_sfc::Array{FT,1},
            rdif::Array{FT,2},
            tdif::Array{FT,2},
            src_dn::Array{FT,2},
            src_up::Array{FT,2},
            src_sfc::Array{FT,1},
            flux_up::AbstractArray{FT,2},
            flux_dn::AbstractArray{FT,2}) where {I<:Int,FT<:AbstractFloat}

 - `mo` mesh orientation, see [`MeshOrientation`](@ref)

integer,                          intent(in   ) :: ncol, nlay
real(FT), dimension(ncol       ), intent(in   ) :: albedo_sfc
real(FT), dimension(ncol,nlay  ), intent(in   ) :: rdif, tdif
real(FT), dimension(ncol,nlay  ), intent(in   ) :: src_dn, src_up
real(FT), dimension(ncol       ), intent(in   ) :: src_sfc
real(FT), dimension(ncol,nlay+1), intent(  out) :: flux_up
# intent(inout) because top layer includes incident flux
real(FT), dimension(ncol,nlay+1), intent(inout) :: flux_dn
# ------------------
integer :: ilev
real(FT), dimension(ncol,nlay+1)  :: albedo,   # reflectivity to diffuse radiation below this level
                                                # α in SH08
                                     src        # source of diffuse upwelling radiation from emission or
                                                # scattering of direct beam
                                                # G in SH08
real(FT), dimension(ncol,nlay  )  :: denom      # β in SH08
# ------------------
"""
function adding!(ncol::I, nlay::I,
                 mo::MeshOrientation{I},
                 albedo_sfc::Array{FT,1},
                 rdif::Array{FT,2},
                 tdif::Array{FT,2},
                 src_dn::Array{FT,2},
                 src_up::Array{FT,2},
                 src_sfc::Array{FT,1},
                 flux_up::AbstractArray{FT,2},
                 flux_dn::AbstractArray{FT,2}) where {I<:Int,FT<:AbstractFloat}
  albedo = Array{FT}(undef, ncol,nlay+1)
  src    = Array{FT}(undef, ncol,nlay+1)
  denom  = Array{FT}(undef, ncol,nlay)

  b = binary(mo)
  n̂ = nhat(mo)
  i_lev_sfc = ilev_bot(mo)
  i_lev_top = ilev_top(mo)

  # Indexing into arrays for upward and downward propagation depends on the vertical
  #   orientation of the arrays (whether the domain top is at the first or last index)
  # We write the loops out explicitly so compilers will have no trouble optimizing them.

  ilev = i_lev_sfc
  # Albedo of lowest level is the surface albedo...
  albedo[:,ilev] .= albedo_sfc
  # ... and source of diffuse radiation is surface emission
  src[:,ilev] .= src_sfc

  # From bottom to top of atmosphere --
  #   compute albedo and source of upward radiation
  @inbounds for ilev in lay_range_reversed(mo)
    denom[:, ilev] .= FT(1) ./ (FT(1) .- rdif[:,ilev].*albedo[:,ilev+b]) # Eq 10
    albedo[:,ilev+1-b] .= rdif[:,ilev] .+
                          tdif[:,ilev] .*
                          tdif[:,ilev] .*
                          albedo[:,ilev+b] .*
                          denom[:,ilev] # Equation 9

    # Equation 11 -- source is emitted upward radiation at top of layer plus
    #   radiation emitted at bottom of layer,
    #   transmitted through the layer and reflected from layers below (tdiff*src*albedo)
    src[:,ilev+1-b] .= src_up[:, ilev] .+
                     tdif[:,ilev] .*
                     denom[:,ilev] .*
                     (src[:,ilev+b] .+
                      albedo[:,ilev+b] .*
                      src_dn[:,ilev])
  end

  # Eq 12, at the top of the domain upwelling diffuse is due to ...
  ilev = i_lev_top
  flux_up[:,ilev] .= flux_dn[:,ilev] .* albedo[:,ilev] .+  # ... reflection of incident diffuse and
                    src[:,ilev]                            # emission from below/scattering by the direct beam below
  # From the top of the atmosphere downward -- compute fluxes
  @inbounds for ilev in lev_range(mo)
    flux_dn[:,ilev] .= (tdif[:,ilev-b].*flux_dn[:,ilev-n̂] +  # Equation 13
                       rdif[:,ilev-b].*src[:,ilev] +
                       src_dn[:,ilev-b]) .* denom[:,ilev-b]
    flux_up[:,ilev] .= flux_dn[:,ilev] .* albedo[:,ilev] .+  # Equation 12
                      src[:,ilev]
  end

  return nothing
end

#####
##### Upper boundary condition
#####

"""
    apply_BC!(flux_dn::Array{FT},
              ilay::I) where {I<:Integer,FT<:AbstractFloat}

 - `ilay` apply BC at the i-th layer
 - `flux_dn` Flux to be used as input to solvers below
"""
function apply_BC!(flux_dn::Array{FT,3},
                   ilay::I) where {I<:Integer,FT<:AbstractFloat}
  flux_dn[:,ilay, :] .= FT(0)
  return nothing
end

"""
    apply_BC!(flux_dn::Array{FT,3},
              ilay::I,
              inc_flux::Array{FT,2}) where {I<:Integer,B<:Bool,FT<:AbstractFloat}

 - `flux_dn` Flux to be used as input to solvers below
 - `ilay` apply BC at the i-th layer
 - `inc_flux` Flux at top of domain
"""
function apply_BC!(flux_dn::Array{FT,3},
                   ilay::I,
                   inc_flux::Array{FT,2}) where {I<:Integer,FT<:AbstractFloat}
  fill!(flux_dn, 0)
  flux_dn[:, ilay, :] .= inc_flux
  return nothing
end

"""
    apply_BC!(flux_dn::Array{FT},
              ilay::I,
              inc_flux::Array{FT,2},
              factor::Array{FT,1}) where {I<:Integer,B<:Bool,FT<:AbstractFloat}

 - `flux_dn` Flux to be used as input to solvers below
 - `ilay` apply BC at the i-th layer
 - `inc_flux` Flux at top of domain
 - `factor` Factor to multiply incoming flux
"""
function apply_BC!(flux_dn::Array{FT,3},
                   ilay::I,
                   inc_flux::Array{FT,2},
                   factor::Array{FT,1}) where {I<:Integer,FT<:AbstractFloat}
  fill!(flux_dn, 0)
  flux_dn[:, ilay, :]  .= inc_flux .* spread(factor, 2, size(inc_flux,2))
  return nothing
end