#####
##### Broadband by-band
#####

export FluxesByBand

"""
    FluxesByBand{FT} <: AbstractFluxes{FT}

Contains both broadband and by-band fluxes

# Fields
$(DocStringExtensions.FIELDS)
"""
mutable struct FluxesByBand{FT} <: AbstractFluxes{FT}
  fluxes_broadband
  "upward flux"
  bnd_flux_up#::Array{FT,3}
  "downward flux"
  bnd_flux_dn#::Array{FT,3}
  "net flux"
  bnd_flux_net#::Array{FT,3}
  "downward direct flux"
  bnd_flux_dn_dir#::Array{FT,3}
end

"""
    reduce!(this::FluxesByBand,
                       gpt_flux_up::Array{FT,3},
                       gpt_flux_dn::Array{FT,3},
                       spectral_disc::AbstractOpticalProps,
                       top_at_1::Bool,
                       gpt_flux_dn_dir::Union{Nothing,Array{FT,3}}=nothing)

Reduces fluxes by-band to broadband in `FluxesByBand` `this`, given

 - `gpt_flux_up` fluxes by gpoint [W/m2]
 - `gpt_flux_dn` fluxes by gpoint [W/m2]
 - `spectral_disc` a `AbstractOpticalProps` struct containing spectral information
 - `top_at_1` bool indicating at top
and, optionally,
 - `gpt_flux_dn_dir` direct flux downward
"""
function reduce!(this::FluxesByBand,
                       gpt_flux_up::Array{FT,3},
                       gpt_flux_dn::Array{FT,3},
                       spectral_disc::AbstractOpticalProps,
                       top_at_1::Bool,
                       gpt_flux_dn_dir::Union{Nothing,Array{FT,3}}=nothing) where {FT<:AbstractFloat}
  ncol, nlev = size(gpt_flux_up)
  ngpt = get_ngpt(spectral_disc)
  nbnd = get_nband(spectral_disc)
  band_lims = deepcopy(get_band_lims_gpoint(spectral_disc))

  # Compute broadband fluxes
  #   This also checks that input arrays are consistently sized
  #
  reduce_broadband!(this.fluxes_broadband, gpt_flux_up, gpt_flux_dn, spectral_disc, top_at_1, gpt_flux_dn_dir)

  # Check sizes
  @assert size(gpt_flux_up, 3) == ngpt
  associated(this.bnd_flux_up)     && @assert all(size(this.bnd_flux_up) .== (ncol, nlev, nbnd))
  associated(this.bnd_flux_dn)     && @assert all(size(this.bnd_flux_dn) .== (ncol, nlev, nbnd))
  associated(this.bnd_flux_dn_dir) && @assert all(size(this.bnd_flux_dn_dir) .== (ncol, nlev, nbnd))
  associated(this.bnd_flux_net)    && @assert all(size(this.bnd_flux_net) .== (ncol, nlev, nbnd))
  # Self-consistency -- shouldn't be asking for direct beam flux if it isn't supplied
  @assert !(associated(this.bnd_flux_dn_dir) && !present(gpt_flux_dn_dir))

  # Band-by-band fluxes

  # Up flux
  if associated(this.bnd_flux_up)
    this.bnd_flux_up = sum_byband(ncol, nlev, ngpt, nbnd, band_lims, gpt_flux_up)
  end

  # Down flux
  if associated(this.bnd_flux_dn)
    this.bnd_flux_dn = sum_byband(ncol, nlev, ngpt, nbnd, band_lims, gpt_flux_dn)
  end

  # Direct Down flux
  if associated(this.bnd_flux_dn_dir)
    this.bnd_flux_dn_dir = sum_byband(ncol, nlev, ngpt, nbnd, band_lims, gpt_flux_dn_dir)
  end

  # Net flux
  if(associated(this.bnd_flux_net))
    #
    #  Reuse down and up results if possible
    #
    if(associated(this.bnd_flux_dn) && associated(this.bnd_flux_up))
      this.bnd_flux_net = net_byband(ncol, nlev,       nbnd, this.bnd_flux_dn, this.bnd_flux_up)
    else
      this.bnd_flux_net = net_byband(ncol, nlev, ngpt, nbnd, band_lims, gpt_flux_dn, gpt_flux_up)
    end
  end
end

"""
    are_desired(this::FluxesByBand)

Boolean indicating if any fluxes desired from this
set of g-point fluxes.
"""
are_desired(this::FluxesByBand) =
  any([associated(this.bnd_flux_up),
       associated(this.bnd_flux_dn),
       associated(this.bnd_flux_dn_dir),
       associated(this.bnd_flux_net),
       are_desired(this.fluxes_broadband)])

#####
##### Kernels for computing by-band fluxes by summing
##### over all elements in the spectral dimension.
#####

"""
    sum_byband(ncol, nlev, ngpt, nbnd, band_lims, spectral_flux)

Spectral reduction over all points

integer,                               intent(in ) :: ncol, nlev, ngpt, nbnd
integer,  dimension(2,          nbnd), intent(in ) :: band_lims
real(FT), dimension(ncol, nlev, ngpt), intent(in ) :: spectral_flux
real(FT), dimension(ncol, nlev, nbnd), intent(out) :: byband_flux
"""
function sum_byband(ncol, nlev, ngpt, nbnd, band_lims, spectral_flux::Array{FT}) where FT

  byband_flux = Array{FT}(undef, ncol, nlev, nbnd)
  for ibnd = 1:nbnd
    for ilev = 1:nlev
      for icol = 1:ncol
        byband_flux[icol, ilev, ibnd] =  spectral_flux[icol, ilev, band_lims[1, ibnd]]
        for igpt = band_lims[1,ibnd]+1: band_lims[2,ibnd]
          byband_flux[icol, ilev, ibnd] = byband_flux[icol, ilev, ibnd] + spectral_flux[icol, ilev, igpt]
        end
      end
    end
  end
  return byband_flux
end

"""
    net_byband(ncol, nlev, ngpt, nbnd, band_lims, spectral_flux_dn, spectral_flux_up)

Net flux: Spectral reduction over all points

integer,                               intent(in ) :: ncol, nlev, ngpt, nbnd
integer,  dimension(2,          nbnd), intent(in ) :: band_lims
real(FT), dimension(ncol, nlev, ngpt), intent(in ) :: spectral_flux_dn, spectral_flux_up
real(FT), dimension(ncol, nlev, nbnd), intent(out) :: byband_flux_net
"""
function net_byband(ncol, nlev, ngpt, nbnd, band_lims, spectral_flux_dn::Array{FT}, spectral_flux_up) where FT

  byband_flux_net = Array{FT}(undef, ncol, nlev, nbnd)

  for ibnd = 1:nbnd
    for ilev = 1:nlev
      for icol = 1:ncol
        igpt = band_lims[1,ibnd]
        byband_flux_net[icol, ilev, ibnd] = spectral_flux_dn[icol, ilev, igpt] -
                                            spectral_flux_up[icol, ilev, igpt]
        for igpt = band_lims[1,ibnd]+1:band_lims[2,ibnd]
          byband_flux_net[icol, ilev, ibnd] = byband_flux_net[icol, ilev, ibnd] +
                                              spectral_flux_dn[icol, ilev, igpt] -
                                              spectral_flux_up[icol, ilev, igpt]
        end
      end
    end
  end
  return byband_flux_net
end

"""
    net_byband(ncol, nlev, nbnd, byband_flux_dn, byband_flux_up)

integer,                               intent(in ) :: ncol, nlev, nbnd
real(FT), dimension(ncol, nlev, nbnd), intent(in ) :: byband_flux_dn, byband_flux_up
real(FT), dimension(ncol, nlev, nbnd), intent(out) :: byband_flux_net
"""
net_byband(ncol, nlev, nbnd, byband_flux_dn, byband_flux_up) = byband_flux_dn .- byband_flux_up
