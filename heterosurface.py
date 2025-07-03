import drjit as dr
from drjit.auto import Float, UInt, Complex2f

from drjit import lgamma
def gamma(v):
  return dr.exp(lgamma(v))

import mitsuba as mi
import numpy as np

from dataclasses import dataclass
@dataclass
class Material:
  mu: list
  sigma: list
  rho: list
  T: list
  Tc: list

  def __iter__(self):
    return iter((self.mu, self.sigma, self.rho, self.T, self.Tc))

## drjit does not have python binding for complex spectrum
class CSpectrum:

  def __init__(self, a, b = 0):
    if isinstance(a, Complex2f):
      self.real = mi.Spectrum(a.real)
      self.imag = mi.Spectrum(a.imag)
    elif isinstance(a, CSpectrum):
      self.real = mi.Spectrum(a.real)
      self.imag = mi.Spectrum(a.imag)
    else:
      self.real = mi.Spectrum(a)
      self.imag = mi.Spectrum(b)
  
  def __add__(self, b):
    return CSpectrum(self.real + b.real, self.imag + b.imag)
  
  def __sub__(self, b):
    return CSpectrum(self.real - b.real, self.imag - b.imag)
  
  def __mul__(self, b):
    return CSpectrum(
      self.real * b.real - self.imag * b.imag,
      self.real * b.imag + self.imag * b.real)
  
  def __truediv__(self, b):
    denom = b.real**2 + b.imag**2
    real = (self.real * b.real + self.imag * b.imag) / denom
    imag = (self.imag * b.real - self.real * b.imag) / denom
    return CSpectrum(real, imag)

  def __rmul__(self, a): ## does not work
    return self * a
  
  def __rsub__(self, a): ## does not work
    return CSpectrum(a.real - self.real, a.imag - self.imag)

  def __pow__(self, expn):
    if expn < 0:
      inv = CSpectrum(1, 0) / self
      return inv ** -expn
    result = CSpectrum(self)
    for _ in range(expn-1):
      result *= self
    return result
  
  def __neg__(self):
    return CSpectrum(-self.real, -self.imag)
  
  def conj(a):
    return CSpectrum(a.real, -a.imag)

  def sqrt(a):
    r = dr.safe_sqrt(a.real**2+a.imag**2)
    return CSpectrum(dr.safe_sqrt((r+a.real)/2), dr.sign(a.imag)*dr.safe_sqrt((r-a.real)/2))

  def inv(a):
    return CSpectrum(1) / a

  def expj(expn): ## exp(1j*expn)
    return CSpectrum(dr.cos(expn), dr.sin(expn))
  
  def select(mask, a, b):
    return CSpectrum(
      dr.select(mask, a.real, b.real),
      dr.select(mask, a.imag, b.imag)
    )

def Fresnel(mu_r, cosi):
  cost = dr.safe_sqrt(1 - (1-cosi**2) / mu_r**2)
  fs = (cosi - mu_r * cost) / (cosi + mu_r * cost)
  return fs

def Fresnel_1(mu_r, wi, p):
  a, b, c = wi
  delta = CSpectrum.sqrt(mu_r**2 - 1 + c**2)
  
  if p == 0: # s polarized
    mu_f = -(delta - c) / (delta + c)
    denom = delta * (delta + c)**2
    inv_denom = CSpectrum.inv(denom)
    dfdr = inv_denom * -2 * (mu_r * c)
    dfdu = inv_denom * -2 * (mu_r**2 - 1) * a
    dfdv = inv_denom * -2 * (mu_r**2 - 1) * b
  else: # p polarized
    mu_f = -(delta - mu_r**2 * c) / (delta + mu_r**2 * c)
    inv_delta = CSpectrum.inv(delta)
    inv_r2cdelta = CSpectrum.inv(delta + mu_r**2*c)
    r2cdelta_inv2 = (-delta+mu_r**2*c) *inv_r2cdelta**2
    dfdr = (-(inv_delta+c*2) * r2cdelta_inv2 + (-inv_delta+c*2) * inv_r2cdelta) * mu_r
    dfdu = (inv_delta*(a*c) + mu_r**2*a)*r2cdelta_inv2 + (inv_delta*(a*c) - mu_r**2*a)*inv_r2cdelta
    dfdv = (inv_delta*(b*c) + mu_r**2*b)*r2cdelta_inv2 + (inv_delta*(b*c) - mu_r**2*b)*inv_r2cdelta
  
  return mu_f, dfdr, dfdu, dfdv

def Fresnel_2(mu_r, wi, p):
  a, b, c = wi
  delta2 = mu_r**2 - 1 + c**2
  delta = CSpectrum.sqrt(delta2) #if type(mu_r) is CSpectrum else dr.sqrt(delta2)
  if p == 0:
    d2fdu2 = ((-delta+c)*(delta+c)/delta2 * 4*a**2 + (-delta+c)*((-mu_r**2 -a**2 + 1)*delta2 + a**2*c**2)/delta2**2*delta + (delta+c)*((-mu_r**2 -a**2 + 1)*delta2 + a**2*c**2)/delta2**2*delta) / ((delta+c)**2)
    d2fdv2 = ((-delta+c)*(delta+c)/delta2 * 4*b**2 + (-delta+c)*((-mu_r**2 -b**2 + 1)*delta2 + b**2*c**2)/delta2**2*delta + (delta+c)*((-mu_r**2 -b**2 + 1)*delta2 + b**2*c**2)/delta2**2*delta) / ((delta+c)**2)
    d2fdr2 = ((-delta+c)*(delta+c)/delta2**2*delta * mu_r**2 + (-delta+c)/delta2 * 2*mu_r**2 + (delta+c)**2/delta2**2*delta * mu_r**2 + (delta+c)/delta2 * 2*mu_r**2 + (-delta+c)*(delta+c)/delta2*delta - (delta+c)**2/delta2*delta) / ((delta+c)**3)
    d2fduv = ((-delta+c)/delta2**2*delta * (-mu_r**2+1) + (delta+c)/delta2**2*delta * (-mu_r**2+1) + (-delta+c)*(delta+c)/delta2 * 4) * a*b/ ((delta+c)**2)
    d2fdur = ((-delta+c)/delta2**2*delta * -c - (delta+c)/delta2**2*delta * c - delta2**(-1) * 2*c- (-delta+c)/delta2 *2) * (mu_r*a) / ((delta+c)**2)
    d2fdvr = ((-delta+c)/delta2**2*delta * -c - (delta+c)/delta2**2*delta * c - delta2**(-1) * 2*c- (-delta+c)/delta2 *2) * (mu_r*b) / ((delta+c)**2)
  else:
    r2c_delta = -delta + mu_r**2*c
    inv_delta = CSpectrum.inv(delta)
    inv_r2cdelta = CSpectrum.inv(delta+mu_r**2*c)
    cinvr2 = inv_delta*c + mu_r**2
    cinv_r2 = inv_delta*c - mu_r**2
    a1 = inv_delta**3*(a**2*c**2) - inv_delta*(mu_r**2+a**2-1)
    b1 = inv_delta**3*(b**2*c**2) - inv_delta*(mu_r**2+b**2-1)
    t0 = (r2c_delta * inv_r2cdelta + 1) * inv_r2cdelta
    t1 = (r2c_delta*cinvr2*inv_r2cdelta*2 + cinv_r2) * cinvr2 * inv_r2cdelta**2
    d2fdu2 = t0 * a1 + t1 * a**2
    d2fdv2 = t0 * b1 + t1 * b**2
    d2fdr2 = (2*(inv_delta+c*2)**2*r2c_delta*inv_r2cdelta**3 + 2*(inv_delta**2-c**2*4)*inv_r2cdelta**2 ) * mu_r**2 + r2c_delta*(inv_delta**3*mu_r**2 - c*2 - inv_delta)*inv_r2cdelta**2 + (inv_delta**3*mu_r**2 + c*2 - inv_delta)*inv_r2cdelta
    d2fduv = (r2c_delta*(inv_delta**2*c**2-1)*inv_delta*inv_r2cdelta**2 + r2c_delta*cinv_r2**2*inv_r2cdelta**3*2 + (inv_delta**2*c**2-1)*inv_delta*inv_r2cdelta + (inv_delta**2*c**2 - mu_r**4)*inv_r2cdelta**2*2 )*(a*b)
    tr = (((((-delta+mu_r**2*c)*inv_r2cdelta*2 - 1) * -(inv_delta + c*2) + (-inv_delta + c*2)) * cinv_r2 + (-delta+mu_r**2*c)*(inv_delta**3*-c + 2))*inv_r2cdelta**2 + (inv_delta**3*-c - 2)*inv_r2cdelta ) * mu_r
    d2fdur, d2fdvr = tr * a, tr * b

  return d2fdu2, d2fdv2, d2fdr2, d2fduv, d2fdur, d2fdvr

def csqr(x):
  return x.real**2 + x.imag**2

def csqr2(re, im):
  return re**2 + im**2

def safe_div(x, y):
  return x/y

def rand_pdf(s, x, y):
  return dr.exp(-(x**2 + y**2) / (2*s**2)) / (dr.two_pi*s**2)

def rand(s, u2):
  r = dr.sqrt(-2*dr.log(1-u2.x)) * s
  phi = dr.two_pi * u2.y
  x = r * dr.cos(phi)
  y = r * dr.sin(phi)
  return x, y, rand_pdf(s, x, y)

## confluent hypergeometric function
def hyp1f1(a, b, z, n_=4):
  s, term = 1, 1
  for n in range(1, n_):
    term *= (a+n-1)/((b+n-1)*n) * z
    s += term
  return dr.select(a == 0, 1, s)

## generate random variables from normal distribution
def gen_rvs(M, W=1024, sigma=1):
  shape = (M, W, W, 1)
  rvs = np.random.normal(0, sigma, shape)
  return rvs

## convert rvs to mitsuba textures
def gen_texure(rvs, D):
  textures = [mi.load_dict({
    'type': 'bitmap',
    'data': rvs_m,
    'filter_type': 'nearest',
    'raw': True,
    'to_uv': mi.ScalarTransform4f().scale(D)
  }) for rvs_m in rvs]
  return textures


THRS = 0

class HeteroSurface(mi.BSDF):
  
  def __init__(self, props):
    mi.BSDF.__init__(self, props)
    
    def get_props(key, default_val):
      return props[key] if props.has_property(key) else default_val
    
    self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
    self.m_components = [self.m_flags]
    
    ## optionally enable geometry occlusion to avoid grazing angle artifacts
    self.geometry_occlusion = get_props('geometry_occlusion', False)
    
    ## default material
    self.set_materials([
      Material(
        mu    = [0, 1.5, 3.9],
        sigma = [2e-7, 1e-18, 1e-18],
        rho   = [0, 0, 0],
        T     = [2e-6, 2e-7, 2e-7],
        Tc    = [2e-7, 2e-7, 0],
      ),
    ])

    ## mixture parameters (raw)
    self.set_density(get_props('density', 0.5))
    self.set_radius(get_props('radius', 2e-7))
    self.linear_blend = get_props('linear_blend', False)

    ## prepare speckle estimators on init
    self.S = get_props('S', 4) # expansion orders for smooth surface
    self.M = get_props('M', 4) # Monte-Carlo count for the decomposition of Iw
    self.D = get_props('D', 8) # physical scaling factor
    
    ## incident beam paramters (assumed constant)
    self.beam_width = get_props('beam_width', 6e-6)

    ## debug only
    self.render_field = False
    
    ## render speckle patterns
    ## currently it only works well with direct illumination (depth=2)
    ## depth > 2 will lead to artifacts due to the sampling strategy
    ## if set to False, the BSDF will render mean statistics
    self.render_speckle = get_props('render_speckle', True)

    ## expand Fresnel terms to higher orders (at most 2 in our code)
    ## if set to False, it reduces to standard GHS theory
    self.expand_fresnel = get_props('expand_fresnel', True)

    if self.render_speckle:
      self.gen_rvs(self.D)
    
    self.parameters_changed()
    # print('Heterogeneous BSDF initialized')
  
  ## mitsuba3 does not parse list or custom types
  ## the material parameters are set externally
  def set_materials(self, materials, beck_coeff=1):
    self.materials = materials
    self.is_rough = []
    self.distrs = []

    ## load IOR data in mitsuba
    def get_ior_spectrum(name):
      prefix = '../mitsuba3/resources/data/ior/' + name
      eta = mi.load_dict({
        'type': 'spectrum',
        'filename': prefix + '.eta.spd',
      })
      k = mi.load_dict({
        'type': 'spectrum',
        'filename': prefix + '.k.spd',
      })
      return eta, k

    ## handle material parameters for each process
    for material in self.materials:
      mu, sigma, rho, T, Tc = material
      
      ## format 1: mu = [mu_h, 'metal_0', 'metal_1']
      ## only used for height-correlated (conductor) materials
      if len(mu) == 3 and type(mu[1]) is str and type(mu[2]) is str:
        eta_0, k_0 = get_ior_spectrum(mu[1])
        eta_1, k_1 = get_ior_spectrum(mu[2])
        mu[1], mu[2] = eta_0, k_0
        mu += [eta_1, k_1]
      ## format 2: mu = [mu_h, 'metal']
      ## conductor
      elif type(mu[1]) is str:
        eta, k = get_ior_spectrum(mu[1])
        mu[1] = eta
        mu.append(k)
      ## format 3: mu = [mu_h, mu_r_re, mu_r_im]
      ## dielectric, mu_r_im is ususally 0

      self.is_rough.append(
        (dr.two_pi / 3.6e-7 * sigma[0])**2 > 5
      )
      ## use Beckmann distributions to sample rough surfaces
      ## distrs[0] is also used for (optionally-enabled) geometry occlusion
      ## therefore beck_coeff modifies final appearance
      self.distrs.append(
        mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, dr.sqrt(2)*sigma[0]/T[0]*beck_coeff, False)
      )

  ## set (textured) probability density of material 1
  ## called in __init__ constructor
  def set_density(self, density):
    if type(density) is str:
      self.density = mi.load_dict({
        'type': 'bitmap',
        'filename': density,
        'wrap_mode': 'mirror',
        'raw': True,
      })
    else:
      self.density = density
  
  ## set radius of material 1
  ## called in __init__ constructor
  def set_radius(self, radius):
    area = dr.pi * radius**2
    self.c0 = 3 / area


  ## generate random variables (rvs) in uv space to sample speckle patterns
  ## each texel corresponds to a surface patch of size beam_width;
  ## each set of rvs corresponds to a realization of the random surface patch;
  ## sample rays intersecting the surface at close distances (≲ beam_width)
  ## will map to the same rvs texel (with nearest filtering), to form speckles;
  ## increasing D scales that distance in uv space, therefore mapping close rays
  ## to different surface patches, which asymptotes to the mean statistics;
  ## D can either be viewed as the distance to a fix-sized surface,
  ## or as the physical size of the surface ([0,1] => D * W * bw)
  def gen_rvs(self, D, W=1024):
    s = self.beam_width/2
    self.px_rvs = gen_rvs(self.M, W, s)
    self.py_rvs = gen_rvs(self.M, W, s)

    P = 2 # len(self.materials)
    ## rvs layers:
    ## P (number of processes) * 2 (individual terms) * self.S (expansion orders) * self.M (Monte-Carlo count)
    ## + 1 (cross term) * self.M (Monte-Carlo count)
    M = P*2 * self.S * self.M + self.M
    self.zr_rvs = gen_rvs(M, W, dr.inv_sqrt_two)
    self.zi_rvs = gen_rvs(M, W, dr.inv_sqrt_two)
    
    self.update_scale(D)

  ## update the physical scale of the surface
  def update_scale(self, d):
    if not hasattr(self, 'zr_rvs'):
      self.gen_rvs(d)
    else:
      self.px = gen_texure(self.px_rvs, d)
      self.py = gen_texure(self.py_rvs, d)
      self.zr = gen_texure(self.zr_rvs, d)
      self.zi = gen_texure(self.zi_rvs, d)
  
  def parameters_changed(self, keys = None):
    self.irra = dr.pi * self.beam_width**2 / 2
    if self.render_speckle and not hasattr(self, 'zr_rvs'):
      self.gen_rvs(self.D)


  ## get p0 and p1 at surface interaction point
  def get_probs(self, si, active):
    if type(self.density) in [float, int]:
      return 1-self.density, self.density
    else: # textured density
      p1 = self.density.eval_1(si, active)/1.6
      p1 = dr.clamp(p1, 0, 1)
      return 1 - p1, p1

  ## get incident (emission) and outgoing (camera) directions at surface interaction point
  def get_dirs(self, ctx, si, wo_, active):
    ## handle transport direction
    forward = ctx.mode == mi.TransportMode.Importance
    wi = dr.select(forward, si.wi, wo_)
    wo = dr.select(forward, wo_, si.wi)

    ## drjit does not support merged boolean ops...
    active &= wi.z > THRS
    active &= wo.z > THRS
    ## term 1/wo.z in brdf leads to high reflectance at grazing exitance
    ## physically it is correct if we neglect geometry occlusion
    ## for research purpose we use clipping to avoid artifact

    wm = wi + wo
    return wi, wo, wm, active

  ## get wavevector and beam width (assumed constant)
  def get_wavevars(self, k, wm):
    kx, ky, kz = k * wm.x, k * wm.y, k * wm.z
    v2 = kx**2 + ky**2
    
    bw = self.beam_width
    return kx, ky, kz, v2, bw

  ## get material parameters of the indexed process at the surface interaction point
  def get_material(self, si, active, idx):
    mu, sigma, rho, T, Tc = self.materials[idx]
    ## format 1: height-correlated material
    if len(mu) == 5:
      mu_r0_re = mu[1].eval(si, active)
      mu_r0_im = mu[2].eval(si, active)
      mu_r1_re = mu[3].eval(si, active)
      mu_r1_im = mu[4].eval(si, active)
      mu_r_re = (mu_r0_re + mu_r1_re) / 2
      mu_r_im = (mu_r0_im + mu_r1_im) / 2
      mu_r = CSpectrum(mu_r_re, -mu_r_im)
      sigma[1] = dr.abs(mu_r0_re - mu_r1_re) / 6
      sigma[2] = dr.abs(mu_r0_im - mu_r1_im) / 6
    ## format 3: dielectric
    elif type(mu[1]) in [float, int]:
      mu_r = Complex2f(mu[1], -mu[2])
    ## format 2: conductor
    else:
      mu_r_re = mu[1].eval(si, active)
      mu_r_im = mu[2].eval(si, active)
      mu_r = CSpectrum(mu_r_re, -mu_r_im)
    return mu, mu_r, sigma, rho, T, Tc


  ## ********** PART 1. Sampling and PDF **********

  ## get sampling exponent for smooth surface
  def get_s(self, bw, k, wi, sigma, T):
    n = 0 ## TODO: select an exponent order n
    s = dr.sqrt(1/bw**2 + 2*n/T**2) / k[0]
    return s

  ## sample BSDF
  def sample(self, ctx, si, sample1, sample2, active):
    
    active &= si.wi.z > THRS
    
    bs = mi.BSDFSample3f()
    bs.eta = 1.0

    bs.sampled_component = 0
    bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection) #?

    ## use a simple sampling strategy for speckle patterns
    ## regardless of material configs
    if self.render_speckle:
      bs.wo = mi.warp.square_to_uniform_hemisphere(sample2)
      bs.pdf = mi.warp.square_to_uniform_hemisphere_pdf(bs.wo)
      val = self.eval_brdf_speckle(ctx, si, bs.wo, active) / bs.pdf
    
    ## single-process
    elif len(self.materials) == 1:
      val, bs.wo, bs.pdf = self.sample_process(ctx, si, active, 0, sample2)
    
    ## mixed-process, via mixture sampling
    else:
      p0, p1 = self.get_probs(si, active)
      val0, wo0, pdf0 = self.sample_process(ctx, si, active, 0, sample2)
      val1, wo1, pdf1 = self.sample_process(ctx, si, active, 1, sample2)

      ## debug: linear blend, for comparison only
      if self.linear_blend:
        select_0 = sample1 < p0
        val = dr.select(select_0, val0, val1)
        bs.wo = dr.select(select_0, wo0, wo1)
        bs.pdf = dr.select(select_0, pdf0, pdf1)
      
      else:
        val01, wo01, pdf01 = self.sample_cross_process(ctx, si, active, sample2)
        select_0 = sample1 < p0**2
        select_1 = p0**2 <= sample1
        select_1 &= sample1 < p0**2+p1**2
        val = dr.select(select_0, val0, dr.select(select_1, val1, val01))
        bs.wo = dr.select(select_0, wo0, dr.select(select_1, wo1, wo01))
        bs.pdf = dr.select(select_0, pdf0, dr.select(select_1, pdf1, pdf01))

    ## val = self.eval(ctx, si, bs.wo, active) / bs.pdf # slow convergence
    
    if self.geometry_occlusion and self.is_rough[0]:
      val *= self.distrs[0].G(si.wi, bs.wo, dr.normalize(si.wi+bs.wo))
    
    return (bs, val)

  ## sample the indexed process (individual terms in BSDF)
  def sample_process(self, ctx, si, active, idx, sample2):
    ### for bsdf mean, sample 0th Fresnel order \int_J0()e^(-mt^2)dt
    mu, sigma, rho, T, Tc = self.materials[idx]
    bw = self.beam_width

    wvls = si.wavelengths * 1e-9
    k = dr.two_pi / wvls

    ## rough surface
    if self.is_rough[idx]:
      m, pdf = self.distrs[idx].sample(si.wi, sample2)
      wo_ = mi.reflect(si.wi, m)
      pdf /= 4 * dr.dot(wo_, m)
    
    ## smooth surface
    else:
      s = self.get_s(bw, k, si.wi, sigma[0], T[0])

      wx, wy, pdf = rand(s, sample2)
      wo_ = mi.Vector3f(wx-si.wi.x, wy-si.wi.y, 0)

      ro2 = wo_.x**2 + wo_.y**2
      active &= ro2 < 1

      wo_.z = dr.safe_sqrt(1 - ro2)
      pdf *= wo_.z

    wi, wo, wm, active = self.get_dirs(ctx, si, wo_, active)
    u2 = self.eval_u2(si, active, k, wi, wm, idx)

    brdf_mu = u2 * wm.z**2 / (4 * wvls**2 * si.wi.z)
    val = brdf_mu / pdf

    if self.is_rough[idx]:
      val /= m.z

    return val & active, wo_, pdf

  ## sample the cross-process (cross terms in BSDF)
  def sample_cross_process(self, ctx, si, active, sample2):

    wvls = si.wavelengths * 1e-9
    k = dr.two_pi / wvls

    ## rough surface
    if self.is_rough[0] and self.is_rough[1]:
      distr = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, 1, False)
      m, pdf = distr.sample(si.wi, sample2)
      wo_ = mi.reflect(si.wi, m)
      pdf /= 4 * dr.dot(wo_, m)
    
    ## smooth surface
    else:
      mu, sigma, rho, T, Tc = self.materials[0]
      bw = self.beam_width
      
      s = self.get_s(bw, k, si.wi, sigma[0], T[0])

      wx, wy, pdf = rand(s, sample2)
      wo_ = mi.Vector3f(wx-si.wi.x, wy-si.wi.y, 0)

      ro2 = wo_.x**2 + wo_.y**2
      active &= ro2 < 1

      wo_.z = dr.safe_sqrt(1 - ro2)
      pdf *= wo_.z

    wi, wo, wm, active = self.get_dirs(ctx, si, wo_, active)

    u2  = self.eval_u2(si, active, k, wi, wm, 0, self.c0) / 2
    u2 += self.eval_u2(si, active, k, wi, wm, 1, self.c0) / 2
    u2 += self.eval_u2_cross(si, active, k, wi, wm, self.c0)

    brdf_mu_cross = u2 * wm.z**2 / (4 * wvls**2 * si.wi.z)
    val = brdf_mu_cross / pdf

    return val & active, wo_, pdf


  def pdf(self, ctx, si, wo, active):
    
    active &= si.wi.z > THRS
    active &= wo.z > THRS

    ## use a simple sampling strategy for speckle patterns
    ## regardless of material configs
    if self.render_speckle:
      pdf = mi.warp.square_to_uniform_hemisphere_pdf(wo)
    
    ## single-process
    elif len(self.materials) == 1:
      pdf = self.pdf_process(si, wo, 0)
    
    ## mixed-process, via mixture sampling
    else:
      p0, p1 = self.get_probs(si, active)
      pdf0 = self.pdf_process(si, wo, 0)
      pdf1 = self.pdf_process(si, wo, 1)

      ## debug: linear blend, for comparison only
      if self.linear_blend:
        pdf = p0 * pdf0 + p1 * pdf1
      else:
        pdf01 = self.pdf_cross_process(si, wo)
        pdf = p0**2 * pdf0 + p1**2 * pdf1 + 2*p0*p1 * pdf01

    return pdf & active
  
  def pdf_process(self, si, wo, idx):
    bw = self.beam_width
    k = dr.two_pi / si.wavelengths * 1e9

    mu, sigma, rho, T, Tc = self.materials[idx]

    ## rough surface
    if self.is_rough[idx]:
      m = dr.normalize(si.wi + wo)
      pdf = self.distrs[idx].pdf(si.wi, m) / (4 * dr.dot(wo, m))
    
    ## smooth surface
    else:
      s = self.get_s(bw, k, si.wi, sigma[0], T[0])
      w = si.wi + wo
      pdf = rand_pdf(s, w.x, w.y) * wo.z

    return pdf

  def pdf_cross_process(self, si, wo):
    ## rough surface
    if self.is_rough[0] and self.is_rough[1]:
      distr = mi.MicrofacetDistribution(mi.MicrofacetType.Beckmann, 1, False)
      m = dr.normalize(si.wi+wo)
      pdf = distr.pdf(si.wi, m) / (4 * dr.dot(wo, m))
    
    ## smooth surface
    else:
      mu, sigma, rho, T, Tc = self.materials[0]
      bw = self.beam_width
      k = dr.two_pi / si.wavelengths * 1e9
      s = self.get_s(bw, k, si.wi, sigma[0], T[0])
      w = si.wi + wo
      pdf = rand_pdf(s, w.x, w.y) * wo.z
    
    return pdf


  ## eval BSDF
  def eval(self, ctx, si, wo_, active = True):
    if self.render_speckle == True:
      val = self.eval_brdf_speckle(ctx, si, wo_, active)
    else:
      val = self.eval_brdf_mean(ctx, si, wo_, active)

    if self.geometry_occlusion and self.is_rough[0]:
      val *= self.distrs[0].G(si.wi, wo_, dr.normalize(si.wi+wo_))
    
    return val


  ## ********** PART 2. Evaluate Mean **********

  ## eval BSDF by <uu*>
  def eval_brdf_mean(self, ctx, si, wo_, active = True):

    wvls = si.wavelengths * 1e-9
    k = dr.two_pi / wvls

    wi, wo, wm, active = self.get_dirs(ctx, si, wo_, active)

    if len(self.materials) == 1:
      u2 = self.eval_u2(si, active, k, wi, wm, 0)
    else:
      p0, p1 = self.get_probs(si, active)

      ## debug: linear blend, for comparison only
      if self.linear_blend:
        u2 = p0 * self.eval_u2(si, active, k, wi, wm, 0) + p1 * self.eval_u2(si, active, k, wi, wm, 1)
      else:
        u2 = p0*p0 * self.eval_u2(si, active, k, wi, wm, 0)
        u2 += p0*p1 * self.eval_u2(si, active, k, wi, wm, 0, self.c0) # cross term

        u2 += 2*p0*p1 * self.eval_u2_cross(si, active, k, wi, wm, self.c0)  # cross term
        
        u2 += p1*p1 * self.eval_u2(si, active, k, wi, wm, 1)
        u2 += p1*p0 * self.eval_u2(si, active, k, wi, wm, 1, self.c0)  # cross term
    
    brdf_mu = u2 * wm.z**2 / (4 * wvls**2 * si.wi.z)
    return brdf_mu & active

  ## eval <uu*> / irradiance
  def eval_u2(self, si, active, k, wi, wm, idx, c0=0):
    u2_0 = self.eval_u2_p(si, active, k, wi, wm, 0, idx, c0)
    u2_1 = self.eval_u2_p(si, active, k, wi, wm, 1, idx, c0)
    return (u2_0 + u2_1) / 2
  
  ## eval <uu*> / irradiance of polarization p
  def eval_u2_p(self, si, active, k, wi, wm, p, idx, c0=0):
    
    mu, mu_r, sigma, rho, T, Tc = self.get_material(si, active, idx)
    mu_f, dfdr, dfdu, dfdv = Fresnel_1(mu_r, wi, p)

    kx, ky, kz, v2, bw = self.get_wavevars(k, wm)
    g = (kz * sigma[0])**2

    g1 = g/T[0]**2 + 1/(2*bw**2) + c0
    g0 = dr.arange(Float, 0, self.S)/T[0]**2 + 1/(2*bw**2) + c0
    
    def J(m=0, n=0, k=0, hyp=False):
      
      ### rough surface
      m_ = m + g1
      v2_4m = v2/(4*m_)
      J1 = (v2_4m**n)/(gamma(2*n+1)*2*m_**(k+1)) * dr.exp(-v2_4m)
      c = hyp1f1(n-k, 1+n*2, v2_4m)*gamma(k+n+1)
      if hyp == True:
        mT2 = m_ * T[0]**2
        c += gamma(k+n+3)*g/(2*mT2**2) * hyp1f1(n-k-2, 1+n*2, v2_4m)
        c -= gamma(k+n+4)*g/(6*mT2**3) * hyp1f1(n-k-3, 1+n*2, v2_4m)
      J1 *= c
      
      ### smooth surface
      J0 = dr.zeros(mi.Spectrum)
      gs = 1
      for s, g0s in enumerate(g0): ## does drjit unroll for loops?
        m_ = m + g0s
        v2_4m = v2/(4*m_)
        gs *= g/s if s>0 else 1
        J0 += gs*(v2_4m**n)/(2*m_**(k+1))*dr.exp(-v2_4m)*hyp1f1(n-k,1+n*2,v2_4m)
      J0 *= dr.exp(-g) * gamma(k+n+1)/gamma(2*n+1)
      
      return dr.select(self.is_rough[idx], J1, J0)

    ## halfvec approximation by Strogyn
    # cosi = dr.dot(dr.normalize(wm), wi)
    # mu_f = Fresnel(mu_r, cosi, p)

    ### 0th order
    u2 = 0
    u2 = J(hyp=False) * csqr(mu_f)

    ### for smooth surfaces or cross term, fresnel expansion 
    ### has perceptually neglectable contribution in practice

    ### 1st order
    if self.expand_fresnel and self.is_rough[idx] and c0 == 0:
      vs = 2*(sigma[0]/T[0])**2
      vss = 4*(sigma[0]/T[0])**2/T[0]**2
      ## convert nan to zero if k_{x,y}=(0,0)
      ## kd2,kp2 are not well-defined at (0,0) but J_2n(kt)=0 for n>0,k=0
      kd2 = safe_div(kx**2-ky**2, v2)
      kp2 = safe_div(kx*ky, v2)
      kvs = kz*vs
      k2vs2 = kvs**2
      cp = safe_div(kx, dr.sqrt(v2))
      sp = safe_div(ky, dr.sqrt(v2))
      c3p = 4*cp**3-3*cp
      s3p = -4*sp**3+3*sp

      ## r1, r2*
      ## note sigma[1,2] correspond to rho[0,1] and Tc[0,1]
      f0fr = dr.conj(mu_f)*dfdr
      u2 += 2*dr.sqrt(g) * (f0fr.real*sigma[2]*rho[1] + f0fr.imag*sigma[1]*rho[0]) * J()
      u2 -= 2*dr.sqrt(g) * (f0fr.real*sigma[2]*rho[1] * J(1/Tc[1]**2) + f0fr.imag*sigma[1]*rho[0] * J(1/Tc[0]**2))

      ## u1, v1, u2*, v2*
      f0fu = (dr.conj(mu_f) * dfdu).real
      f0fv = (dr.conj(mu_f) * dfdv).real
      u2 -= 2*kvs * (f0fu*cp + f0fv*sp) * J(1/T[0]**2, n=1/2, k=1/2)

      ## r1r2*
      u2 += csqr(dfdr) * (sigma[1]**2 * J(1/T[1]**2) +
                          sigma[2]**2 * J(1/T[2]**2))
      u2 += csqr(dfdr) * g * (
        sigma[1]**2 * rho[0]**2 * (J(2/Tc[0]**2) - 2*J(1/Tc[0]**2) + J()) + \
        sigma[2]**2 * rho[1]**2 * (J(2/Tc[1]**2) - 2*J(1/Tc[1]**2) + J()))
      
      ## u1u2*, v1v2*, u1v2*, v1u2*
      fx, fy = csqr(dfdu), csqr(dfdv)
      u2 += (fx+fy) * vs * J(1/T[0]**2)

      u2 -= (fx+fy)/2 * vss * J(1/T[0]**2,k=1)
      u2 += (fx-fy)/2 * vss * kd2 * J(1/T[0]**2,n=1,k=1)

      u2 -= (fx+fy)/2 * k2vs2 * J(2/T[0]**2,k=1)
      u2 += (fx-fy)/2 * k2vs2 * kd2 * J(2/T[0]**2,n=1,k=1)

      u2 += (dfdu * dr.conj(dfdv)).real*2 * vss * kp2 * J(1/T[0]**2,n=1,k=1)
      u2 += (dfdu * dr.conj(dfdv)).real*2 * k2vs2 * kp2 * J(2/T[0]**2,n=1,k=1)

      ## r1u2*, r1v2*, u1r2*, v1r2*
      frfu = dfdr * dr.conj(dfdu)
      frfv = dfdr * dr.conj(dfdv)
      u2 += (frfu.real*cp+frfv.real*sp) * 2*kvs*dr.sqrt(g)*sigma[2]*rho[1] * (J(1/T[0]**2+1/Tc[1]**2,n=1/2,k=1/2)-J(1/T[0]**2,n=1/2,k=1/2))
      u2 += (frfu.imag*cp+frfv.imag*sp) * 2*kvs*dr.sqrt(g)*sigma[1]*rho[0] * (J(1/T[0]**2+1/Tc[0]**2,n=1/2,k=1/2)-J(1/T[0]**2,n=1/2,k=1/2))

    ### 2nd order
    if self.expand_fresnel and self.is_rough[idx] and c0 == 0:
      d2fdu2, d2fdv2, d2fdr2, d2fduv, d2fdur, d2fdvr = Fresnel_2(mu_r, wi, p)
      
      ## r1r1, r2*r2*
      f0frr = dr.conj(mu_f) * d2fdr2
      u2 += f0frr.real * (sigma[1]**2-sigma[2]**2) * J()

      u2 -= g * f0frr.real * (
        (sigma[1]*rho[0])**2 * (J(2/Tc[0]**2) - 2*J(1/Tc[0]**2) + J()) - \
        (sigma[2]*rho[1])**2 * (J(2/Tc[1]**2) - 2*J(1/Tc[1]**2) + J()))

      u2 += g * f0frr.imag * (sigma[1]*sigma[2]*rho[0]*rho[1]*(
        J(1/Tc[0]**2+1/Tc[1]**2) - J(1/Tc[0]**2) - J(1/Tc[1]**2) + J()))

      ## u1u1, v1v1, u2*u2*, v2*v2*
      f0fuu = (dr.conj(mu_f) * d2fdu2).real
      f0fvv = (dr.conj(mu_f) * d2fdv2).real
      u2 += (f0fuu+f0fvv) * vs * J()
      u2 -= (f0fuu+f0fvv)/2 * k2vs2 * J(2/T[0]**2,k=1)
      u2 += (f0fuu-f0fvv)/2 * k2vs2 * kd2 * J(2/T[0]**2,n=1,k=1)

      ## r1u1, r2*u2*, r1v1, r2*v2*
      f0fur = dr.conj(mu_f) * d2fdur
      f0fvr = dr.conj(mu_f) * d2fdvr
      u2 += (f0fur.real*cp+f0fvr.real*sp) * 2*kvs*dr.sqrt(g)*sigma[2]*rho[1] * (J(1/T[0]**2+1/Tc[1]**2,n=1/2,k=1/2)-J(1/T[0]**2,n=1/2,k=1/2))
      u2 += (f0fur.imag*cp+f0fvr.imag*sp) * 2*kvs*dr.sqrt(g)*sigma[1]*rho[0] * (J(1/T[0]**2+1/Tc[0]**2,n=1/2,k=1/2)-J(1/T[0]**2,n=1/2,k=1/2))

      ## other fresnel expansion terms have negligible contribution
    
    u2 *= dr.two_pi
    return dr.select(u2 > 0, u2, 0)

  ## <fe^(ikh)>, second-order approximation
  def f_exp_ikh(self, si, active, kz, wi, p, idx):
  
    mu, mu_r, sigma, rho, T, Tc = self.get_material(si, active, idx)   
    mu_f, dfdr, dfdu, dfdv = Fresnel_1(mu_r, wi, p)
    
    ## 0th order
    f = CSpectrum(mu_f)

    ## 1st/2nd order (additional terms)
    if self.expand_fresnel:
      d2fdu2, d2fdv2, d2fdr2, d2fduv, d2fdur, d2fdvr = Fresnel_2(mu_r, wi, p)
      Rhr = sigma[0] * (rho[0]*sigma[1] + 1j*rho[1]*sigma[2])
      Rrr = sigma[1]**2 - sigma[2]**2
      kzRhr = CSpectrum(kz) * Rhr
      f -= 1j*kzRhr * dfdr
      f += (Rrr - kzRhr**2) * d2fdr2/2
      f += 2*(sigma[0]/T[0])**2 * (d2fdu2+d2fdv2)/2

    return f * CSpectrum.expj(-kz*mu[0]) * dr.exp(-(kz*sigma[0])**2/2)

  ## eval <u1u2*> / irradiance (cross terms)
  def eval_u2_cross(self, si, active, k, wi, wm, c0):
    u2_0 = self.eval_u2_p_cross(si, active, k, wi, wm, 0, c0)
    u2_1 = self.eval_u2_p_cross(si, active, k, wi, wm, 1, c0)
    return (u2_0 + u2_1) / 2

  ## eval <u1u2*> / irradiance (cross terms) of polarization p
  def eval_u2_p_cross(self, si, active, k, wi, wm, p, c0):
    kx, ky, kz, v2, bw = self.get_wavevars(k, wm)
    def J(m):
      v2_4m = v2/(4*m)
      J0 = dr.exp(-v2_4m)/(2*m)
      return J0
    
    J0 = J(1/(2*bw**2)) - J(1/(2*bw**2)+c0)
    ## TODO: approximation of p01 gives high reflectivity 
    ## when kz=0 and v2=0 (both wi,wo at grazing angle)
    ## temporal fix: remove contribution of first term
    # J0 = 0 - J(1/(2*bw**2)+c0)
    
    ## (f1 e^-ikh1) and (f2* e^ikh2) are independent
    f1 = self.f_exp_ikh(si, active, kz, wi, p, 0)
    f2 = self.f_exp_ikh(si, active, kz, wi, p, 1)
    u2 = J0 * (f1 * f2.conj()).real
    
    u2 *= 2*np.pi
    return u2


  ## *********** PART 3. Evaluate speckle **********

  ## eval BSDF by uu*
  def eval_brdf_speckle(self, ctx, si, wo_, active = True):

    wi, wo, wm, active = self.get_dirs(ctx, si, wo_, active)

    wvls = si.wavelengths * 1e-9
    k = dr.two_pi / wvls

    vw = self.get_vw(si, active, k*wm.x, k*wm.y)
    
    ## mean u + fluctuation e
    u2 = [0, 0] # two polarizations
    for p in [0, 1]:

      ## single process
      if len(self.materials) == 1:
        u = self.eval_u_mean(si, active, k, wi, wm, p, 0)
        e = self.eval_u_fluc(si, active, k, wi, wm, p, 0, vw)
      
      ## mixed process
      else:
        p0, p1 = self.get_probs(si, active)
        p01 = dr.sqrt(p0*p1)
        
        u  = self.eval_u_mean(si, active, k, wi, wm, p, 0) * p0
        u += self.eval_u_mean(si, active, k, wi, wm, p, 1) * p1
        
        e  = self.eval_u_fluc(si, active, k, wi, wm, p, 0, vw) * p0
        e += self.eval_u_fluc(si, active, k, wi, wm, p, 0, vw, self.c0) * p01

        ## cross term
        e += self.eval_u_cross_fluc(si, active, k, wi, wm, p, vw, self.c0) * p01

        e += self.eval_u_fluc(si, active, k, wi, wm, p, 1, vw) * p1
        e += self.eval_u_fluc(si, active, k, wi, wm, p, 1, vw, self.c0) * p01
      
      if self.render_field == True:
        return e.real / (wvls * dr.sqrt(self.irra))
      
      u2[p] = csqr(u+e)
    
    u2 = (u2[0] + u2[1]) / 2
    
    brdf = u2 * wm.z**2 / (4 * wvls**2 * self.irra * si.wi.z)
    return brdf & active
    
  ## eval <u> (speckle mean field)
  def eval_u_mean(self, si, active, k, wi, wm, p, idx):
    kx, ky, kz, v2, bw = self.get_wavevars(k, wm)
    u = dr.pi*bw**2 * self.f_exp_ikh(si, active, kz, wi, p, idx) * dr.exp(-v2*bw**2/4)
    return u

  ## sample speckle fluctuation from autocovariance
  def eval_u_fluc(self, si, active, k, wi, wm, p, idx, vw, c0=0):
    mu, mu_r, sigma, rho, T, Tc = self.get_material(si, active, idx)
    mu_f, dfdr, dfdu, dfdv = Fresnel_1(mu_r, wi, p)

    kx, ky, kz, v2, bw = self.get_wavevars(k, wm)
    g = (kz * sigma[0])**2

    ofst = idx*2 + (0 if c0==0 else 1)

    ## rough surface
    if self.is_rough[idx]:
      vw_z = CSpectrum(0, 0)
      m1 = g/T[0]**2 + 1/(2*bw**2) + c0
      vx = dr.exp(-v2/(8*m1)) * dr.rsqrt(m1)
      
      for m in range(self.M):
        id = ofst*self.S*self.M + m
        vw_z += vw[m] * self.get_z(si, active, id)

      f = CSpectrum(mu_f) * CSpectrum.expj(-kz*mu[0])
      e = dr.pi*bw * dr.rsqrt(2*self.M) * f * vw_z * vx
    
    ## smooth surface
    else:
      vw_vx_z = CSpectrum(0, 0)
      
      for s in range(1, self.S):
        ms = s/T[0]**2 + 1/(2*bw**2) + c0
        for m in range(self.M):
          id = ofst*self.S*self.M + s*self.M + m
          vxs = (kz*sigma[0])**s * dr.exp(-v2/(8*ms)) * dr.rsqrt(ms*gamma(s+1))
          vw_vx_z += vw[m] * vxs * self.get_z(si, active, id)
      
      f = CSpectrum(mu_f) * CSpectrum.expj(-kz*mu[0]) * dr.exp(-g/2)
      e = dr.pi*bw * dr.rsqrt(2*self.M) * f * vw_vx_z
    
    return e
  
  ## sample speckle fluctuation from cross-covariance
  def eval_u_cross_fluc(self, si, active, k, wi, wm, p, vw, c0):
    
    kx, ky, kz, v2, bw = self.get_wavevars(k, wm)

    f1 = self.f_exp_ikh(si, active, kz, wi, p, 0)
    f2 = self.f_exp_ikh(si, active, kz, wi, p, 1)

    vwz = CSpectrum(0, 0)
    m1 = 1/(2*bw**2) + c0
    vx = dr.exp(-v2/(8*m1)) * dr.rsqrt(m1)

    for m in range(self.M):
      id = 4*self.S*self.M + m
      vwz += vw[m] * self.get_z(si, active, id)

    ec = dr.pi*bw * dr.rsqrt(2*self.M) * (f1-f2) * vwz * vx
    return ec
  
  ## get complex rv to sample Ic
  def get_z(self, si, active, id):
    zr = self.zr[id].eval(si, active)
    zi = self.zi[id].eval(si, active)
    return CSpectrum(zr, zi)

  ## prepare Monte Carlo estimators of Iw
  def get_vw(self, si, active, kx, ky):
    vw = [CSpectrum(0, 0)] * self.M
    
    # zp = dr.sum(dr.exp(-1j*(kx*self.px+ky*self.py)) * self.zm)
    # TODO: use 1-bit {-1,+1} distribution to reduce texture fetch
    for m in range(self.M):
      px = self.px[m].eval(si, active)
      py = self.py[m].eval(si, active)
      k_dot_p = kx * px + ky * py
      vw[m] = CSpectrum.expj(-k_dot_p)
    
    return vw

  
mi.register_bsdf('heterosurface', lambda props: HeteroSurface(props))