import pyexr
import numpy as np

import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_spectral')
from mitsuba import ScalarTransform4f as T

import heterosurface
from heterosurface import Material

cbsdf = mi.load_dict({
  'type': 'heterosurface',
  'M': 4,   # samples for speckle rendering, see BSDF init for other parameters
})
cbsdf.set_materials([
  Material(
    mu    = [0, 1.5, 3.9],        # mean: mu_h, mu_r_re, mu_r_im
    sigma = [2e-7, 1e-18, 1e-18], # stddev: sigma_h, sigma_r_re, sigma_r_im
    rho   = [0, 0, 0],            # correlation coeff: rho_hr (h and r_re), rho_hj (h and r_im), 0
    T     = [6e-7, 2e-7, 2e-7],   # correlation distance: T_h, T_r_re, T_r_im
    Tc    = [2e-7, 2e-7, 0],      # cross-correlation distance: T_hr (h and r_re), T_hj (h and r_im), 0
  ),
])
cbsdf.beam_width = 6e-6           # beam width
cbsdf.parameters_changed()

scene_dict = {
  'type': 'scene',
  'integrator': {
    'type': 'path',
    'max_depth': 2
  },
  'light': mi.load_dict({
    'type': 'point',
    'position': [0, 0, 1],
    'intensity': {
      'type': 'spectrum',
      'value': 1e1
    }
  }),
  'scatterer': {
    'type': 'rectangle',
    'bsdf': cbsdf
  },
  'sensor': {
    'type': 'perspective',
    'to_world': T().look_at(
      origin=[0, 0, 1], target=[0, 0, 0], up=[0, 1, 0]
    ),
    'fov': 90,
    'film': {
      'type': 'hdrfilm',
      'width': 1024,
      'height': 1024,
    },
    'sampler': {
      'type': 'stratified',
    },
  },
}


def render(scene, spp=1024, filename=None, prompt='Rendering'):
  
  timer.begin_stage(prompt)
  img = mi.render(scene, spp=spp)
  timer.end_stage('')

  if filename is not None:
    pyexr.write(f'{filename}.exr', np.array(img))
  
  return img


if __name__ == '__main__':

  timer = mi.Timer()
  spp = 1024
  
  scene = mi.load_dict(scene_dict)

  ## 1. render mean intensity
  cbsdf.render_speckle = False
  render(scene, spp, 'example_mean', 'Rendering mean instensity')

  ## 2. render speckles from near to far
  dists = [1, 2, 4, 8, 32]
  cbsdf.render_speckle = True

  for d in dists:
    cbsdf.update_scale(d)
    render(scene, spp, f'example_speckle_{d}', f'Rendering speckle at distance {d}')