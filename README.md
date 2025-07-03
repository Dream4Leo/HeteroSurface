# A Fully-statistical Wave Scattering Model for Heterogeneous Surfaces
This repository contains the implementation for the paper "A Fully-statistical Wave Scattering Model for Heterogeneous Surfaces", to be presented in SIGGRAPH 2025 (Journal Track).

The implementation is presented as a BSDF plugin of [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3), and comes with an [example script](example.py).

## Notes and Prerequisites

The BSDF python plugin is developed with Mitsuba 3.6.0 and tested in `cuda_spectral` variant. Additional packages include:

- numpy
- pyexr (to export renderings in the example script)

## Usage

```shell
python example.py
```

The script renders the mean appearance and speckle patterns of a surface with the implemented BSDF, and corresponds to Figure 11 of the paper.