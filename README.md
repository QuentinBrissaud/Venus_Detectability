# Aerial and Space-borne Seismology on Venus: Viability and Design Implications for Future Missions

## Summary
The following codes allow the computation of venusquake detection probability for airglow and/or balloon missions on Venus. 

## Installation
- conda create -n venus_detectability python=3.10
- conda activate venus_detectability
- pip install -r requirements.txt

## Usage
- Rayleigh wave detection probabilities along trajectory (Figure 3): "notebooks/create_Figure_trajectory_seismic.ipynb"
- Seismic and Airglow final detection probabilities (Figure 4): "notebooks/create_Figure_statistics.ipynb"
- Volcano detection probabilities along balloon trajectories (Figure 5): "notebooks/create_Figure_volcano.ipynb"

## Paper abstract
Venus' evolution remains a mystery because of the lack of in-situ geophysical data to constrain its interior structure. Recently-selected planetary missions VERITAS (NASA), DAVINCI+ (NASA), and EnVision (ESA) will investigate the planet's interior, surface, and atmospheric chemistry. However, none of these missions includes sensors capable of accurately probing Venus' crustal and mantle properties. Seismometer deployments are challenging on Venus due to high surface temperature and pressure. Acoustic balloon measurements and airglow observations -- that monitor Venus' upper atmosphere glow caused by chemical and radiative processes -- have been suggested as alternatives to surface deployments. However, it is critical to assess the potential of such missions under realistic conditions of geology, atmospheric states, network geometry, and seismicity using physics-based modeling. We employ a probabilistic framework to investigate detection probabilities as a function of Signal-to-Noise Ratio (SNR) for airglow and acoustic balloon missions using wave simulations, thermodynamically-consistent seismic velocity models, and realistic seismicity estimates. Our results demonstrate that the probability of detecting a single venusquake at SNR>1 over 6 months is around $65\%$ across an entire 3-balloon network of about 5000km extent. Probabilities using dayglow imager data are below 60% and below 10% using nightglow data. Seismo-volcanic sequences enhance detectability but only if high seismic activity occurs at multiple volcanoes. Long-duration missions with both airglow and balloon-borne sensors could allow seismic wave measurements over a broad range of frequencies. Our results are highly dependent on seismic velocities, attenuation, seismicity, noise levels, mission duration, and airglow-coupling efficiency which should be the focus of future studies.

## Citation
Brissaud, Q., et al. (2026). Aerial and Space-borne Seismology on Venus: Viability and Design Implications for Future Missions. Earth and Space Science
```
@article{brissaud2026aerial,
  title={Aerial and Space-borne Seismology on Venus: Viability and Design Implications for Future Missions},
  author={Brissaud, Quentin et al},
  journal={Earth and Space Science},
  year={2026},
}
```
