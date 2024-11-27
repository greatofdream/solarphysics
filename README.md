# Introduction
Conceptual research and theory analysis

## Detector
### dat file
+ `SK/QETable.dat`: QE used in SK, comes from `SKG4/dat_file/PMT/QETable.dat`
+ `WDSim.py`: water parameter used in HyperK simulation 
+ `Jinping/Water.xml`: water parameter used in Jinping
+ `refraction.csv`: water parameters from website, `make refraction.csv`

### scripts
+ `cherenkovYield.py`: refraction index vs frequency; energy loss; estimate the cherenkov yield for water.
+ `EnergyLoss.py`: energy loss. (it is same as `CherenkovYield.py`)
+ `Preview.py`: compare absorption length parameters in simulation of different experiments

## MultipeScatter
+ `electronDirectionPreview.py`: check the electron and photon direction in the MC
+ `electronDirectionMerge.py`: merge the electron and photon direction in the MC
+ `electronDirectionTheory.py`: compare the electron and photon direction transfer probability between the MC and theory
+ `electronDirectionTheoryMerge.py`: merge the output from `electronDirectionTheory.py`
+ `CalCherenkovDir.py`: calculate the photon direction distribution

> Following script need to be validated

+ `MultipleAngle.py`: calculate the theoretical RMS of scattering angle.
+ `electronTheta.py`: comparing the theoretical and MC RMS of scattering angle.

## PhotonScatter
The photon scattering effect in the different shape of detector.
+ `ToyMC.py`: simulate the photon scattering effect for Tub or Sphere and isotropic or Cerenkov photons

## SSM
