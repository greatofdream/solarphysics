.PHONY:
all: parameters.pdf

parameters.pdf:
	python3 absorptionLength.py -o $@
energyloss.pdf:
	python3 EnergyLoss.py -o $@
cherenkov.pdf:
	python3 cherenkovYield.py -o $@
.SECONDARY:
.DELETE_ON_ERROR: