.PHONY:
all: parameters.pdf
refraction.csv:
	wget https://refractiveindex.info/tmp/data/main/H2O/Hale.csv -O refraction.csv
parameters.pdf:
	python3 Preview.py -o $@
energyloss.pdf:
	python3 EnergyLoss.py -o $@
cherenkov.pdf:
	python3 cherenkovYield.py -o $@
angleRMS.pdf:
	python3 MultipleAngle.py -o $@
.SECONDARY:
.DELETE_ON_ERROR:
