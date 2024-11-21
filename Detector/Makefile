.PHONY:
all: output/parameters.pdf
refraction.csv:
	wget https://refractiveindex.info/tmp/data/main/H2O/Hale.csv -O refraction.csv
output/parameters.pdf:
	mkdir -p $(@D)
	python3 Preview.py -o $@
output/energyloss.pdf:
	mkdir -p $(@D)
	python3 EnergyLoss.py -o $@
output/cherenkov.pdf:
	mkdir -p $(@D)
	python3 cherenkovYield.py -o $@
output/angleRMS.pdf:
	mkdir -p $(@D)
	python3 MultipleAngle.py -o $@
.SECONDARY:
.DELETE_ON_ERROR:
