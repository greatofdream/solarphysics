.PHONY: all sim toymc
PlayG4:=/SK/SolarNeutrino/PlayG4-build/PlayG4
Es:=$(shell seq 4 20)
all: $(Es:%=simulation/%_summary.h5)
thetas:=$(shell seq 0 10 180)
theory: Theory/merge.pdf
# theta_E.h5
sim: $(Es:%=simulation/%MeV.h5)
toymc: ToyMC/gaus.pdf ToyMC/expGaus_cos.pdf
# simulation
simulation/%MeV.mac: electron.mac
	mkdir -p $(@D)
	sed -e 's/@Energy@/$*/' $^ > $@
simulation/%MeV.root: simulation/%MeV.mac $(PlayG4)
	cd $(dir $(PlayG4)) && $(PlayG4) $(realpath $<) $(shell pwd)/$@
simulation/%MeV.h5: simulation/%MeV.root
	python3 /SK/SolarNeutrino/PlayG4/test/converter.py -i $^ -o $@
simulation/%_summary.h5: simulation/%MeV.h5
	PYTHONPATH=.. python3 electronDirectionPreview.py -i $^ -o $@ --energy $*

# old simulation, I have forgotten the origin of those hdf5 in MC_direction
angleRMS.pdf:
	python3 MultipleAngle.py -o $@
theta_E.h5: $(Es:%=MC_direction/%MeV.h5)
	python3 electronTheta.py -i $^ -o $@

# Theory calculation
Theory/%.h5:
	mkdir -p $(@D)
	python3 electronDirectionTheory.py --angle $* -o $@
Theory/merge.pdf: $(thetas:%=Theory/%.h5)
	python3 electronDirectionTheoryMerge.py -i $^ -o $@

ToyMC/%.pdf:
	mkdir -p $(@D)
	python3 CalCherenkovDir.py -i $* -o $@

.SECONDARY:
.DELETE_ON_ERROR:
