.PHONY: all
all: Tub/0_0_1600.h5 Sphere/0_0_1600.h5 Tub/0_0_-1600.h5 Sphere/0_0_-1600.h5
Tub/%.h5:
	mkdir -p $(@D)
	PYTHONPATH=../MultipleScatter python3 ToyMC.py -o $@ --vertex $(subst _, ,$*) --geo Tub
Sphere/%.h5:
	mkdir -p $(@D)
	PYTHONPATH=../MultipleScatter python3 ToyMC.py -o $@ --vertex $(subst _, ,$*) --geo Sphere
