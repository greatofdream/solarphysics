.PHONY: all
Zs:=$(shell seq -1600 800 1600)
all: $(foreach z,$(Zs),Tub/0_1_0/0_0_$z.h5 Tub/0_0_-1/0_0_$z.h5 Sphere/0_1_0/0_0_$z.h5 Sphere/0_0_-1/0_0_$z.h5)
Tub/0_0_-1/%.h5:
	mkdir -p $(@D)
	PYTHONPATH=../MultipleScatter python3 ToyMC.py -o $@ --vertex $(subst _, ,$*) --geo Tub
Tub/0_1_0/%.h5:
	mkdir -p $(@D)
	PYTHONPATH=../MultipleScatter python3 ToyMC.py -o $@ --vertex $(subst _, ,$*) --dir 0 1 0 --geo Tub
Sphere/0_0_-1/%.h5:
	mkdir -p $(@D)
	PYTHONPATH=../MultipleScatter python3 ToyMC.py -o $@ --vertex $(subst _, ,$*) --geo Sphere
Sphere/0_1_0/%.h5:
	mkdir -p $(@D)
	PYTHONPATH=../MultipleScatter python3 ToyMC.py -o $@ --vertex $(subst _, ,$*) --dir 0 1 0 --geo Sphere
