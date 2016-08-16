VENV = env/.built
PYTHON = env/bin/python3
FUEL_DOWNLOAD = env/bin/fuel-download
FUEL_CONVERT = env/bin/fuel-convert

# Read or setup default for FUEL_DATA_PATH
FUELRC = $(HOME)/.fuelrc
ifeq ($(wildcard $(FUELRC)),) 
	export FUEL_DATA_PATH ?= fuel
else
	FUEL_DATA_PATH ?= $(shell \
		sed -n 's/data_path: *\("\?\)\(.*\)\1/\2/p' \
		< $(FUELRC))
endif

MNIST = $(FUEL_DATA_PATH)/mnist.hdf5
CIFAR10 = $(FUEL_DATA_PATH)/cifar10.hdf5

all: solve

cifar10: $(CIFAR10)

$(VENV) env: env.sh
	sh env.sh
	touch $(VENV)

$(MNIST): $(VENV)
	mkdir -p $(FUEL_DATA_PATH) \
		&& $(FUEL_DOWNLOAD) mnist -d $(FUEL_DATA_PATH) \
		&& $(FUEL_CONVERT)  mnist -d $(FUEL_DATA_PATH) -o $(FUEL_DATA_PATH) \
		&& $(FUEL_DOWNLOAD) mnist -d $(FUEL_DATA_PATH) --clear

$(CIFAR10): $(VENV)
	mkdir -p $(FUEL_DATA_PATH) \
		&& $(FUEL_DOWNLOAD) cifar10 -d $(FUEL_DATA_PATH) \
		&& $(FUEL_CONVERT)  cifar10 -d $(FUEL_DATA_PATH) -o $(FUEL_DATA_PATH) \
		&& $(FUEL_DOWNLOAD) cifar10 -d $(FUEL_DATA_PATH) --clear

movie: $(VENV) $(MNIST) Makefile histograms.pkl
	$(PYTHON) intent/pic.py --num-epochs=10 --unit-order=histograms.pkl
	sh makemovies.sh

movie2: $(VENV) $(MNIST) Makefile histograms.pkl
	$(PYTHON) intent/comp.py --num-epochs=10 --unit-order=histograms.pkl
	sh makemovies.sh

crun: $(VENV) $(CIFAR10) Makefile
	$(PYTHON) intent/crun.py --num-epochs=25

histograms.pkl: $(VENV) $(MNIST) Makefile
	$(PYTHON) intent/run.py --num-epochs=25

solve: histograms.pkl

clean:
	rm -f env
