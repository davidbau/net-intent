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

all: solve

$(VENV) env: env.sh
	sh env.sh
	touch $(VENV)

$(MNIST): $(VENV)
	mkdir -p $(FUEL_DATA_PATH) \
		&& $(FUEL_DOWNLOAD) mnist -d $(FUEL_DATA_PATH) \
		&& $(FUEL_CONVERT)  mnist -d $(FUEL_DATA_PATH) -o $(FUEL_DATA_PATH) \
		&& $(FUEL_DOWNLOAD) mnist -d $(FUEL_DATA_PATH) --clear

solve: $(VENV) $(MNIST) Makefile
	$(PYTHON) intent/run.py \
      --num-epochs=20

clean:
	rm -f env
