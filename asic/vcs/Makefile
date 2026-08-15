# VCS functional simulation of qcore_top (co-sim) — P10/M9 ASIC flow.
#
# Compiles the Verilator-frozen RTL (rtl/, read-only) plus the VCS testbench
# (tb_qcore_top.sv) into `simv`.  The testbench reproduces rtl/tb/sim_main.cpp
# at the binary file-format level, so VCS and Verilator outputs diff bit-exactly.
#
# The RTL uses a few forward-referenced `logic` declarations (Verilator accepts,
# VCS O-2018.09 rejects); gen_rtl.py snapshots rtl/ into asic/vcs/gen/ with a
# semantically-neutral declaration hoist (zero functional change).
#
# The compiled simv MUST be run with:
#   +vcs+initreg+0 +vcs+initmem+0
# to reproduce Verilator's 2-state (all-zero) power-on state; the co-sim
# preload only emits non-zero SRAM runs.  asic/vcs/run_vcs_tests.py does this.
VCS      ?= /home/public/app/synopsys/vcs-mx/O-2018.09-SP2/bin/vcs
SNPS_NS  := /home/public/app/synopsys/compat/bin/snps-centos7
GEN_DIR  := gen
TB       := tb_qcore_top.sv
SIMV     := simv

SRC := $(GEN_DIR)/qcore_top.sv $(TB)

.PHONY: all clean

all: $(SIMV)

# Rebuild the snapshot if the generator or any rtl/ source is newer.
$(GEN_DIR)/qcore_top.sv: gen_rtl.py ../../rtl/*.sv
	python3 gen_rtl.py

# Compile under the CentOS 7 compatibility namespace: VCS O-2018.09's prebuilt
# objects reference the legacy glibc symbol __pthread_unwind@GLIBC_PRIVATE
# (removed in glibc >= 2.34), so the compile AND the resulting simv must both
# run inside the compat rootfs (glibc 2.17).
$(SIMV): $(SRC) Makefile
	$(SNPS_NS) $(VCS) -full64 -sverilog +v2k \
	  -timescale=1ns/1ps \
	  +vcs+initreg+random \
	  +incdir+$(GEN_DIR) \
	  -top tb_qcore_top \
	  -o $(SIMV) \
	  $(SRC) \
	  -l compile.log
	@echo "VCS compile done -> $(SIMV)"

clean:
	rm -rf $(SIMV) $(SIMV).daidir csrc ucli.key compile.log *.vpd *.fsdb *.log $(GEN_DIR)
