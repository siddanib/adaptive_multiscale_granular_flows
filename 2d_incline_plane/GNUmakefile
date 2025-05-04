AMREX_HOME = /path/to/amrex
AMREX_HYDRO_HOME = /path/to/AMReX-Hydro
TOP = /path/to/incflo

EBASE = incflo

DIM = 2

PRECISION  = DOUBLE

TINY_PROFILE = FALSE

DEBUG      = FALSE

COMP       = gnu

USE_MPI    = TRUE
USE_AMREX_MPMD = TRUE

USE_OMP    = FALSE
USE_CUDA   = TRUE

USE_HYPRE = FALSE
BL_NO_FORT = TRUE

USE_EB = FALSE

USE_CSG = FALSE

include $(TOP)/src/Make.incflo

ifeq ($(USE_AMREX_MPMD), TRUE)
  DEFINES += -DUSE_AMREX_MPMD
endif

