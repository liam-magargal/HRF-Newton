

######################################
Descriptions of files:
######################################


######
Burgers' equation

Burgers_getECSWweights.m : Get sample indices and weights for ECSW-G and ECSW-LSPG schemes for Burgers' equation. Note: this should be executed before any other Burgers' solution scripts, as sample indices/weights are used in the other scripts.

Burgers_allTestParams.m : Burgers' equation FOM/ROM solutions for parameter sweep for all test parameter sets

Burgers_allTrainParams.m : Burgers' equation FOM/ROM solutions for parameter sweep for all train parameter sets

Burgers_getAvgTime.m : Burgers' equation for getting the average time to run the FOMs/ROMs

Burgers_main.m : Burgers' equation for FOM/ROM solutions to a single user-defined test parameter set

#######
Heat equation with cubic reaction term

heatDiff_getECSWweights.m : Get sample indices and weights for ECSW-G and ECSW-LSPG for heat equation with cubic reaction term. Note: this script should be executed before any other heat equation with cubic reaction term scripts, as sample indices/weights are used in the other scripts. 

heatDiff_getavgTime.m : Heat equation with cubic reaction term for getting average time to run the FOM/ROMs

heatDiff_main.m : Heat equation with cubic reaction term for FOM/ROM solutions to a single user-defined test parameter set