# songbird-motorpool-organisation
Contains the scripts used during my MSc thesis project on the organisation of the motor pool in birdsong motor control.

4 different scripts are included, with some overlap. Note that 2 of them requires data that is not publicly available and thus cannot be run but are for viewing purpose only.

'overview of all specimens and identification of new MUs' is used for producing a near complete overview of all the data per specimen, showing the fibre mask, living fibres, stimulated fibres, identified motor units and their territories, and whether these territories are signficiantly non-random. It also contains the code used for identifying new MUs by manually comparing the muscle fibre activity signals of the stimulated fibres.
This script requires data to run.

'network analysis' is used for running the described network analysis on the data, trying to objectively identify MUs based on activity signals per specimen and electrode. It also has different options for optimisation methods relating to the network analysis.
This script requires data to run.

'motor pool model' is used for modelling a simulated motor pool based on models by Enoka & Fuglevand 2001 and Petersen & Rostalski 2019 and is designed to mimic the actual data used in this study by taking parameters of the motor pool size from Adam et al. 2021. It is also used to test the network analysis against a ground-truthed data set where the MUs are known.
This script can be run as is.

'motor pool distribution' is used for showing the distribution of MUs in the motor pool and compare the size distribution to other studies (as shown in fig. 3E-F).
This script can be run as is.
