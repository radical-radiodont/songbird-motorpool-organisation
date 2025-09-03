# muscle fibres in motion
https://github.com/user-attachments/assets/9b730ac2-9f13-4cc7-bd71-3983aad6a8a7

https://github.com/user-attachments/assets/e1f51ec5-40fd-4881-8878-115ac33fe1e3

These clips show a Ca2-imaging scan of the MDS muscle of the syrinx being actively stimulated _ex vivo_ as described in Adam _et al._ 2021 (https://doi.org/10.1016/j.cub.2021.05.008). The muscle is subjected to stimulation pulses through an array of electrodes connected to the nerve innervating it. Over time, the pulses increase in strength, recruiting gradually more and larger motor units. As fibres contract and fill with calcium, they light up, which is what is seen here.






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
