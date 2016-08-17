#!/bin/bash

rotation.json (ignoring this because it had to manually fixed because it was killed in mid json dump)
for json in 4layers_couple.json 6layers_couple.json 18_layers_couple.json flip.json no_augmentation.json flip_rot.json 5layers_couple.json; do
    scp dresdnerg@hal.cbio.mskcc.org:/cbio/ski/fuchs/home/dresdnerg/projects/tissue-microarray-resnet/notebooks/results/$json /tmp
done
