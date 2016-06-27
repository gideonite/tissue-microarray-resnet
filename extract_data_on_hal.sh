#!/bin/bash
cp /cbio/ski/fuchs/home/dresdnerg/data/CedersSinai_PCa.zip $TMPDIR
pushd .
cd $TMPDIR
unzip CedersSinai_PCa.zip
mkdir -p models
popd
