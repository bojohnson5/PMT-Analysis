#!/bin/zsh

for min in 140 145 150 155 160 165; do
	for max in 170 175 180 185 190; do
		root -l -q -b "view_spectrum.cpp(\"12.root\", $min, $max, true)"
		root -l -q -b "fit_spectrum.cpp(\"12spectrum.root\", 500, 1000)"
		PV=$(python peak_to_valley.py 12spectrum)
		printf "%s %s %s\n" $min $max $PV >> out.txt
	done
done
