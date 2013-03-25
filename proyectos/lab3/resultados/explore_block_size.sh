#!/bin/bash

PROG="../cuda5/headless.out"
N=1024
MAX_BLOCK_SIZE=32

# Body
BX=1
while ((BX<=MAX_BLOCK_SIZE)); do
	BY=1
	while ((BY<=MAX_BLOCK_SIZE)); do
		echo "BX=$BX BY=$BY"
		$PROG $N $BX $BY && cut -d "," -f 1;
		((BY+=1))
		echo " "
	done
	((BX+=1))
done
