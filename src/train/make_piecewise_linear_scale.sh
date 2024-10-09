show_help() {
	cat << EOF
Usage: ${0##*/} [OPTIONS] BURN_IN START_EXP END_EXP EXP_STEPS BURN_OUT
Constructs a string of floating-point numbers, starting with a burn-in of length
BURN_IN, then a ramp-up from 10^START_EXP to 10^END_EXP with EXP_STEPS linearly
spaced steps in the exponent, followed by a burn-out of length BURN_OUT. The
burn-out value is equal to END_EXP. Note that the ramp-up section has EXP_STEPS
steps, starting with 10^START_EXP and ending with 10^END_EXP (both inclusive).
Outputs a comma-delimited list of numbers in a single line, followed by \\n.
	-i|--burn-in-val    If given, use this value as the burn-in value; default 0
	-b|--base           If given, the base of the exponent; default 10
	-s|--scale          The precision of the outputs in digits after the decimal
	                    point (base 10); defaults to 20
EOF
}

POSARGS=""  # Positional arguments
while [ $# -gt 0 ]
do
	case "$1" in
		-h|--help)
			show_help
			exit 0
			;;
		-i|--burn-in-val)
			burninval=$2
			shift 2
			;;
		-b|--base)
			base=$2
			shift 2
			;;
		-s|--scale)
			scale=$2
			shift 2
			;;
		*)
			POSARGS="$POSARGS $1"  # Preserve positional arguments
			shift
	esac
done
eval set -- "$POSARGS"  # Restore positional arguments to expected indices

if [[ -z $5 ]]
then
	show_help
	exit 1
fi

if [[ -z $burninval ]]
then
	burninval=0
fi
if [[ -z $base ]]
then
	base=10
fi
if [[ -z $scale ]]
then
	scale=20
fi

burnin=$1
startexp=$2
endexp=$3
expsteps=$4
burnout=$5

# Create BURN_IN copies of burn-in value
burninresult=$(yes $burninval | head -n $burnin | tr "\n" ",")

# Compute burn-out value and create BURN_OUT copies
burnoutval=$(echo "scale=$scale; e($endexp * l($base))" | bc -l)
burnoutresult=$(yes $burnoutval | head -n $burnout | tr "\n" ",")

# Create linearly interpolated exponent portion
linearresult=''
stepsize=$(echo "scale=$scale; ($endexp - $startexp) / ($expsteps - 1)" | bc -l)
exp=$startexp
for stepnum in `seq 1 $expsteps`
do
	val=$(echo "scale=$scale; e($exp * l($base))" | bc -l)
	linearresult=${linearresult}${val},
	exp=$(echo "scale=$scale; $exp + $stepsize" | bc -l)
done

# Concatenate result and cut off last trailing comma
result=${burninresult}${linearresult}${burnoutresult}
result=${result::-1}
echo $result
