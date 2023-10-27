cmd=""

for i in `seq 5.0 0.1 9.0`; 
do  
    cmd="${cmd}python3 data_gen_fric_rand_uniform.py $i & "
    # cmd="${cmd}python3 data_gen_fric_track.py $i & "
done

eval "$cmd"