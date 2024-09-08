cmd=""

for i in `seq 5.0 1.0 20.0`; 
do  
    cmd="${cmd}python3 data_gen_kine_rand_uniform.py $i & "
    # cmd="${cmd}python3 data_gen_fric_track.py $i & "
done

eval "$cmd"