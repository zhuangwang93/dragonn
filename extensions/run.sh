for size in 64 128 256
#for size in 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768
do
    for sparsity in 0.001 0.005 0.01 0.02 0.05 0.1
    do
        python topk.py $size $sparsity | tee -a log
    done
done
