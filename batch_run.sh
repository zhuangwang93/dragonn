horovodrun -np 1 /usr/bin/python3.7 main_vit.py --pcie --lr 1e-5 --batch_size 32 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 0.125 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 0.125 --lr 1e-5 --batch_size 32 --batching --threshold 102400 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 0.125 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor qsgd --memory none --comm allgather --compress-ratio 0.125 --lr 1e-5 --batch_size 32  --threshold 1024 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor topk --memory none --comm allgather --compress-ratio 0.125 --lr 1e-5 --batch_size 32 --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10

# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 0.0625 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_mixer.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 0.0625 --lr 1e-5 --batch_size 32 --batching --threshold 102400 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 0.0625 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor qsgd --memory none --comm allgather --compress-ratio 0.125 --lr 1e-5 --batch_size 32 --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor topk --memory none --comm allgather --compress-ratio 0.0625 --lr 1e-5 --batch_size 32 --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10

# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 0.03125 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 0.03125 --lr 1e-5 --batch_size 32 --batching --threshold 102400 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 0.03125 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# # horovodrun -np 4 /usr/bin/python3.7 main_mixer.py --pcie --compress --compressor qsgd --memory none --comm allgather --compress-ratio 0.03125 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor topk --memory none --comm allgather --compress-ratio 0.03125 --lr 1e-5 --batch_size 32 --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10

# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 0.016 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_mlp.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 0.016 --lr 1e-5 --batch_size 32 --batching --threshold 102400 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 0.016 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor topk --memory none --comm allgather --compress-ratio 0.016 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10


# horovodrun -np 4 /usr/bin/python3.7 main_mixer.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 0.125 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_mixer.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 0.0625 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_mixer.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 0.03125 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_mixer.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 0.016 --lr 1e-5 --batch_size 32  --threshold 10240 --epochs 1 --print_every 10 --pretrained --dataset cifar10

# horovodrun -np 8 python3 main_mixer.py --pcie --lr 1e-5 --batch_size 32 --epochs 1 --print_every 1 --pretrained --dataset cifar10
# horovodrun -np 8 python3 main_mixer.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 1 --pretrained --dataset cifar10
# horovodrun -np 8 python3 main_mixer.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --threshold 1024 --epochs 1 --print_every 1 --pretrained --dataset cifar10

# horovodrun -np 4 python3 main_mixer.py --pcie --lr 1e-5 --batch_size 32 --epochs 1 --print_every 1 --pretrained --dataset cifar100
# horovodrun -np 4 python3 main_mixer.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 1 --pretrained --dataset cifar100
# horovodrun -np 4 python3 main_mixer.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --threshold 4096 --epochs 1 --print_every 1 --pretrained --dataset cifar100

# horovodrun -np 8 python3 main_mixer.py --pcie --lr 1e-5 --batch_size 32 --epochs 1 --print_every 1 --pretrained --dataset cifar100
# horovodrun -np 8 python3 main_mixer.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 1 --pretrained --dataset cifar100
# horovodrun -np 8 python3 main_mixer.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --threshold 4096 --epochs 1 --print_every 1 --pretrained --dataset cifar100

# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --lr 5e-4 --batch_size 64 --epochs 200 --print_every 10 --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --compress --compressor atopk --memory dgc --comm allgather --compress-ratio 1e-2 --lr 5e-4 --batch_size 64 --batching --threshold 409600 --epochs 200 --print_every 100 --dataset cifar10
# horovodrun -np 4 /usr/bin/python3.7 main_vit.py --compress --compressor dgc --memory dgc --comm allgather --compress-ratio 1e-2 --lr 5e-4 --batch_size 64 --threshold 10240 --epochs 200 --print_every 100 --dataset cifar10

# horovodrun -np 8 python3 main_vit.py --pcie --lr 1e-5 --batch_size 32 --epochs 1 --print_every 1 --pretrained --dataset cifar10
# horovodrun -np 8 python3 main_vit.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 1 --pretrained --dataset cifar10
# horovodrun -np 8 python3 main_vit.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --threshold 1024 --epochs 1 --print_every 1 --pretrained --dataset cifar10

# horovodrun -np 4 python3 main_vit.py --pcie --lr 1e-5 --batch_size 32 --epochs 1 --print_every 1 --pretrained --dataset cifar100
# horovodrun -np 4 python3 main_vit.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 1 --pretrained --dataset cifar100
# horovodrun -np 4 python3 main_vit.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --threshold 4096 --epochs 1 --print_every 1 --pretrained --dataset cifar100

# horovodrun -np 8 python3 main_vit.py --pcie --lr 1e-5 --batch_size 32 --epochs 1 --print_every 1 --pretrained --dataset cifar100
# horovodrun -np 8 python3 main_vit.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 1 --pretrained --dataset cifar100
# horovodrun -np 8 python3 main_vit.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --threshold 4096 --epochs 1 --print_every 1 --pretrained --dataset cifar100
