horovodrun -np 4 python3 main_xml.py --test_every 2 --epoch 50 --dataset wiki10 --hidden_dim 4096 --lr 1e-3 --batch_size 256
# horovodrun -np 4 python3 main_xml.py --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --test_every 2 --epoch 50 --dataset wiki10 --hidden_dim 4096 --lr 1e-3 --batching --batch_size 256
# horovodrun -np 4 python3 main_xml.py --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --test_every 2 --epoch 50 --dataset wiki10 --hidden_dim 4096 --lr 1e-3 --batch_size 256

# horovodrun -np 8 python3 main_xml.py --test_every 2 --epoch 50 --dataset wiki10 --hidden_dim 4096 --lr 1e-3 --batch_size 256
# horovodrun -np 8 python3 main_xml.py --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --test_every 2 --epoch 50 --dataset wiki10 --hidden_dim 4096 --lr 1e-3 --batching --batch_size 256
# horovodrun -np 8 python3 main_xml.py --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --test_every 2 --epoch 50 --dataset wiki10 --hidden_dim 4096 --lr 1e-3 --batch_size 256

#horovodrun -np 8 python3 main_xml.py --compress --compressor signsgd --memory none --comm allgather --compress-ratio 1e-2 --test_every 2 --epoch 100 --dataset wiki10 --hidden_dim 4096 --lr 1e-3 --batching --batch_size 256

# horovodrun -np 4 python3 main_xml.py --test_every 5 --epoch 30 --dataset eurlex4k --hidden_dim 10240 --lr 1e-2 --batch_size 256
# horovodrun -np 4 python3 main_xml.py --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --test_every 5 --epoch 30 --dataset eurlex4k --hidden_dim 10240 --lr 1e-2 --batching --batch_size 256
# horovodrun -np 4 python3 main_xml.py --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --test_every 5 --epoch 30 --dataset eurlex4k --hidden_dim 10240 --lr 1e-2 --batch_size 256

# horovodrun -np 8 python3 main_xml.py --test_every 5 --epoch 30 --dataset eurlex4k --hidden_dim 10240 --lr 1e-2 --batch_size 256
# horovodrun -np 8 python3 main_xml.py --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --test_every 5 --epoch 30 --dataset eurlex4k --hidden_dim 10240 --lr 1e-2 --batching --batch_size 256
# horovodrun -np 8 python3 main_xml.py --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --test_every 5 --epoch 30 --dataset eurlex4k --hidden_dim 10240 --lr 1e-2 --batch_size 256
