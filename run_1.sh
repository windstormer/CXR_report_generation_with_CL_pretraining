python3 train_RNN.py --pretrain=SSL_p224_ep250_b32 --strategy=finetune -c GRU --gpu_id 1
python3 train_RNN.py --pretrain=SSL_p224_ep250_b32 --strategy=freeze -c GRU --gpu_id 1
python3 train_RNN.py --pretrain=Imagenet --strategy=finetune -c GRU --gpu_id 1
python3 train_RNN.py --pretrain=Imagenet --strategy=freeze -c GRU --gpu_id 1
python3 train_RNN.py --pretrain=AE_p224_ep250_b32 --strategy=finetune -c GRU --gpu_id 1
python3 train_RNN.py --pretrain=AE_p224_ep250_b32 --strategy=freeze -c GRU --gpu_id 1
python3 train_RNN.py --strategy=finetune -c GRU --gpu_id 1