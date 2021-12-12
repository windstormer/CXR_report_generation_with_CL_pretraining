# python3 train_RNN.py --pretrain=SSL_p256_ep500_b32 --strategy=finetune -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=SSL_p256_ep500_b32 --strategy=freeze -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=Imagenet --strategy=finetune -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=Imagenet --strategy=freeze -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=AE_p256_ep500_b32 --strategy=finetune -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=AE_p256_ep500_b32 --strategy=freeze -c GRU --gpu_id 1
# python3 train_RNN.py --strategy=finetune -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=SegSSL_p256_ep500_b32 --strategy=finetune -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=SegSSL_p256_ep500_b32 --strategy=freeze -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=Tag_ResNet50_p256_ep2000_b32_balance_40 --strategy=finetune -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=Tag_ResNet50_p256_ep2000_b32_balance_40 --strategy=freeze -c GRU --gpu_id 1
# python3 train_RNN.py --pretrain=Moco_p256_ep500_b32 --strategy=finetune -c GRU --gpu_id 1
python3 train_RNN.py --pretrain=Moco_p256_ep500_b32 --strategy=freeze -c GRU --gpu_id 1