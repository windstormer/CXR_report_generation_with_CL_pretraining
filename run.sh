# python3 train_SSL.py
# python3 train_AE.py 
# python3 train_RNN.py --pretrain=SSL_p256_ep500_b32 --strategy=finetune
# python3 train_RNN.py --pretrain=SSL_p256_ep500_b32 --strategy=freeze
# python3 train_RNN.py --pretrain=Imagenet --strategy=finetune
# python3 train_RNN.py --pretrain=Imagenet --strategy=freeze
# python3 train_RNN.py --pretrain=AE_p256_ep500_b32 --strategy=finetune
# python3 train_RNN.py --pretrain=AE_p256_ep500_b32 --strategy=freeze
# python3 train_RNN.py --strategy=finetune
# python3 train_RNN.py --pretrain=SegSSL_p256_ep500_b32 --strategy=finetune
# python3 train_RNN.py --pretrain=SegSSL_p256_ep500_b32 --strategy=freeze
# python3 train_RNN.py --pretrain=Tag_ResNet50_p256_ep2000_b32_balance_40 --strategy=finetune
# python3 train_RNN.py --pretrain=Tag_ResNet50_p256_ep2000_b32_balance_40 --strategy=freeze
# python3 train_RNN.py --pretrain=Moco_p256_ep500_b32 --strategy=finetune
python3 train_RNN.py --pretrain=Moco_p256_ep500_b32 --strategy=freeze

