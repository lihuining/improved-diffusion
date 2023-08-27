

MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --learn_sigma True --noise_schedule cosine --use_kl True"
Train_FLAGS ="--lr 1e-4 --batch_size 24"
python scripts/image_train.py --data_dir datasets/cifar_train $MODEL_FLAGS $DIFFUSION_FLAGS $Train_FLAGS
