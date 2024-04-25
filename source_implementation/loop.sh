screen -dmS "100" bash -c "CUDA_VISIBLE_DEVICES=3 python train_msp_podcast_singletask.py --backbone base --seed 200 --batch_size 32 --num_epochs 40 --patience 20 --train_csv random_train_100 --ckpt_name random_100 --label valence >> base_100.txt;"
screen -dmS "100-de" bash -c "CUDA_VISIBLE_DEVICES=5 python train_msp_podcast_singletask.py --backbone base --seed 200 --batch_size 32 --num_epochs 50 --patience 20 --train_csv random_train_100_de --ckpt_name random_100_de --label valence >> base_100_de.txt;"
