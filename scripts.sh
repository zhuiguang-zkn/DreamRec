python -u DreamRec.py --data yc --timesteps 500 --batch_size 1024 --lr 0.001 --beta_sche exp --w 2 --optimizer adamw --diffuser_type mlp1 --random_seed 100

python -u DreamRec.py --data ks --timesteps 2000 --lr 0.00005 --beta_sche cosine --w 2 --optimizer adamw --diffuser_type mlp1 --random_seed 100

python -u DreamRec.py --data zhihu --timesteps 500 --lr 0.01 --beta_sche linear --w 4 --optimizer adamw --diffuser_type mlp1 --random_seed 100 