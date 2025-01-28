# 0: Circle 1: XOR
python sneat_tune_cls.py \
    --cls_id 0 \
    --log_dir ../runs/sneat_tune_cls_0 \
    --pop_size 64 \
    --n_iter 120 \
    --topology_pick 9 \
    --backprop_pick 3 \
    --refresh_pop_interval 3 \
    --train_size 2000 \
    --test_size 500 \
    --batch_size 1000