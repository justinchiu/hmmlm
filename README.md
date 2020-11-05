# HMMs for Language Modeling (Pytorch version)

## Dependencies
* [TVM](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github). This has a dependency on LLVM.
* Pytorch 1.5
* Torchtext 0.6.0
* Wandb

### Brown Clusters
Clone the Brown Cluster repo from
[github.com/percyliang/brown-cluster](https://github.com/percyliang/brown-cluster)
and install it via `make`.

Preprocess the data by flattening the data , which is only used for producing Brown Clusters.
```
python scripts/preprocess_dataset.py
```

Run the Brown Cluster script to obtain clusters.
```
bash scripts/brown_cluster.sh lm128
bash scripts/brown_cluster.sh w2flm128
```

### Very Large HMM (VL-HMM)

To train a 32k state HMM on PTB, run
```
WANDB_MODE=dryrun python main.py --lr 0.01 --model factoredhmm --assignment brown --states_per_word 256 --train_spw 128 --num_clusters 128 --num_classes 32768 --bsz 16 --eval_bsz 16 --bptt 32 --dataset ptb --iterator bptt --reset_eos 1 --no_shuffle_train 0 --optimizer adamw --state ind --tw slIlrIrp
```

To train a 32k state HMM on WikiText-2, run
```
WANDB_MODE=dryrun python main.py --lr 0.01 --model factoredhmm --assignment brown --states_per_word 256 --train_spw 128 --num_clusters 128 --num_classes 32768 --bsz 16 --eval_bsz 16 --bptt 32 --dataset wikitext2 --iterator bptt --reset_eos 1 --no_shuffle_train 0 --optimizer adamw --state ind --tw slIlrIrp
```


