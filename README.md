# HMMs for Language Modeling (Pytorch version)

## Dependencies
* [TVM](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github) 0.7.dev1. This has a dependency on LLVM.
* Pytorch 1.5
* Torchtext 0.6.0
* Wandb 0.10.1

### Brown Clusters
Clone the Brown Cluster repo from
[github.com/percyliang/brown-cluster](https://github.com/percyliang/brown-cluster)
and install it via `make`.

Preprocess the data by flattening the data , which is only used for producing Brown Clusters.
```
python scripts/preprocess_dataset.py
```

Export the `l_cluster` to the path of the `brown-cluster/wcluster` command.
```
export l_cluster=/path/to/brown-cluster/wcluster
```

Run the Brown Cluster script to obtain clusters for PTB and WikiText-2.
```
bash scripts/brown_cluster.sh lm128
bash scripts/brown_cluster.sh w2flm128
```

### Very Large HMM (VL-HMM)

To train a 32k state HMM on PTB, run
```
source scripts/hmm_commands.sh && run_ptb
```
An example run can be found [here](https://wandb.ai/justinchiu/hmm-lm/runs/1bkzvixn/logs).

To train a 32k state HMM on WikiText-2, run
```
source scripts/hmm_commands.sh && run_w2
```
An example run can be found [here](https://wandb.ai/justinchiu/hmm-lm/runs/1ciz3kxf/logs).


