You need to specify your data path (main.py & data_processing.py) and model (model.py)
I put a sample model into the folder (sample_model.py)

1. python data_processing.py
2. python main [CUDA_VISIBLE_DEVICES] [train / test] [snapshot_path]
    e.g. python main 0,1,2,3 train
         python main 0 test ./snapshots/weight.pth