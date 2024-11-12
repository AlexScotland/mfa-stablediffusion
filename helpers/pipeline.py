import gc
import torch

def __clean_up_pipeline(pipeline):
    # Cleanup the pipeline
    del pipeline
    # Run garbage collection
    gc.collect()

    # Clear GPU memory cache
    torch.cuda.empty_cache()
