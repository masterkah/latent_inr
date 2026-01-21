"""GPU utility functions for automatic GPU selection"""
import os
import subprocess


def auto_select_gpu():
    """
    Automatically select the most available GPU based on free memory and utilization.
    Sets CUDA_VISIBLE_DEVICES environment variable.
    """
    try:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            print(f"Using manually set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
            return

        cmd = "nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,nounits,noheader"
        result = subprocess.check_output(cmd.split(), encoding='utf-8')
        lines = result.strip().split('\n')

        best_gpu = -1
        max_free = 0
        
        print("Scanning GPUs...")
        for line in lines:
            try:
                idx, free_mem, util = line.split(', ')
                idx, free_mem, util = int(idx), int(free_mem), int(util)
                print(f"  GPU {idx}: Free Memory: {free_mem}MiB, Utilization: {util}%")
                
                if util < 10 and free_mem > max_free:
                    max_free = free_mem
                    best_gpu = idx
            except:
                continue

        if best_gpu != -1:
            print(f"\n✅ Auto-selected GPU {best_gpu} (Free: {max_free}MiB)")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        else:
            print("⚠️ No ideal GPU found, using default strategy.")

    except Exception as e:
        print(f"⚠️ GPU auto-selection failed: {e}")
