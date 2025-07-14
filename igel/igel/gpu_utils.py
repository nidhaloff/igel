def detect_gpu():
    """
    Detect if a GPU is available for PyTorch or TensorFlow.
    Returns a string describing the GPU status.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return f"PyTorch GPU available: {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return f"TensorFlow GPU available: {gpus[0].name}"
    except ImportError:
        pass
    return "No GPU detected"


def report_gpu_utilization():
    """
    Report GPU utilization using GPUtil if available.
    """
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.name}, load: {gpu.load*100:.1f}%, memory: {gpu.memoryUsed}/{gpu.memoryTotal}MB")
    except ImportError:
        print("GPUtil not installed. Skipping detailed GPU utilization.")
    except Exception as e:
        print(f"Error reporting GPU utilization: {e}") 