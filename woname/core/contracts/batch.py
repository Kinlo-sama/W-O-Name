class BatchSpec:
    """
    Contract of all batches in the framework
    """
    image: "Tensor[B,3,H,W]"
    target: "Dict[str, Tensor]"