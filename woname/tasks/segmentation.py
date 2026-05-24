from base import TaskSpec

class SegmentationTask(TaskSpec):

    name = "segmentation"

    input_keys = ["image"]
    target_keys = ["mask"]
    output_keys = ["mask_logits"]

    def format_targets(self, batch):
        return {
            "mask": batch["mask"]
        }

    def format_outputs(self, outputs):
        return {
            "mask_logits": outputs["mask_logits"]
        }