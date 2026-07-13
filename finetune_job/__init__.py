"""FineTuneJob: the Python fine-tuning adapter for SmolVLA's action expert.

Per docs/design/model-runtime/design.md component 01.3 ("FineTuneJob
(Python)"): this package drives LeRobot's own real training entry point
(`lerobot-train` / `lerobot.scripts.lerobot_train.train`) and dataset format
(LeRobotDataset) rather than inventing a second training framework (the
model-runtime foundation's "Not a second training framework" no-goal). See
``finetune_job.job`` for the ``FineTuneJob`` class itself.
"""

from finetune_job.job import FineTuneJob

__all__ = ["FineTuneJob"]
