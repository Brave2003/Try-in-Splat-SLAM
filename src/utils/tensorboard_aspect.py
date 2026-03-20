import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


class TensorboardLoggingAspect:
    """AOP-style logging aspect for mapping runtime statistics."""

    def __init__(
        self,
        training_cfg: Dict[str, Any],
        save_dir: str,
        printer: Optional[Any] = None,
        font_color: Optional[Any] = None,
    ) -> None:
        self.enabled = bool(training_cfg.get("tensorboard", False))
        self.log_every = int(training_cfg.get("tb_log_every", 20))
        self.hist_every = int(training_cfg.get("tb_hist_every", 200))
        self.log_dir = training_cfg.get("tb_log_dir", None)
        self.new_run_subdir = bool(training_cfg.get("tb_new_run_subdir", True))
        self.run_name = training_cfg.get("tb_run_name", None)
        self.writer: Optional[Any] = None
        self._printer = printer
        self._font_color = font_color

        if not self.enabled:
            return

        if SummaryWriter is None:
            self._print(
                "TensorBoard disabled because torch.utils.tensorboard is unavailable."
            )
            self.enabled = False
            return

        base_log_dir = self.log_dir
        if not base_log_dir:
            base_log_dir = os.path.join(save_dir, "tensorboard")

        final_log_dir = base_log_dir
        if self.new_run_subdir:
            run_name = self.run_name
            if not run_name:
                run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
            final_log_dir = os.path.join(base_log_dir, run_name)

        os.makedirs(final_log_dir, exist_ok=True)
        self.log_dir = final_log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self._print(f"TensorBoard enabled. log_dir={self.log_dir}")

    def _print(self, message: str) -> None:
        if self._printer is None:
            return
        if self._font_color is None:
            self._printer.print(message)
            return
        self._printer.print(message, self._font_color)

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0.0
            if value.numel() == 1:
                return float(value.detach().item())
            return float(value.detach().mean().item())
        return float(value)

    def log_mapping_stats(
        self,
        step: int,
        stats: Dict[str, Any],
        gaussians: Optional[Any] = None,
        namespace: str = "mapping",
    ) -> None:
        if self.writer is None or not self.enabled:
            return
        if step % max(1, self.log_every) != 0:
            return

        for key, value in stats.items():
            if value is None:
                continue
            self.writer.add_scalar(
                f"{namespace}/{key}", self._to_float(value), int(step)
            )

        if gaussians is not None and step % max(1, self.hist_every) == 0:
            try:
                scaling = gaussians.get_scaling
                if isinstance(scaling, torch.Tensor) and scaling.numel() > 0:
                    self.writer.add_histogram(
                        "gaussians/scaling",
                        scaling.detach().float().reshape(-1),
                        int(step),
                    )
            except Exception:
                pass
            try:
                opacity = gaussians.get_opacity
                if isinstance(opacity, torch.Tensor) and opacity.numel() > 0:
                    self.writer.add_histogram(
                        "gaussians/opacity",
                        opacity.detach().float().reshape(-1),
                        int(step),
                    )
            except Exception:
                pass

        self.writer.flush()

    def log_scalar_group(
        self,
        step: int,
        main_tag: str,
        scalar_dict: Dict[str, Any],
    ) -> None:
        if self.writer is None or not self.enabled:
            return
        if step % max(1, self.log_every) != 0:
            return
        if not scalar_dict:
            return

        converted: Dict[str, float] = {}
        for k, v in scalar_dict.items():
            if v is None:
                continue
            converted[k] = self._to_float(v)

        if not converted:
            return

        self.writer.add_scalars(main_tag, converted, int(step))
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
