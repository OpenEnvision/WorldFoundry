import subprocess
from pathlib import Path
from typing import List, Optional

from .local_script import LocalScriptRunner


class LingBotRunner(LocalScriptRunner):
    """World-model adapter for LingBot-World.

    LingBot takes --action_path (a folder with poses.npy + intrinsics.npy)
    rather than a --camera_control string.  This runner maps each pose string
    to a pre-computed action folder under lingbot_actions_root.

    Pre-compute the action folders with:
        python -m generation.world_models.precompute_lingbot_actions

    Config keys (in addition to LocalScriptRunner keys):
        lingbot_actions_root  (str)  Path to the directory containing
                                     pre-computed action sub-folders.
                                     Relative paths are resolved from the
                                     repo root (two levels up from this file).
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        actions_root = config.get("lingbot_actions_root", "")
        if not actions_root:
            raise ValueError(f"[{name}] 'lingbot_actions_root' is required in the model config.")
        p = Path(actions_root).expanduser()
        if not p.is_absolute():
            # Resolve relative to the repo root (this file lives two levels deep)
            p = (Path(__file__).parents[2] / p).resolve()
        self._actions_root = p

    def build_daemon_cmd(
        self,
        tasks_json_path: str,
        nproc_per_node: Optional[int] = None,
        master_port: Optional[int] = None,
    ) -> List[str]:
        """Extend the base daemon command with the LingBot actions root."""
        return super().build_daemon_cmd(
            tasks_json_path, nproc_per_node=nproc_per_node, master_port=master_port
        ) + [
            "--lingbot_actions_root", str(self._actions_root),
        ]

    @staticmethod
    def _pose_to_folder(pose_string: str) -> str:
        """Sanitize a pose string into its action folder name."""
        return pose_string.replace(", ", "_").replace(" ", "_")

    def generate(
        self,
        task_id: str,
        prompt: str,
        init_frame: Optional[Path],
        output_path: Path,
        camera_control: Optional[str] = None,
    ) -> bool:
        script_path = self.repo_path / self.script
        if not script_path.exists():
            print(f"[{self.name}] Script not found: {script_path}")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resolve pose string → action folder (fall back to model default)
        effective_cc = camera_control or self.camera_control_default
        if not effective_cc:
            print(f"[{self.name}] No camera control for task {task_id} and no default set.")
            return False

        action_path = self._actions_root / self._pose_to_folder(effective_cc)
        if not action_path.exists():
            print(
                f"[{self.name}] Action folder not found: {action_path}\n"
                f"  Pose string: '{effective_cc}'\n"
                f"  Run: python -m generation.world_models.precompute_lingbot_actions"
            )
            return False

        cmd = [
            "bash", str(script_path),
            "--prompt", prompt,
            "--output", str(output_path),
        ]
        if init_frame and init_frame.exists():
            cmd += ["--init_frame", str(init_frame)]
        cmd += ["--action_path", str(action_path)]
        if self.n_gpu is not None:
            cmd += ["--n_gpu", str(self.n_gpu)]
        cmd += self.extra_args

        result = subprocess.run(
            cmd,
            cwd=str(self.repo_path),
            capture_output=not self.verbose,
            text=not self.verbose,
        )

        if not self.verbose and result.returncode != 0:
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")

        if result.returncode != 0:
            print(f"[{self.name}] Script exited {result.returncode} for task {task_id}")
            return False

        if not output_path.exists():
            print(f"[{self.name}] Script exited 0 but output not found: {output_path}")
            return False

        return True
