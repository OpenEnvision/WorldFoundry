import subprocess
from pathlib import Path
from typing import List, Optional

from .base import WorldModelRunner


class LocalScriptRunner(WorldModelRunner):
    """Generic world-model adapter for locally-installed models exposed via a
    shell script.

    The script is invoked with a standard set of arguments so this single adapter
    covers any local model (Hunyuan, WAN2.2, etc.) as long as its repo ships a
    thin wrapper script that translates these args into the model's native CLI.

    Standard script contract — the script MUST accept:
        --prompt TEXT           The text prompt.
        --output  PATH          Where to write the generated .mp4.
        --init_frame PATH       (optional) Conditioning image.
        --camera_control STRING (optional) Camera-control parameters; only passed
                                when the model config has a camera_control_field.

    Any additional model-specific flags can be forwarded via `extra_args` in the
    model config.

    Config keys (in addition to base keys):
        repo_path   (str)        Absolute path to the model's local repository.
        script      (str)        Wrapper script path, relative to repo_path.
        extra_args  (list[str])  Extra CLI args appended verbatim (default []).
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        repo_path_str = config.get("repo_path", "")
        if not repo_path_str:
            raise ValueError(
                f"[{name}] 'repo_path' is required in the model config."
            )
        self.repo_path = Path(repo_path_str).expanduser().resolve()
        self.script: str = config.get("script", "run.sh")
        self.extra_args: List[str] = list(config.get("extra_args") or [])
        self.camera_control_default: Optional[str] = config.get("camera_control_default") or None
        self.n_gpu: Optional[int] = config.get("n_gpu") or None
        self.gpu_per_task: Optional[int] = config.get("gpu_per_task") or None
        self.verbose: bool = bool(config.get("verbose", False))
        self.daemon_script: Optional[str] = config.get("daemon_script") or None
        self.daemon_args: List[str] = list(config.get("daemon_args") or [])
        self.torchrun_path: str = config.get("torchrun_path") or "torchrun"

    def build_daemon_cmd(
        self,
        tasks_json_path: str,
        nproc_per_node: Optional[int] = None,
        master_port: Optional[int] = None,
    ) -> List[str]:
        """Return the torchrun command to launch the batch daemon.

        Subclasses may override to append runner-specific flags.
        The daemon script receives --tasks_json and any extra daemon_args.

        Args:
            nproc_per_node: Override the number of processes per node.
                            Defaults to self.n_gpu (or 1 if unset).
            master_port:    Port for torchrun rendezvous.  Each parallel worker
                            must use a distinct port to avoid EADDRINUSE errors.
                            Defaults to torchrun's built-in default (29500).
        """
        if not self.daemon_script:
            raise RuntimeError(f"[{self.name}] No daemon_script configured.")
        n = nproc_per_node if nproc_per_node is not None else (self.n_gpu or 1)
        script_path = self.repo_path / self.daemon_script
        cmd = [
            self.torchrun_path,
            f"--nproc_per_node={n}",
        ]
        if master_port is not None:
            cmd.append(f"--master_port={master_port}")
        cmd += [
            str(script_path),
            "--tasks_json", tasks_json_path,
        ] + self.daemon_args
        return cmd

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

        cmd = [
            "bash", str(script_path),
            "--prompt", prompt,
            "--output", str(output_path),
        ]
        if init_frame and init_frame.exists():
            cmd += ["--init_frame", str(init_frame)]
        effective_camera_control = camera_control or self.camera_control_default
        if effective_camera_control:
            cmd += ["--camera_control", effective_camera_control]
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
            print(
                f"[{self.name}] Script exited {result.returncode} for task {task_id}"
            )
            return False

        if not output_path.exists():
            print(
                f"[{self.name}] Script exited 0 but output not found: {output_path}"
            )
            return False

        return True
