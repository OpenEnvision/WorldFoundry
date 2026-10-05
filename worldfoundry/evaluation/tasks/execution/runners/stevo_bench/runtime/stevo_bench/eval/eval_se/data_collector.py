"""
Data collector for Gemini-based state evolution evaluation.
Collects task data from summary.json files and organizes resources.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import yaml


class TaskDataCollector:
    def __init__(self, project_root: str, task_dir: str = "benchmark/tasks_v4"):
        self.project_root = Path(project_root)
        self.runs_dir = self.project_root / "runs"
        self.outputs_dir = self.project_root / "outputs"
        self.benchmark_dir = self.project_root / task_dir

    def get_run_configs(self) -> List[Dict[str, str]]:
        """Define the runs to evaluate."""
        return [
            {
                "run_name": "sora_pro_sora_pro_subset_run_3",
                "summary_path": "sora_pro_output_map_sora_pro_subset_run_3",
                "output_path": "sora_pro_sora_pro_subset_run_3"
            },
            {
                "run_name": "veo_veo_run_6",
                "summary_path": "veo_output_map_veo_run_6",
                "output_path": "veo_veo_run_6"
            },
            {
                "run_name": "LingBot_lingbot_run_1",
                "summary_path": "LingBot_output_map_lingbot_run_1",
                "output_path": "LingBot_lingbot_run_1"
            },
            {
                "run_name": "HY-WorldPlay_hunyuan_run_2",
                "summary_path": "HY-WorldPlay_output_map_hunyuan_run_2",
                "output_path": "HY-WorldPlay_hunyuan_run_2"
            },
            {
                "run_name": "genie_genie_run_1",
                "summary_path": "genie",
                "output_path": "genie_genie_run_1"
            },
            {
                "run_name": "wan22_wan_run_1",
                "summary_path": "wan22_wan_run_1",
                "output_path": "wan22_wan_run_1"
            }
        ]

    def load_summary(self, run_config: Dict[str, str]) -> Dict:
        """Load summary.json for a run."""
        summary_path = self.runs_dir / run_config["summary_path"] / "summary.json"
        with open(summary_path, 'r') as f:
            return json.load(f)

    def filter_tasks(self, summary: Dict) -> List[Dict]:
        """Filter tasks that don't end with _00 (all levels)."""
        filtered_tasks = []
        for task in summary.get("tasks", []):
            task_id = task.get("task_id", "")

            # Filter: NOT ending with _00 (all levels)
            if not task_id.endswith("_00"):
                filtered_tasks.append(task)

        return filtered_tasks

    def find_task_yaml(self, task_id: str) -> Optional[Path]:
        """Find the YAML file for a task in either image_implied or simple_kickoff."""
        for subdir in ["image_implied", "simple_kickoff"]:
            yaml_path = self.benchmark_dir / subdir / task_id / f"{task_id}.yaml"
            if yaml_path.exists():
                return yaml_path
        return None

    def find_init_image(self, task_id: str) -> Optional[Path]:
        """Find the initial image for a task."""
        for subdir in ["image_implied", "simple_kickoff"]:
            # Try different possible image names
            for img_name in [f"{task_id}_init_frame.png", f"{task_id}_init_frame_large.png"]:
                img_path = self.benchmark_dir / subdir / task_id / img_name
                if img_path.exists():
                    return img_path
        return None

    def find_video(self, run_config: Dict[str, str], task_id: str) -> Optional[Path]:
        """Find the generated video for a task."""
        video_path = self.outputs_dir / run_config["output_path"] / f"{task_id}.mp4"
        if video_path.exists():
            return video_path
        return None

    def load_task_yaml(self, yaml_path: Path) -> Dict:
        """Load and parse task YAML file."""
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def collect_task_data(self, run_config: Dict[str, str], task_info: Dict) -> Optional[Dict]:
        """Collect all data for a single task."""
        task_id = task_info.get("task_id")

        # Find resources
        yaml_path = self.find_task_yaml(task_id)
        init_image = self.find_init_image(task_id)
        video_path = self.find_video(run_config, task_id)

        # Check if all required resources exist
        if not all([yaml_path, init_image, video_path]):
            print(f"Warning: Missing resources for task {task_id}")
            print(f"  YAML: {yaml_path}")
            print(f"  Image: {init_image}")
            print(f"  Video: {video_path}")
            return None

        # Load YAML data
        yaml_data = self.load_task_yaml(yaml_path)

        # Extract camera_WM (action prompt)
        camera_wm = yaml_data.get("prompts", {}).get("camera_WM", "")
        if not camera_wm or camera_wm.strip() == "":
            camera_wm = None

        return {
            "task_id": task_id,
            "task_level": task_info.get("task_level"),
            "run_name": run_config["run_name"],
            "init_image_path": str(init_image),
            "video_path": str(video_path),
            "action_prompt": camera_wm,
            "yaml_data": yaml_data
        }

    def collect_all_tasks(self) -> List[Dict]:
        """Collect data for all level 4 tasks across all runs."""
        all_tasks = []

        for run_config in self.get_run_configs():
            print(f"\nProcessing run: {run_config['run_name']}")

            # Load summary
            summary = self.load_summary(run_config)

            # Filter tasks (non-_00)
            filtered_tasks = self.filter_tasks(summary)
            print(f"  Found {len(filtered_tasks)} tasks (non-_00)")

            # Collect data for each task
            for task_info in filtered_tasks:
                task_data = self.collect_task_data(run_config, task_info)
                if task_data:
                    all_tasks.append(task_data)
                    print(f"    ✓ {task_data['task_id']}")

        return all_tasks


if __name__ == "__main__":
    # Test the data collector
    collector = TaskDataCollector(".")
    tasks = collector.collect_all_tasks()

    print(f"\n{'='*60}")
    print(f"Total tasks collected: {len(tasks)}")
    print(f"{'='*60}")

    # Show sample
    if tasks:
        print("\nSample task:")
        sample = tasks[0]
        for key, value in sample.items():
            if key != "yaml_data":
                print(f"  {key}: {value}")
