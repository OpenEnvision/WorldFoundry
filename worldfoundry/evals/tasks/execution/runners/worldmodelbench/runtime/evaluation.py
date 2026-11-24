import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from mmengine import dump, load
from rich.console import Console
from rich.panel import Panel

# from rich.progress import track
# from rich import print as rprint
from rich.progress import BarColumn, Progress, TimeRemainingColumn
from rich.table import Table
from tqdm import tqdm

from worldfoundry.base_models.llm_mllm_core.mllm.vila import Video, load_vila_model
from worldfoundry.evals.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset


class EvaluationType(Enum):
    INSTRUCTION = "instruction"
    PHYSICAL_LAWS = "physical_laws"
    COMMON_SENSE = "common_sense"


def get_default_prompt_templates() -> dict[str, str]:
    """Factory function for default prompt templates."""
    return {
        EvaluationType.INSTRUCTION.value: """
            Evaluate if this video follows the instruction: '{instruction}'.
            Use the following scoring criteria:
            
            - 0: The video does not follow the instruction at all.
            - 1: The video includes the correct object but performs the wrong action, or vice versa.
            - 2: The video follows the instruction and shows a tendency toward the intended goal.
            - 3: The video follows the instruction precisely and successfully achieves the goal.
            
            Let's analyze step-by-step and conclude with 'Score: [score]'.
        """.strip(),
        EvaluationType.PHYSICAL_LAWS.value: """
            Watch the video and determine if it shows any '{physical_laws}'
            Let's think step-by-step and conclude with "Yes" or "No".
        """.strip(),
        EvaluationType.COMMON_SENSE.value: """
            Does the video exhibit '{common_sense}'?
            Let's think step-by-step and conclude with "Yes" or "No".
        """.strip(),
    }


def get_default_question_pool() -> dict[str, list[str] | None]:
    """Factory function for default question pool."""
    return {
        EvaluationType.INSTRUCTION.value: None,
        EvaluationType.PHYSICAL_LAWS.value: [
            "Violation of Newton's Law: Objects move without any external force.",
            "Violation of the Law of Conservation of Mass or Solid Constitutive Law: Objects deform irregularly.",
            "Violation of Fluid Constitutive Law: Liquids flow in an unnatural manner.",
            "Violation of Non-physical Penetration: Objects unnaturally pass through each other.",
            "Violation of Gravity: Objects behave inconsistently with gravity.",
        ],
        EvaluationType.COMMON_SENSE.value: [
            "Poor Aesthetics: Visually unappealing or low-quality content.",
            "Temporal Inconsistency: Noticeable flickering or abrupt changes.",
        ],
    }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation prompts and scoring criteria."""

    PROMPT_TEMPLATES: dict[str, str] = field(default_factory=get_default_prompt_templates)
    QUESTION_POOL: dict[str, list[str] | None] = field(default_factory=get_default_question_pool)


class ResultsPrinter:
    """Handles formatted output of evaluation results."""

    def __init__(self):
        self.console = Console()

    def print_header(self, text: str):
        """Print a styled header."""
        self.console.print(f"\n[bold blue]{text}[/bold blue]")

    def print_score(self, category: str, score: float, indent: int = 0):
        """Print a score with proper formatting."""
        indent_str = " " * indent
        self.console.print(f"{indent_str}[cyan]{category}:[/cyan] [yellow]{score:.2f}[/yellow]")

    def create_results_table(self, category: str, scores: dict[str, float]) -> Table:
        """Create a rich table for displaying results."""
        table = Table(title=f"{category} Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right", style="yellow")

        for metric, score in scores.items():
            table.add_row(metric, f"{score:.2f}")

        return table

    def print_summary_panel(self, total_score: float, num_categories: int):
        """Print a panel with summary information."""
        panel = Panel(
            f"[bold green]Total Score: {total_score:.2f}[/bold green]\n",
            # f"[blue]Average per category: {total_score/num_categories:.2f}[/blue]",
            title="Evaluation Summary",
            border_style="green",
        )
        self.console.print(panel)


class WorldModelEvaluator:
    """Evaluates world model benchmark videos using VILA model."""

    def __init__(self, judge_path: str, video_dir: str, config: EvaluationConfig):
        self.judge = self._load_judge(judge_path)
        self.video_dir = Path(video_dir)
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.printer = ResultsPrinter()

    @staticmethod
    def _load_judge(judge_path: str):
        """Load the VILA judge model through WorldFoundry's in-tree runtime."""
        return load_vila_model(
            judge_path,
            device_map=os.environ.get("WORLDFOUNDRY_WORLDMODELBENCH_DEVICE_MAP", "auto"),
            torch_dtype=os.environ.get("WORLDFOUNDRY_WORLDMODELBENCH_DTYPE", "float16"),
            attn_implementation=os.environ.get("WORLDFOUNDRY_WORLDMODELBENCH_ATTN_IMPL", "flash_attention_2"),
        )

    def _load_video(self, video_name: str) -> Video | None:
        """Load a video file for evaluation."""
        video_path = self.video_dir / f"{video_name}.mp4"
        if not video_path.exists():
            self.logger.warning(f"Video not found: {video_path}")
            return None
        return Video(str(video_path))

    def evaluate_video(self, video: Video, prompt: str, cot: bool = True) -> str:
        """Generate evaluation content for a video."""
        if not cot:
            prompt = prompt.replace("Let's think step-by-step and conclude with", "Answer with").replace(
                "Let's analyze step-by-step and conclude with", "Answer with"
            )
        return self.judge.generate_content([video, prompt])

    def process_results(self, preds: dict, accs: defaultdict) -> float:
        """Process and print evaluation results with rich formatting."""
        num_insts = len(preds)
        total_score = 0

        category_mapping = {2: [("framewise", "temporal")], 5: [("newton", "mass", "fluid", "penetration", "gravity")]}

        for category, scores in accs.items():
            self.printer.print_header(f"{category.replace('_', ' ').title()} Details")
            num_sub = len(scores) // num_insts

            if num_sub == 1:
                overall = np.mean(scores)
                self.printer.print_score("Overall", overall)
                total_score += overall
            elif num_sub in category_mapping:
                sub_scores = {}
                for i, sub in enumerate(category_mapping[num_sub][0]):
                    sub_mean = np.mean(scores[i::num_sub])
                    sub_scores[sub.title()] = sub_mean

                # Create and display results table
                table = self.printer.create_results_table(category.replace("_", " ").title(), sub_scores)
                self.printer.console.print(table)

                overall = np.sum(list(sub_scores.values()))
                self.printer.print_score("Overall", overall, indent=2)
                total_score += overall
            else:
                raise ValueError(f"Unexpected number of subcategories: {num_sub}")

        self.printer.print_summary_panel(total_score, len(accs))
        return total_score


def save_results(results: dict, save_path: str):
    """Save evaluation results to a file."""
    dump(results, save_path, indent=4)
    Console().print(f"[green]Results saved to: {save_path}[/green]")


class RichLogHandler(logging.Handler):
    """Custom logging handler that uses Rich for formatting."""

    def __init__(self):
        super().__init__()
        self.console = Console()

    def emit(self, record):
        try:
            msg = self.format(record)
            style = "bold red" if record.levelno >= logging.WARNING else "blue"
            self.console.print(f"[{style}]{msg}[/{style}]")
        except Exception:
            self.handleError(record)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate World Model Benchmark")
    parser.add_argument("--judge", type=str, required=True, help="Path to judge model checkpoint")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to generated video directory")
    parser.add_argument("--model_name", type=str, required=True, help="Tested model name")
    parser.add_argument(
        "--save_name", type=str, default="worldmodelbench_results", help="Path to save evaluation results"
    )
    parser.add_argument("--cot", action="store_true", help="Enable Chain-of-Thought output")
    parser.add_argument("--no-save", action="store_true", help="Disable saving results")

    args = parser.parse_args()

    # Setup logging with custom Rich handler
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichLogHandler()])
    logger = logging.getLogger(__name__)

    # Initialize evaluator
    config = EvaluationConfig()
    evaluator = WorldModelEvaluator(args.judge, args.video_dir, config)
    printer = ResultsPrinter()

    # Load validation set with status message
    printer.console.print("[bold]Loading validation set...[/bold]")
    manifest_path = os.environ.get("WORLDFOUNDRY_WORLDMODELBENCH_MANIFEST")
    if manifest_path is None:
        bundled_manifest = bundled_benchmark_asset("worldmodelbench", "worldmodelbench.json")
        manifest_path = str(bundled_manifest) if bundled_manifest.is_file() else "./worldmodelbench.json"
    validation_set = load(manifest_path)

    # Check for existing results
    save_path = f"{args.save_name}_cot.json" if args.cot else f"{args.save_name}.json"
    if os.path.exists(save_path):
        printer.console.print("[bold yellow]Loading existing results...[/bold yellow]")
        results = load(save_path)
        try:
            preds, accs = results["preds"], results["accs"]
        except KeyError:
            raise KeyError("Expected keys not found in results file")  # noqa: B904 - Preserve the implicit context.
    else:
        printer.console.print("[bold green]Starting new evaluation...[/bold green]")
        preds = {}
        accs = defaultdict(list)

        # Create a single progress instance for all operations
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            console=printer.console,
        ) as progress:
            # Main task for video processing
            video_task = progress.add_task("Processing videos", total=len(validation_set))

            for _vid, v_i in tqdm(enumerate(validation_set), total=len(validation_set)):
                video_name = Path(v_i["first_frame"]).stem
                video = evaluator._load_video(video_name)
                if not video:
                    progress.advance(video_task)
                    continue

                # Evaluation task
                eval_task = progress.add_task(f"Evaluating {video_name}", total=len(EvaluationType))

                for eval_type in EvaluationType:
                    preds_i = []
                    prompt_template = config.PROMPT_TEMPLATES[eval_type.value]
                    questions = config.QUESTION_POOL[eval_type.value]

                    if questions:
                        accs_i = []
                        # Questions task
                        question_task = progress.add_task(
                            f"Processing {eval_type.value} questions", total=len(questions)
                        )

                        for question in questions:
                            format_kwargs = {f"{eval_type.value}": question.lower()}
                            prompt = prompt_template.format(**format_kwargs)
                            pred = evaluator.evaluate_video(video, prompt, args.cot)
                            preds_i.append(pred)
                            accs_i.append("no" in pred.lower())
                            progress.advance(question_task)

                        progress.remove_task(question_task)
                        accs[eval_type.value].extend(accs_i)
                    else:
                        prompt = prompt_template.format(instruction=v_i["text_instruction"])
                        pred = evaluator.evaluate_video(video, prompt, args.cot)
                        preds_i.append(pred)
                        try:
                            score = float(pred.split(":")[-1].strip(" ."))
                        except ValueError:
                            logger.warning(f"Could not parse score from prediction: {pred}")
                            score = 0
                        accs[eval_type.value].append(score)

                    if video_name not in preds:
                        preds[video_name] = {}
                    preds[video_name][eval_type.value] = preds_i
                    progress.advance(eval_task)

                progress.remove_task(eval_task)
                progress.advance(video_task)

        # Save results if requested
        if not args.no_save:
            results = {"model_name": args.model_name, "preds": preds, "accs": accs}
            save_results(results, save_path)

    # Process and display results
    printer.console.print("\n[bold]Final Evaluation Results[/bold]")
    evaluator.process_results(preds, accs)


if __name__ == "__main__":
    main()
