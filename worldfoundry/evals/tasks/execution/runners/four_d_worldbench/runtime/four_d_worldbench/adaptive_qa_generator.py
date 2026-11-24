#!/usr/bin/env python3
"""
Adaptive question generator
Automatically generate evaluation questions based on different dimensions and input content
"""

import argparse
import json
import os
from typing import Any

from openai import OpenAI

# Import question cache manager
try:
    from question_cache_manager import QuestionCacheManager

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("Warning: question_cache_manager not available. Questions will not be cached.")


class AdaptiveQAGenerator:
    def __init__(self, api_base_url: str | None = None, api_key: str | None = None):
        """Initialize question generator"""
        # Use default configuration or passed-in configuration
        self.client = OpenAI(
            base_url=api_base_url or os.environ.get("OPENAI_BASE_URL", ""),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
        )
        self.model_type = os.environ.get("OPENAI_MODEL", "gpt-4o")

        # Initialize cache manager
        if CACHE_AVAILABLE:
            self.cache_manager = QuestionCacheManager()
        else:
            self.cache_manager = None

        # Define specialized question generation templates for different dimensions
        self.dimension_prompts = {
            "alignment_relationship_control": """
You are an expert in caption analysis focusing on object spatial relationship and relative position changes in the whole caption.
Analyze the following video content and generate 5 specific yes/no questions that evaluate the spatial relationship between objects and their relative position changes over time. 

Video content:{content}
 
Requirements:
- Generate exactly 5 questions
- Each question should be answerable with yes/no and the answer of every question should be yes.
- Focus on spatial relationships and relative position changes
- Questions should be specific to the video content
- Format as a JSON list of strings
""",
            "alignment_attribute_control": """
You are an expert in video analysis focusing on dynamic attributes and object transformations.
Analyze the following video content and generate 5 specific yes/no questions that evaluate whether objects in the video show dynamic changes in their attributes (color, size, shape, texture, state, etc.).

Video content: {content}

Requirements:
- Generate exactly 5 questions
- Each question should be answerable with yes/no
- Focus on temporal changes and object transformations
- Questions should be specific to the video content
- Format as a JSON list of strings
""",
            "alignment_event_control": """
You are an expert in story analysis and plot understanding.
Analyze the following video content and generate 10 specific yes/no questions that evaluate the story event, character actions, and narrative progression in chronological order.

Video content: {content}

Requirements:
- Generate exactly 10 questions
- Each question should be answerable with yes/no
- Questions should follow chronological order of events
- Focus on story elements, character actions, and plot development
- Questions should be specific to the video content
- Format as a JSON list of strings
""",
            "alignment_scene_control": """
You are an expert in scene analysis focusing on complex landscapes and environments.
Analyze the following video caption and generate 10 specific yes/no questions that evaluate the detailed content of the landscape and environment of the video caption in time order.

Video content: {content}

Requirements:
- Generate exactly 10 questions
- Each question should be answerable with yes/no and the answer of every question should be yes.
- Focus on detailed landscape and environment elements
- Raise questions about the landscape and scene content of the video caption in time order
- Format as a JSON list of strings.
""",
            "alignment_motion_control": """
You are an expert in motion analysis and temporal motion sequence understanding.
Analyze the following video caption and generate 10 specific yes/no questions that evaluate the temporal order and sequence of motions described in the video caption.

Video content: {content}
Requirements:
- Generate exactly 10 questions
- Each question should be answerable with yes/no and the answer of every question should be yes.
- Focus on temporal order and sequence of motions.
- Raise specific questions about the existing motions in the video to validate whether the motions in the video are consistent with those described in the caption in time order.
- Format as a JSON list of strings.

""",
            "consistency_motion_qa_orin": """
You are an expert in physics and spatiotemporal reasoning, with deep knowledge of real-world motion and dynamics.

Analyze the following video content and generate 5 specific yes/no questions that evaluate whether the motions depicted are physically rational and consistent.

Video content: {content}

Requirements:
- Generate exactly 5 questions
- Each question should be answerable with yes/no
- Focus on motion rationality: trajectory continuity, speed consistency, interaction plausibility
- Questions should test whether motions follow real-world physics
- Questions should be specific to the video content
- Format as a JSON list of strings
""",
            "consistency_motion_qa": """
You are an expert in physics and spatiotemporal reasoning, with deep knowledge of real-world motion and dynamics in 4D space (3D + time).

Evaluation Goal: Assess the motion consistency of a 4D generation model, focusing on whether object and scene dynamics evolve plausibly over time.

Core Evaluation Principles for Motion Consistency:
1. Temporal Continuity: Are object trajectories and transformations smooth over time, without temporal flickering or abrupt discontinuities?
2. Inter-object Interaction: Are interactions (e.g., collisions, pushes, pulls) physically reasonable and temporally aligned?
3. Speed and Acceleration Coherence: Are velocity and acceleration patterns consistent with the object's mass, size, and environment?
4. Scene-wide Consistency: Do all objects in the scene obey coherent motion logic, including global camera motion if present?

Analyze the following video content and generate 5 specific yes/no questions that evaluate these motion rationality principles:

Video content: {content}

Requirements:
- Generate exactly 5 questions covering the above principles
- Each question should be answerable with yes/no
- Questions must assess physical realism and motion logic
- Focus on detecting unrealistic physics violations
- Questions should be specific to the video content
- Format as a JSON list of strings
""",
            "physics_realism": """
You are a scientist who designs diagnostic yes/no questions about short real‑world scenarios for physics evaluation.

Core Evaluation Principles for Physics Reasoning:
1. Fundamental Physics: Core physical principles such as energy conservation, causality, equilibrium, and state transitions
2. Temporal & Spatial Rationality: The ability to interpret spatial relations and temporal sequences in physical events
3. Physical Event Reasoning: Comprehension of structured, goal-driven sequences of physical actions and their outcomes
4. Force & Motion: Physical interactions governed by forces, including motion, pushing, pulling, lifting, and inertia properties
5. Optics: Light behavior, reflection, refraction, shadows, and visual phenomena
6. Material Interaction & Transformation: How materials respond to forces, including melting, freezing, breaking, or chemical change
7. Thermal & Phase Transition: Temperature-related phenomena, heat transfer, and state changes between solid, liquid, and gas

Generate 10 yes/no questions based on the video content, covering the above physics dimensions:

Video content: {content}

Requirements:
- Generate exactly 10 questions covering physics reasoning principles
- Each question should be answerable with yes/no based on observable phenomena
- Questions should test whether the video follows real-world physics laws
- Focus on what should happen in the real world based on the described scenario
- Use present tense, neutral tone, and end each question with '(yes or no)'
- Questions should be answerable 'yes' if the scene follows real-world physics
- Format as a JSON list of strings
""",
        }

    def generate_questions_for_dimension(self, content: str, dimension: str) -> list[str]:
        """Generate questions for specific dimension (with caching support)"""
        # First check cache
        if self.cache_manager:
            cached_questions = self.cache_manager.get_questions(content, dimension)
            if cached_questions:
                return cached_questions

        # Cache miss, generate new questions
        questions = self._generate_new_questions(content, dimension)

        # Cache newly generated questions
        if self.cache_manager and questions:
            self.cache_manager.cache_questions(content, dimension, questions)

        return questions

    def _generate_new_questions(self, content: str, dimension: str) -> list[str]:
        """Generate new questions (without using cache)"""
        print(f"Generating new questions: Calling API to generate questions for dimension '{dimension}'...")

        # Select appropriate prompt template
        if dimension.lower() in self.dimension_prompts:
            prompt_template = self.dimension_prompts[dimension.lower()]
            prompt = prompt_template.format(content=content)
        else:
            prompt_template = self.dimension_prompts["default"]
            prompt = prompt_template.format(content=content, dimension=dimension)

        try:
            response = self.client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that only outputs valid JSON format. Return the questions as a JSON array of strings.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            content_response = response.choices[0].message.content
            content_response = content_response.strip().lstrip("```json").rstrip("```").strip()  # noqa: B005 - Character-set strip of the Markdown fence; retain upstream parsing.

            # Try to parse JSON
            try:
                questions = json.loads(content_response)
                if isinstance(questions, list):
                    print(f"API successfully generated {len(questions)} questions")
                    return questions
                else:
                    print(f"Warning: Generated content is not a list: {questions}")
                    return []
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw content: {content_response}")
                return []

        except Exception as e:
            print(f"API call failed: {e}")
            return []

    def process_json_file(self, json_file_path: str, dimension: str) -> bool:
        """Process JSON file, check and generate missing questions"""
        try:
            with open(json_file_path, encoding="utf-8") as f:
                data = json.load(f)

            modified = False

            # Process data
            if isinstance(data, list):
                # VBench format JSON array
                for item in data:
                    if self._needs_questions(item):
                        questions = self._generate_questions_for_item(item, dimension)
                        if questions:
                            item["auxiliary_info"] = questions
                            modified = True
                            print(f"Generated {len(questions)} questions for video")
                        else:
                            print("Question generation failed, skipping this item")
                            return False
            elif isinstance(data, dict):
                # Other JSON object formats
                for key, item in data.items():
                    if self._needs_questions(item):
                        questions = self._generate_questions_for_item(item, dimension)
                        if questions:
                            item["auxiliary_info"] = questions
                            modified = True
                            print(f"Generated {len(questions)} questions for {key}")
                        else:
                            print("Question generation failed, skipping this item")
                            return False

            # Save file if modified
            if modified:
                with open(json_file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"File updated: {json_file_path}")
                return True
            else:
                print("JSON file already has sufficient questions, no need to generate")
                return False

        except Exception as e:
            print(f"Error processing JSON file: {e}")
            return False

    def _needs_questions(self, item: dict[str, Any]) -> bool:
        """Check if item needs question generation"""
        auxiliary_info = item.get("auxiliary_info", [])
        return not auxiliary_info or len(auxiliary_info) == 0

    def _generate_questions_for_item(self, item: dict[str, Any], dimension: str) -> list[str]:
        """Generate questions for a single item"""
        # Try to extract content from item for question generation
        content = ""
        # breakpoint()
        # Try different content fields
        if "prompt_en" in item:
            content = item["prompt_en"]
        elif "prompt" in item:
            content = item["prompt"]
        elif "caption" in item:
            content = item["caption"]
        elif "description" in item:
            content = item["description"]
        else:
            # If no text content, use video path
            video_list = item.get("video_list", [])
            if video_list:
                content = f"Video file: {video_list[0] if isinstance(video_list, list) else video_list}"
        # breakpoint()
        if content:
            # Try to generate questions
            questions = self.generate_questions_for_dimension(content, dimension)
            if questions:
                return questions
            else:
                print(f"Error: Unable to generate questions for dimension '{dimension}', API call failed")
                return []
        else:
            print(f"Error: Unable to generate questions for dimension '{dimension}', missing content description")
            return []


def main():
    parser = argparse.ArgumentParser(description="Adaptive question generator")
    parser.add_argument("--json_file", required=True, help="Path to JSON file to process")
    parser.add_argument("--dimension", required=True, help="Evaluation dimension")
    parser.add_argument("--api_base_url", help="OpenAI API base URL")
    parser.add_argument("--api_key", help="OpenAI API key")
    parser.add_argument("--migrate_cache", action="store_true", help="Migrate existing questions to cache system")
    parser.add_argument("--cache_stats", action="store_true", help="Display cache statistics")

    args = parser.parse_args()

    # Create question generator
    generator = AdaptiveQAGenerator(args.api_base_url, args.api_key)

    # Handle cache-related operations
    if args.cache_stats and generator.cache_manager:
        stats = generator.cache_manager.get_cache_stats(args.dimension)
        print(f"Cache statistics for dimension '{args.dimension}':")
        print(f"  - Cached items: {stats['total_cached_items']}")
        print(f"  - Cached questions: {stats['total_cached_questions']}")
        print(f"  - Cache file: {stats['cache_file']}")
        return

    if args.migrate_cache and generator.cache_manager:
        migrated = generator.cache_manager.migrate_existing_questions(args.json_file, args.dimension)
        print(f"Migration complete: {migrated} questions migrated to cache")
        return

    # Process JSON file
    success = generator.process_json_file(args.json_file, args.dimension)

    if success:
        print("Question generation complete!")
    else:
        print("Processing complete, no items need question generation.")


if __name__ == "__main__":
    main()
