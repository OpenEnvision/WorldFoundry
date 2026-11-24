#!/usr/bin/env python3
"""
Question cache manager
Implement question caching and reuse based on content hashing, ensuring same questions for same content
"""

import hashlib
import json
from pathlib import Path
from typing import Any


class QuestionCacheManager:
    def __init__(self, cache_dir: str = "4DWorldBench/dimension_description_json/cache"):
        """
        Initialize cache manager

        Args:
            cache_dir: Cache directory path
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_content_hash(self, content: str) -> str:
        """
        Generate hash value of content

        Args:
            content: Text content

        Returns:
            str: SHA256 hash value of content
        """
        # Normalize content: remove extra whitespace, convert to lowercase
        normalized_content = " ".join(content.strip().lower().split())
        return hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()[:16]  # Use first 16 characters

    def _get_cache_file_path(self, dimension: str) -> Path:
        """
        Get cache file path for dimension

        Args:
            dimension: Evaluation dimension

        Returns:
            Path: Cache file path
        """
        return self.cache_dir / f"{dimension}_cache.json"

    def _load_cache(self, dimension: str) -> dict[str, Any]:
        """
        Load cache data for dimension

        Args:
            dimension: Evaluation dimension

        Returns:
            Dict: Cache data
        """
        cache_file = self._get_cache_file_path(dimension)
        if cache_file.exists():
            try:
                with open(cache_file, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Unable to load cache file {cache_file}: {e}")
                return {}
        return {}

    def _save_cache(self, dimension: str, cache_data: dict[str, Any]) -> None:
        """
        Save cache data for dimension

        Args:
            dimension: Evaluation dimension
        """
        cache_file = self._get_cache_file_path(dimension)
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"Warning: Unable to save cache file {cache_file}: {e}")

    def get_questions(self, content: str, dimension: str) -> list[str] | None:
        """
        Get cached questions

        Args:
            content: Content text
            dimension: Evaluation dimension

        Returns:
            Optional[List[str]]: List of cached questions, or None if not cached
        """
        content_hash = self._get_content_hash(content)
        cache_data = self._load_cache(dimension)

        if content_hash in cache_data:
            cached_item = cache_data[content_hash]
            print(f"Cache hit: Reusing generated questions for dimension '{dimension}' (hash: {content_hash[:8]}...)")
            return cached_item.get("questions", [])

        return None

    def cache_questions(self, content: str, dimension: str, questions: list[str]) -> None:
        """
        Args:
            content: Content text
            dimension: Evaluation dimension
            questions: List of questions
        """
        content_hash = self._get_content_hash(content)
        cache_data = self._load_cache(dimension)

        # Store cache item
        cache_data[content_hash] = {
            "content_preview": content[:100] + ("..." if len(content) > 100 else ""),  # Content preview
            "questions": questions,
            "question_count": len(questions),
            "created_at": __import__("datetime").datetime.now().isoformat(),
        }

        self._save_cache(dimension, cache_data)
        print(
            f"Cache saved: Cached {len(questions)} questions for dimension '{dimension}' (hash: {content_hash[:8]}...)"
        )

    def get_cache_stats(self, dimension: str) -> dict[str, Any]:
        """
        Get cache statistics

        Args:
            dimension: Evaluation dimension

        Returns:
            Dict: Cache statistics
        """
        cache_data = self._load_cache(dimension)

        total_items = len(cache_data)
        total_questions = sum(item.get("question_count", 0) for item in cache_data.values())

        return {
            "dimension": dimension,
            "total_cached_items": total_items,
            "total_cached_questions": total_questions,
            "cache_file": str(self._get_cache_file_path(dimension)),
        }

    def list_all_cache_stats(self) -> list[dict[str, Any]]:
        """
        List cache statistics for all dimensions

        Returns:
            List[Dict]: Cache statistics for all dimensions
        """
        stats = []

        # Find all cache files
        for cache_file in self.cache_dir.glob("*_cache.json"):
            dimension = cache_file.stem.replace("_cache", "")
            stats.append(self.get_cache_stats(dimension))

        return stats

    def clear_cache(self, dimension: str | None = None) -> None:
        """
        Args:
            dimension: Dimension to clear, if None clear all caches
        """
        if dimension:
            cache_file = self._get_cache_file_path(dimension)
            if cache_file.exists():
                cache_file.unlink()
                print(f"Cleared cache for dimension '{dimension}'")
        else:
            # Clear all cache files
            for cache_file in self.cache_dir.glob("*_cache.json"):
                cache_file.unlink()
            print("Cleared all caches")

    def migrate_existing_questions(self, json_file_path: str, dimension: str) -> int:
        """
        Migrate existing questions to cache system

        Args:
            json_file_path: Path to existing JSON file
            dimension: Evaluation dimension

        Returns:
            int: Number of questions migrated
        """
        migrated_count = 0

        try:
            with open(json_file_path, encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    content = self._extract_content_from_item(item)
                    questions = item.get("auxiliary_info", [])

                    # Check if already cached
                    if content and questions and not self.get_questions(content, dimension):
                        self.cache_questions(content, dimension, questions)
                        migrated_count += 1

            print(f"Migration complete: Migrated {migrated_count} questions to cache for dimension '{dimension}'")

        except (OSError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Migration failed: {e}")

        return migrated_count

    def _extract_content_from_item(self, item: dict[str, Any]) -> str:
        """
        Extract content text from item

        Args:
            item: Data item

        Returns:
            str: Extracted content text
        """
        # Try different content fields
        for field in ["prompt_en", "prompt", "caption", "description"]:
            if item.get(field):
                return str(item[field])

        # If no text content, use video path
        video_list = item.get("video_list", [])
        if video_list:
            video_path = video_list[0] if isinstance(video_list, list) else video_list
            return f"Video file: {video_path}"

        return ""


def main():
    """Command line tool"""
    import argparse

    parser = argparse.ArgumentParser(description="Question cache manager")
    parser.add_argument("--action", choices=["stats", "clear", "migrate"], required=True, help="Operation type")
    parser.add_argument("--dimension", help="Dimension name")
    parser.add_argument("--json_file", help="Path to JSON file to migrate")

    args = parser.parse_args()

    cache_manager = QuestionCacheManager()

    if args.action == "stats":
        if args.dimension:
            stats = cache_manager.get_cache_stats(args.dimension)
            print(f"Cache statistics for dimension '{stats['dimension']}':")
            print(f"  - Cached items: {stats['total_cached_items']}")
            print(f"  - Cached questions: {stats['total_cached_questions']}")
            print(f"  - Cache file: {stats['cache_file']}")
        else:
            all_stats = cache_manager.list_all_cache_stats()
            print("Cache statistics for all dimensions:")
            for stats in all_stats:
                print(
                    f"  {stats['dimension']}: {stats['total_cached_items']} items, "
                    f"{stats['total_cached_questions']} questions"
                )

    elif args.action == "clear":
        cache_manager.clear_cache(args.dimension)

    elif args.action == "migrate":
        if not args.json_file or not args.dimension:
            print("Migration operation requires --json_file and --dimension parameters")
            return

        migrated = cache_manager.migrate_existing_questions(args.json_file, args.dimension)
        print(f"Migration complete: {migrated} questions")


if __name__ == "__main__":
    main()
