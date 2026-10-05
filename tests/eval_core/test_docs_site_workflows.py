from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
DEPLOY_PATH = WORKFLOWS / "deploy-docs.yml"
BUILD_PATH = WORKFLOWS / "docs-build.yml"
ACTION_SHA_RE = re.compile(r"uses:\s+[^\s]+@[0-9a-f]{40}\b")
FLOATING_ACTION_RE = re.compile(r"uses:\s+[^\s]+@v\d")

GENERATED_SITE_PATHS = {
    "docs/**",
    "scripts/docs/**",
    "worldfoundry/data/models/**",
    "worldfoundry/data/benchmarks/**",
    "worldfoundry/evaluation/api/**",
}


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _on_block(payload: dict) -> dict:
    # PyYAML 1.1 treats the unquoted GitHub Actions key `on` as boolean true.
    return payload.get("on", payload.get(True, {}))


def test_docs_workflows_cover_generated_site_sources() -> None:
    deploy = _workflow(DEPLOY_PATH)
    build = _workflow(BUILD_PATH)
    deploy_paths = set(_on_block(deploy)["push"]["paths"])
    build_paths = set(_on_block(build)["pull_request"]["paths"])

    assert GENERATED_SITE_PATHS <= deploy_paths
    assert GENERATED_SITE_PATHS <= build_paths
    assert ".github/workflows/deploy-docs.yml" in deploy_paths
    assert {".github/workflows/deploy-docs.yml", ".github/workflows/docs-build.yml"} <= build_paths


def test_deploy_docs_scopes_permissions_and_bounds_jobs() -> None:
    payload = _workflow(DEPLOY_PATH)
    text = DEPLOY_PATH.read_text(encoding="utf-8")
    assert payload["permissions"] == {"contents": "read"}
    assert payload["concurrency"] == {"group": "github-pages", "cancel-in-progress": False}
    assert "Drop demos from Pages artifact and add .nojekyll" in text
    assert "Preserve Next.js assets" not in text
    assert "--skip-bootstrap" not in text

    build = payload["jobs"]["build"]
    assert build["timeout-minutes"] == 30
    assert build["permissions"] == {"contents": "read", "pages": "write"}
    assert "id-token" not in build["permissions"]

    deploy = payload["jobs"]["deploy"]
    assert deploy["timeout-minutes"] == 10
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}


def test_deploy_docs_derives_site_url_without_rewriting_demo_asset_ref() -> None:
    payload = _workflow(DEPLOY_PATH)
    env = payload["jobs"]["build"]["env"]

    assert env["NEXT_PUBLIC_SITE_URL"] == "https://${{ github.repository_owner }}.github.io"
    assert "${{ github.ref_name }}" in env["NEXT_PUBLIC_DEMO_ASSET_BASE_URL"]
    assert "default_branch" not in env["NEXT_PUBLIC_DEMO_ASSET_BASE_URL"]


def test_docs_build_is_read_only_and_never_deploys() -> None:
    payload = _workflow(BUILD_PATH)
    text = BUILD_PATH.read_text(encoding="utf-8")

    assert payload["permissions"] == {"contents": "read"}
    assert payload["jobs"]["build"]["timeout-minutes"] == 30
    assert payload["concurrency"]["cancel-in-progress"] is True
    assert "bash scripts/docs/build.sh" in text
    assert "--skip-bootstrap" not in text
    assert "upload-pages-artifact" not in text
    assert "deploy-pages" not in text
    assert "pages: write" not in text
    assert "id-token: write" not in text


def test_docs_workflows_pin_all_actions_to_full_shas() -> None:
    for path in (DEPLOY_PATH, BUILD_PATH):
        text = path.read_text(encoding="utf-8")
        uses_lines = [line for line in text.splitlines() if line.strip().startswith("uses:")]
        assert uses_lines, path.name
        assert all(ACTION_SHA_RE.search(line) for line in uses_lines), path.name
        assert not FLOATING_ACTION_RE.search(text), path.name


def test_docs_workflows_use_the_checked_in_node_version() -> None:
    for path in (DEPLOY_PATH, BUILD_PATH):
        text = path.read_text(encoding="utf-8")
        assert "node-version-file: docs/fumadocs/.nvmrc" in text
        assert "node-version:" not in text


def test_codeowners_covers_docs_workflows_and_generated_data() -> None:
    text = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")

    for pattern in (".github/workflows/", "docs/", "scripts/docs/", "worldfoundry/data/"):
        assert f"{pattern} @dhsaikhgdius" in text
