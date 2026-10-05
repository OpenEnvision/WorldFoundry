from worldfoundry.studio.serving.workspace import (
    _normalize_official_links,
    _official_links_from_catalog_entry,
    _official_links_from_sources,
)


def test_official_links_read_technical_report_and_homepage() -> None:
    links = _official_links_from_sources(
        {
            "github": {"url": "https://github.com/example/model"},
            "technical_report": "https://example.com/paper/model.pdf",
            "homepage": "https://example.com/model",
        }
    )
    assert links["github"] == "https://github.com/example/model"
    assert links["paper"] == "https://example.com/paper/model.pdf"
    assert links["project"] == "https://example.com/model"


def test_catalog_entry_promotes_non_github_official_url_to_project() -> None:
    links = _official_links_from_catalog_entry(
        {
            "id": "hosted-demo",
            "source": {"official_repo_url": "https://runwayml.com/"},
            "official_sources": {"project_page": "https://runwayml.com/"},
        }
    )
    assert "github" not in links
    assert links["project"] == "https://runwayml.com/"


def test_normalize_keeps_github_host() -> None:
    assert _normalize_official_links(
        {"github": "https://github.com/foo/bar", "project": "https://foo.github.io/bar"}
    ) == {
        "github": "https://github.com/foo/bar",
        "project": "https://foo.github.io/bar",
    }
