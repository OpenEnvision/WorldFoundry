class OptionalDependencyNotInstalled(Exception):
    pass


def handle_module_not_found_error(e: ModuleNotFoundError, suggestions: list[str] | None = None):
    extras = ", ".join(suggestions or ["worldscore"])
    raise OptionalDependencyNotInstalled(
        f"Optional dependency {e.name} is not installed for WorldFoundry WorldScore metrics. "
        f"Install the documented WorldFoundry evaluation environment for: {extras}."
    ) from e
