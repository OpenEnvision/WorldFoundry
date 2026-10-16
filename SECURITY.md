# Security Policy

## Supported Versions

Security fixes target the current `main` branch and are included in subsequent
releases. Older revisions do not receive independent security backports.
Use a current checkout and the dependencies specified by its runtime profiles.

## Reporting a Vulnerability

Report potential vulnerabilities privately. Use **Report a vulnerability** on
[the repository Security page](https://github.com/OpenEnvision/WorldFoundry/security)
when that option is available. Otherwise, contact a maintainer listed in
[CODEOWNERS](.github/CODEOWNERS) through the community Slack or Discord linked in
[README.md](README.md#-community) to arrange a private reporting channel.

Include:

- The affected revision, component, and dependency versions.
- Minimal reproduction steps and the observed impact.
- Relevant configuration, logs, or a proof of concept, with credentials and
  personal data removed.

Keep vulnerability details out of public issues and pull requests until
maintainers have assessed the report and coordinated disclosure. Ordinary bugs
and installation problems belong in the
[issue tracker](https://github.com/OpenEnvision/WorldFoundry/issues).

Maintainers coordinate investigation, fixes, disclosure, and reporter credit.
Report upstream-only vulnerabilities to the upstream project as well, and
identify any affected WorldFoundry integration in your report.

## Runtime Safety

WorldFoundry is designed for cooperating operators running models on their own
machines. Keep Studio and preview services bound to localhost unless access is
controlled; use the documented authentication options for remote access.

Only execute trusted runtime code, configuration files, and model checkpoints.
Some upstream loaders use formats that can execute code. Prefer the documented
safe checkpoint formats, keep API credentials outside Git and run artifacts,
and review upstream licenses and runtime requirements before loading assets.
