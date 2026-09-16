# Security policy

## Reporting a vulnerability

Please report potential security issues privately to
[valentin.bacher@cs.ox.ac.uk](mailto:valentin.bacher@cs.ox.ac.uk). Do not open
a public issue for a suspected vulnerability or exposed credential.

## Deployment guidance

The Gradio application accepts uploaded files and writes each web run to a new
temporary directory by default. Server-side input paths and custom output
directories are disabled by default because they are unsafe on a public
deployment. They may be enabled only on a trusted local deployment with:

```bash
RFLASH_ENABLE_SERVER_PATHS=1 python app.py
```
