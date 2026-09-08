# RAW_DATA

Source PDFs go here. **The PDFs themselves are git-ignored** — they are large and
copyrighted, so each machine keeps its own copy.

The directory is scanned recursively for `*.pdf`, so subfolders are fine.

Current corpus on the development machine:

| File | Size |
|---|---|
| `Joint Formulary Committee - British National Formulary (BNF85)...pdf` | 18 MB |
| `[Oxford Medical Handbooks] ... Oxford Handbook of Clinical Medicine ...pdf` | 31 MB |
| `B09474-eng.pdf` | 1.3 MB |

After copying PDFs here, ingest them:

```bash
python scripts/ingest.py --recreate
```
