# CLI Reference

`cv-benchkit` is a computer-vision benchmark toolkit for converting and validating datasets.

```bash
cv-benchkit [-h] {convert,validate} ...
```

| Subcommand | Description |
|------------|-------------|
| `convert`  | Convert between dataset formats. |
| `validate` | Validate a COCO JSON file. |

A subcommand is required. Use `cv-benchkit --help` or `cv-benchkit <subcommand> --help` to see built-in help.

---

## `cv-benchkit convert`

Convert annotations between COCO, YOLO and Pascal VOC formats.

```bash
cv-benchkit convert --from {coco,yolo,voc} --to {coco,yolo,voc} \
    --input INPUT_PATH --output OUTPUT_PATH \
    [--image-width IMAGE_WIDTH] [--image-height IMAGE_HEIGHT] \
    [--categories CATEGORIES [CATEGORIES ...]]
```

### Options

| Flag | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `--from` | `coco` \| `yolo` \| `voc` | Yes | — | Source format. |
| `--to` | `coco` \| `yolo` \| `voc` | Yes | — | Target format. |
| `--input` | path | Yes | — | Input file or directory. |
| `--output` | path | Yes | — | Output directory. |
| `--image-width` | int | YOLO source only | `None` | Image width (required for YOLO source). |
| `--image-height` | int | YOLO source only | `None` | Image height (required for YOLO source). |
| `--categories` | one or more strings | YOLO/VOC source only | `None` | Category names (required for YOLO/VOC source). |
| `-h`, `--help` | — | No | — | Show help and exit. |

### Behaviour

- `--from` and `--to` must differ; otherwise the command exits with code `1`.
- **COCO source**: `--input` is a COCO JSON file. It is validated, then written as YOLO `.txt` or VOC `.xml` files into `--output`.
- **YOLO source**: `--input` is a directory of YOLO `.txt` label files. `--categories`, `--image-width` and `--image-height` are required; the given width/height is applied to every image.
- **VOC source**: `--input` is a directory of VOC XML files. `--categories` is required.
- When the target is `coco`, the output is written to `<output>/annotations.json`.

### Examples

```bash
# COCO -> YOLO
cv-benchkit convert --from coco --to yolo --input annotations.json --output labels/

# YOLO -> COCO
cv-benchkit convert --from yolo --to coco --input labels/ --output out/ \
    --image-width 640 --image-height 480 --categories cat dog

# VOC -> COCO
cv-benchkit convert --from voc --to coco --input voc_xml/ --output out/ \
    --categories cat dog
```

---

## `cv-benchkit validate`

Validate a COCO JSON file.

```bash
cv-benchkit validate FILE
```

### Arguments

| Argument | Description |
|----------|-------------|
| `FILE` | Path to the COCO JSON file. |
| `-h`, `--help` | Show help and exit. |

On success, prints a summary such as:

```
Valid COCO dataset: 10 images, 42 annotations, 3 categories.
```

If the file does not exist or fails validation, an error is printed to stderr and the command exits with code `1`.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success. |
| `1` | Error (invalid arguments, missing file, or validation failure). |
| `2` | Argument parsing error (raised by `argparse`, e.g. missing required flag). |
