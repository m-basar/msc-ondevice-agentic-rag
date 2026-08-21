"""Constant metadata for generated figures, so that a regeneration is checkable.

Amendment 1.30.7. Amendment 1.26 called the figures reproducible. They were not
byte-reproducible: matplotlib stamps its own version into the metadata of both
the PNG and the SVG, so the same script over the same data under a different
matplotlib produced different bytes while this repository claimed otherwise.
Suppressing the SVG creation date under 1.26.6 was necessary and not sufficient.

Two things are done here and they are different in kind.

**The tool's identity is replaced with the project's.** A figure records what
produced it, which is this repository at a stated amendment, not whichever
matplotlib happened to be installed. That makes two regenerations comparable.

**The environment is recorded rather than hidden.** Removing the version from
the file does not make the rendering independent of it: FreeType and the font
stack decide where glyphs land, and Pillow writes the PNG bytes, so the same
script on a different machine can produce a different file. ``write_environment`` writes the versions beside the
figures so that a reader can tell which machine drew them, and so that a
regeneration producing different pixels can be explained rather than argued
about.
"""

from __future__ import annotations

import json
import platform
import re
import sys
import zlib
from pathlib import Path

#: What the figures record as their producer. Constant by design: it names the
#: repository and the amendment governing figure generation, and it does not
#: move when a dependency is upgraded.
CREATOR = ("sme-assistant figure generator, "
           "docs/PREREGISTRATION.md amendment 1.30.7")

#: Passed to ``savefig`` for PNG. Overrides matplotlib's default ``Software``
#: value, which carries its version string.
PNG_METADATA: dict[str, str] = {"Software": CREATOR}

#: Passed to ``savefig`` for SVG. ``Date: None`` suppresses the creation stamp,
#: which would otherwise differ on every run.
SVG_METADATA: dict[str, object] = {"Date": None, "Creator": CREATOR}

#: Any residual "matplotlib 3.10.9" or "matplotlib version3.10.9" left in a
#: text-format figure. SVG is plain text and can be rewritten safely; PNG
#: carries chunk lengths and CRCs and must not be, which is why the PNG path
#: sets its metadata at write time instead and a test checks the result.
_VERSION = re.compile(rb"matplotlib[ ]*(?:version)?[ ]*v?\d+\.\d+(?:\.\d+)?",
                      re.IGNORECASE)


def png_text(path: Path) -> dict[str, str]:
    """Read a PNG's textual metadata using the standard library only.

    Pillow reads this in one line, and Pillow is in the optional ``figures``
    dependency group. A test that needs it either skips on a machine without it
    or fails there, and both happened: the Raspberry Pi 5 has neither Pillow nor
    matplotlib, and two screenshot tests imported Pillow unconditionally while a
    third was gated on matplotlib while importing Pillow.

    A skipped check is a check that did not run. Parsing the chunks here costs
    thirty lines and lets the producer metadata be verified on every machine,
    which is where the value of the check actually is.

    Handles ``tEXt``, ``zTXt`` and ``iTXt``. Returns whatever it can read and
    raises only on a file that is not a PNG at all.
    """
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name} is not a PNG")
    out: dict[str, str] = {}
    offset = 8
    while offset + 8 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        offset += 12 + length  # length, type, payload, CRC
        if kind == b"IEND":
            break
        try:
            if kind == b"tEXt":
                keyword, _, value = payload.partition(b"\x00")
                out[keyword.decode("latin-1")] = value.decode("latin-1")
            elif kind == b"zTXt":
                keyword, _, rest = payload.partition(b"\x00")
                out[keyword.decode("latin-1")] = zlib.decompress(
                    rest[1:]).decode("latin-1")
            elif kind == b"iTXt":
                keyword, _, rest = payload.partition(b"\x00")
                compressed, _method, rest = rest[0], rest[1], rest[2:]
                _language, _, rest = rest.partition(b"\x00")
                _translated, _, text = rest.partition(b"\x00")
                if compressed:
                    text = zlib.decompress(text)
                out[keyword.decode("latin-1")] = text.decode("utf-8")
        except (zlib.error, UnicodeDecodeError, IndexError):
            # A chunk this cannot read is reported as absent rather than
            # crashing the caller; the tests assert on presence and value.
            continue
    return out


def scrub_svg(path: Path) -> bool:
    """Replace any matplotlib version string in an SVG with a bare name.

    Returns whether anything changed, so a caller can say so rather than
    silently depending on it.
    """
    data = path.read_bytes()
    scrubbed = _VERSION.sub(b"matplotlib", data)
    if scrubbed == data:
        return False
    path.write_bytes(scrubbed)
    return True


def carries_a_version(path: Path) -> bool:
    """Whether a written figure still names a specific matplotlib version."""
    return bool(_VERSION.search(path.read_bytes()))


def environment() -> dict[str, str]:
    """The versions that decide what a regeneration looks like."""
    import matplotlib

    versions = {
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "matplotlib_backend": matplotlib.get_backend(),
    }
    try:  # FreeType decides glyph rasterisation and therefore PNG pixels.
        from matplotlib import ft2font

        versions["freetype"] = ft2font.__freetype_version__
    except Exception:  # noqa: BLE001 - recorded as unknown, never fatal
        versions["freetype"] = "unknown"
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001
        versions["numpy"] = "unknown"
    try:  # Pillow writes the PNG files, so it decides their bytes.
        import PIL

        versions["pillow"] = PIL.__version__
    except Exception:  # noqa: BLE001
        versions["pillow"] = "unknown"
    return versions


def write_environment(directory: Path) -> Path:
    """Record the figure-generation environment beside the figures.

    Written with ``newline="\\n"`` and sorted keys, for the same reason the
    analysis outputs are: a file generated on Windows that differs from the same
    file generated on Linux only in line endings is a diff a reader has to
    discount by hand.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "FIGURE_ENVIRONMENT.json"
    payload = {
        "creator": CREATOR,
        "note": (
            "Recorded, not pinned by this file. Figure metadata is constant so "
            "that two runs on one machine are byte-identical; rendering still "
            "depends on the versions below, so a regeneration elsewhere may "
            "differ in pixels and this states what drew the committed images."
        ),
        "versions": environment(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def main() -> int:
    """Report the current environment, for checking a machine before a rerun."""
    json.dump(environment(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
