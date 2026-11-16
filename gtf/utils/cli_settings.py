"""Advanced settings of a command-line tool behind one option: --set KEY=VALUE.

A stage script keeps a dedicated flag for what people and drivers actually pass. Every other knob is a named default in
one table of the script (SETTINGS) and is changed with `--set key=value` (repeatable; `-` and `_` are interchangeable in
the key). After `apply_settings` the values are ordinary attributes of the parsed arguments, exactly as if each had its
own flag.
"""

from typing import Any, NamedTuple


class Setting(NamedTuple):
    default: Any
    kind: Any  # bool, int, float or str: how the text after `=` is read
    help: str = ""
    choices: tuple | None = None


_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}


def _read(text, setting, key):
    low = text.strip().lower()
    if low in ("none", "null") and setting.default is None:
        return None
    if setting.kind is bool:
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"{key}: expected true or false, got {text!r}")
    value = setting.kind(text)
    if setting.choices is not None and value not in setting.choices:
        raise ValueError(f"{key}: {value!r} is not one of {list(setting.choices)}")
    return value


def add_settings(parser, settings):
    """the --set option; call before parse_args"""
    parser.add_argument(
        "--set",
        dest="settings",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="advanced setting, repeatable. Keys: " + ", ".join(sorted(settings)) + " (defaults and meaning: SETTINGS in this script)",
    )


def apply_settings(parser, args, settings):
    """put every setting on `args`: its default, or the value given with --set"""
    for key, setting in settings.items():
        setattr(args, key, setting.default)
    for item in getattr(args, "settings", None) or []:
        key, sep, text = item.partition("=")
        key = key.strip().lstrip("-").replace("-", "_")
        if not sep or key not in settings:
            parser.error(f"--set {item!r}: unknown setting. Known: {', '.join(sorted(settings))}")
        try:
            setattr(args, key, _read(text, settings[key], key))
        except ValueError as e:
            parser.error(f"--set {e}")
    return args
