"""Exercise the packaged ACL entry point outside the source checkout."""
from pathlib import Path
import shutil
import subprocess
import tomllib
import sys


def test_installed_wheel_protocol(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    config = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))
    for name in ("pyproject.toml", config["project"]["readme"]):
        shutil.copy2(root / name, source / name)
    for package in config["tool"]["setuptools"]["packages"]["find"]["include"]:
        shutil.copytree(root / package, source / package, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                    "--no-build-isolation", "-w", str(tmp_path), str(source)],
                   check=True, cwd=tmp_path)
    wheel, = tmp_path.glob("*.whl")
    target = tmp_path / "installed"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                    "--target", str(target), str(wheel)], check=True, cwd=tmp_path)
    check = """
from pathlib import Path
import sys
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
# An editable developer checkout can otherwise mask wheel packaging defects.
class ExcludeCheckoutScripts:
    def find_spec(self, fullname, *args):
        if fullname == 'scripts' or fullname.startswith('scripts.'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, ExcludeCheckoutScripts())
from qb_data import acl_protocol
assert Path(acl_protocol.__file__).resolve().is_relative_to(target)
assert acl_protocol.digest({}) == '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'
assert 'question_prefix' in acl_protocol.policy_prompt('prefix', ['a', 'b'])
"""
    subprocess.run([sys.executable, "-I", "-c", check, str(target)], check=True, cwd=tmp_path)
