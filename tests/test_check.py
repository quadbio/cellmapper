import packaging
import pytest

from cellmapper import check
from cellmapper.check import Checker, check_deps


class TestCheck:
    def test_checker_available_module(self):
        # Should not raise for a real installable package
        Checker("packaging").check()

    def test_checker_missing_module(self):
        # Should raise RuntimeError for a missing module
        with pytest.raises(RuntimeError):
            Checker("not_a_real_module").check()

    def test_checker_version_requirement(self):
        # Should raise if vmin is higher than installed version
        installed_version = packaging.version.parse(packaging.__version__)
        higher_version = str(installed_version.major + 1) + ".0.0"
        with pytest.raises(RuntimeError):
            Checker("packaging", vmin=higher_version).check()

    def test_check_deps_missing(self):
        # Should raise for a missing dependency (not registered in CHECKERS)
        with pytest.raises(RuntimeError):
            check_deps("not_a_real_module")

    def test_check_deps_available(self):
        # Should not raise for a real installable package
        check.CHECKERS["packaging"] = Checker("packaging")
        check_deps("packaging")
        del check.CHECKERS["packaging"]

    def test_checker_missing_metadata_no_vmin(self, mocker):
        # Should not raise when package is importable but has no metadata and no vmin
        mocker.patch("importlib.import_module")
        # No vmin means no version check needed
        Checker("fake_module").check()

    def test_checker_missing_metadata_with_vmin(self, mocker):
        # Should not raise when package is importable but has no metadata (conda packages)
        # even when vmin is specified - we skip version check in this case
        from importlib.metadata import PackageNotFoundError

        mocker.patch("importlib.import_module")
        mocker.patch("cellmapper.check.version", side_effect=PackageNotFoundError("fake_module"))
        # Should succeed even with vmin - version check is skipped for missing metadata
        Checker("fake_module", vmin="1.0.0").check()
