"""Local checks for the default assets required in every Vulkan release."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location('vulkan_package', Path(__file__).with_name('package.py'))
assert spec is not None and spec.loader is not None
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class DefaultPrefillPackagingTests(unittest.TestCase):
    def test_changed_template_is_rejected(self):
        source = package.TOOLS / 'astudio_template'
        original = package.load_default_template(source)
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary) / 'astudio_template'
            shutil.copytree(source, assets)
            (assets / 'template.txt').write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                package.load_default_template(assets)

    def make_archive(self, directory, omit_prefix=False, corrupt_prefix=False, omit_logo=False):
        version = 'package-check'
        paths = [*package.LAUNCHERS, *package.BRANDING_FILES,
                 'README.zh-CN.md', 'PACKAGING.zh-CN.md',
                 *(f'astudio_template/{name}' for name in package.TEMPLATE_FILES)]
        data = {name: (package.TOOLS / name).read_bytes() for name in paths}
        if omit_prefix:
            del data['astudio_template/template.txt']
        if omit_logo:
            del data[package.BRANDING_FILES[0]]
        manifest = {'version': version, 'files': [
            {'path': name, 'bytes': len(content), 'sha256': package.digest(content)}
            for name, content in data.items()]}
        if corrupt_prefix:
            data['astudio_template/template.txt'] += b'changed'
        archive = directory / 'package.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for name, content in data.items():
                bundle.writestr(f'{version}/{name}', content)
            bundle.writestr(f'{version}/version.json', json.dumps(manifest))
        return archive, version, copy.deepcopy(manifest)

    def test_zip_contains_verified_startup_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, version, manifest = self.make_archive(Path(temporary))
            package.verify_archive(archive, version, manifest)

    def test_manifest_cannot_omit_default_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, version, manifest = self.make_archive(Path(temporary), omit_prefix=True)
            with self.assertRaisesRegex(ValueError, 'default prefill assets'):
                package.verify_archive(archive, version, manifest)

    def test_zip_with_valid_crc_but_changed_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, version, manifest = self.make_archive(Path(temporary), corrupt_prefix=True)
            with self.assertRaisesRegex(ValueError, 'Final ZIP file mismatch'):
                package.verify_archive(archive, version, manifest)

    def test_manifest_cannot_omit_logo(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive, version, manifest = self.make_archive(Path(temporary), omit_logo=True)
            with self.assertRaisesRegex(ValueError, 'branding'):
                package.verify_archive(archive, version, manifest)


if __name__ == '__main__':
    unittest.main()
