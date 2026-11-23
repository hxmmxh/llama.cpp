#!/usr/bin/env python3
"""Package a frozen native Windows Vulkan build after its startup checks."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile
from zoneinfo import ZoneInfo

TOOLS = Path(__file__).resolve().parent
REPOSITORY = TOOLS.parents[2]
WORKSPACE = REPOSITORY.parent
DEFAULT_OUTPUT = WORKSPACE / 'dist'
PROJECT_NAME = 'llama.cpp Spark Intel GPU Edition'
LAUNCHERS = ('run.ps1', 'run-cli.cmd', 'run-server.cmd', 'check-vulkan.cmd')
TEMPLATE_FILES = ('template.txt', 'template-info.json', 'README.txt')
BRANDING_FILES = ('assets/spark-llama-intel-gpu-logo.png',)
SYSTEM_DLLS = {
    'advapi32.dll', 'avrt.dll', 'bcrypt.dll', 'cfgmgr32.dll', 'combase.dll',
    'crypt32.dll', 'dbghelp.dll', 'dnsapi.dll', 'gdi32.dll', 'imm32.dll',
    'iphlpapi.dll', 'kernel32.dll', 'msvcrt.dll', 'ntdll.dll', 'ole32.dll',
    'oleaut32.dll', 'powrprof.dll', 'propsys.dll', 'psapi.dll', 'rpcrt4.dll',
    'secur32.dll', 'setupapi.dll', 'shell32.dll', 'shlwapi.dll', 'ucrtbase.dll',
    'user32.dll', 'userenv.dll', 'version.dll', 'winmm.dll', 'ws2_32.dll',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def record(path, root):
    return dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size,
                sha256=digest(path.read_bytes()))


def find_readobj(requested=None):
    if requested is not None:
        candidate = Path(shutil.which(str(requested)) or requested).expanduser().resolve()
        if not candidate.is_file():
            raise ValueError(f'llvm-readobj not found: {requested}')
        return candidate
    executable = shutil.which('llvm-readobj')
    if executable:
        return Path(executable)
    for candidate in sorted(WORKSPACE.glob('toolchain/llvm-mingw-*/bin/llvm-readobj'), reverse=True):
        if candidate.is_file():
            return candidate
    raise ValueError('llvm-readobj not found; specify --readobj PATH')


def load_default_template(directory):
    for name in TEMPLATE_FILES:
        if not (directory / name).is_file():
            raise ValueError(f'Missing default prefill asset: {directory / name}')
    info = read_json(directory / 'template-info.json')
    data = (directory / 'template.txt').read_bytes()
    if not data.decode('utf-8').strip():
        raise ValueError('Default prompt template is empty')
    if len(data) != info['bytes'] or digest(data) != info['sha256']:
        raise ValueError('Default prompt template identity changed')
    return data


def verify_archive(archive, version, manifest):
    prefix = version + '/'
    required = {prefix + name for name in LAUNCHERS}
    required.update(prefix + 'astudio_template/' + name for name in TEMPLATE_FILES)
    required.add(prefix + 'README.zh-CN.md')
    required.add(prefix + 'PACKAGING.zh-CN.md')
    required.update(prefix + name for name in BRANDING_FILES)
    expected = {prefix + item['path'] for item in manifest['files']}
    expected.add(prefix + 'version.json')
    if not required.issubset(expected):
        raise ValueError('Release is missing its launchers, branding, or default prefill assets')
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(expected) or set(names) != expected or bundle.testzip() is not None:
            raise ValueError('Final ZIP contents or CRC verification failed')
        for item in manifest['files']:
            data = bundle.read(prefix + item['path'])
            if len(data) != item['bytes'] or digest(data) != item['sha256']:
                raise ValueError(f'Final ZIP file mismatch: {item["path"]}')
        if json.loads(bundle.read(prefix + 'version.json')) != manifest:
            raise ValueError('Final ZIP manifest mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT,
                        help='Release directory (default: ../dist relative to the repository)')
    parser.add_argument('--readobj', type=Path,
                        help='llvm-readobj executable (default: PATH or workspace toolchain)')
    parser.add_argument('--template-dir', type=Path, default=TOOLS / 'astudio_template',
                        help='Template assets to bundle (default: assets next to this script)')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.version):
        parser.error('Invalid version name')
    readobj = find_readobj(args.readobj)
    run = args.run_dir.resolve()
    source = read_json(run / 'source-manifest.json')
    native = read_json(run / 'native-provenance.json')
    startup = read_json(run / 'startup-validation.json')
    if source['dirty'] or native['commit'] != source['commit']:
        raise ValueError('Native build must match the clean frozen source')
    if digest((run / 'source.zip').read_bytes()) != source['source_zip_sha256']:
        raise ValueError('Source ZIP changed')
    if native['source_files_verified'] != len(source['files']):
        raise ValueError('Incomplete native source verification')
    expected = {
        'GGML_VULKAN': 'ON', 'BUILD_SHARED_LIBS': 'OFF', 'GGML_BACKEND_DL': 'OFF',
        'CMAKE_MSVC_RUNTIME_LIBRARY': 'MultiThreaded', 'GGML_OPENMP': 'ON',
        'GGML_NATIVE': 'OFF', 'GGML_AVX2': 'ON', 'GGML_AVX512': 'OFF',
        'LLAMA_BUILD_COMMIT': source['commit'][:9],
    }
    for key, value in expected.items():
        if native['cmake_cache'].get(key) != value:
            raise ValueError(f'Build option mismatch: {key}')
    if not startup['passed']:
        raise ValueError('Missing startup validation')
    template_dir = args.template_dir.resolve()
    template_data = load_default_template(template_dir)
    tested_files = {item['path']: item for item in startup['files']}
    for item in native['runtime_files']:
        if tested_files.get(item['path'], {}).get('sha256') != item['sha256']:
            raise ValueError(f'Runtime differs from the tested file: {item["path"]}')
    output_dir = args.output_dir.expanduser().resolve()
    destination = output_dir / args.version
    if destination.exists() or destination.is_symlink():
        raise ValueError(f'Release already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_files = {f['path']: f for f in source['files']}
    with zipfile.ZipFile(run / 'source.zip') as source_zip:
        for name, item in source_files.items():
            data = source_zip.read(name)
            if len(data) != item['bytes'] or digest(data) != item['sha256']:
                raise ValueError(f'Frozen source mismatch: {name}')
        with tempfile.TemporaryDirectory(prefix='.vulkan-release-', dir=output_dir) as temp:
            release = Path(temp) / args.version
            payload = release / args.version
            payload.mkdir(parents=True)
            runtime_names = {item['path'] for item in native['runtime_files']}
            with zipfile.ZipFile(run / 'native-runtime.zip') as runtime_zip:
                entries = [i for i in runtime_zip.infolist() if not i.is_dir()]
                entry_names = {i.filename.replace('\\', '/'): i.filename for i in entries}
                if len(entries) != len(runtime_names) or set(entry_names) != runtime_names:
                    raise ValueError('Unexpected native runtime files')
                for item in native['runtime_files']:
                    name = item['path']
                    if Path(name).is_absolute() or '..' in Path(name).parts:
                        raise ValueError('Unsafe native runtime path')
                    data = runtime_zip.read(entry_names[name])
                    if len(data) != item['bytes'] or digest(data) != item['sha256']:
                        raise ValueError(f'Runtime hash mismatch: {name}')
                    out = payload / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(data)
            for name in LAUNCHERS:
                data = (TOOLS / name).read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
                tested = next(item for item in startup['files'] if item['path'] == name)
                if digest(data) != tested['sha256']:
                    raise ValueError(f'Launcher differs from the tested file: {name}')
                (payload / name).write_bytes(data)
            shutil.copyfile(TOOLS / 'README.zh-CN.md', payload / 'README.zh-CN.md')
            shutil.copyfile(TOOLS / 'PACKAGING.zh-CN.md', payload / 'PACKAGING.zh-CN.md')
            for name in BRANDING_FILES:
                (payload / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(TOOLS / name, payload / name)
            (payload / 'astudio_template').mkdir()
            (payload / 'astudio_template' / 'template.txt').write_bytes(template_data)
            for name in TEMPLATE_FILES[1:]:
                shutil.copyfile(template_dir / name, payload / 'astudio_template' / name)
            licenses = {
                'LICENSE': 'llama.cpp-LICENSE.txt',
                'licenses/LICENSE-jsonhpp': 'jsonhpp-LICENSE.txt',
                'vendor/cpp-httplib/LICENSE': 'cpp-httplib-LICENSE.txt',
                'vendor/hash/xxhash/LICENSE': 'xxhash-LICENSE.txt',
                'vendor/hash/sha256/LICENSE': 'sha256-LICENSE.txt',
                'vendor/hash/rotate-bits/LICENSE.md': 'rotate-bits-LICENSE.md',
            }
            for source_name, name in licenses.items():
                (payload / 'licenses' / name).write_bytes(source_zip.read(source_name))
            shutil.copyfile(TOOLS / 'Vulkan-Headers-MIT.txt', payload / 'licenses/Vulkan-Headers-MIT.txt')
            # These single-header libraries keep their complete notices in the header.
            for name in ('vendor/stb/stb_image.h', 'vendor/miniaudio/miniaudio.h',
                         'vendor/sheredom/subprocess.h'):
                out = payload / 'licenses' / Path(name).name
                out.write_bytes(source_zip.read(name))
            pe_files = sorted((payload / 'bin').iterdir())
            required = {'llama-cli.exe', 'llama-server.exe', 'llama-bench.exe', 'vcomp140.dll'}
            if {p.name for p in pe_files} != required:
                raise ValueError('Unexpected executable set')
            pe_records = []
            for path in pe_files:
                result = subprocess.check_output([str(readobj), '--file-headers', '--coff-imports', str(path)], text=True)
                (run / (path.name + '.pe.txt')).write_text(result)
                machine = re.search(r'Machine: (\S+)', result).group(1)
                if machine != 'IMAGE_FILE_MACHINE_AMD64':
                    raise ValueError(f'Wrong PE architecture: {path.name}')
                imports = sorted(set(re.findall(r'^\s*Name: (\S+\.dll)\s*$', result, re.M | re.I)))
                missing = [dll for dll in imports if dll.lower() not in required
                           and dll.lower() not in SYSTEM_DLLS
                           and dll.lower() != 'vulkan-1.dll'
                           and not dll.lower().startswith(('api-ms-win-', 'ext-ms-win-'))]
                if missing:
                    raise ValueError(f'Unbundled DLL dependencies in {path.name}: {missing}')
                if path.suffix == '.exe' and 'vulkan-1.dll' not in {d.lower() for d in imports}:
                    raise ValueError(f'Missing Vulkan linkage: {path.name}')
                pe_records.append(dict(**record(path, payload), machine=machine, imports=imports))
            manifest = dict(schema_version=2, project_name=PROJECT_NAME, version=args.version,
                logo=BRANDING_FILES[0],
                packaged_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                source_commit=source['commit'], source_dirty=False,
                source_zip_sha256=source['source_zip_sha256'], architecture='windows-x64',
                backend='Vulkan', cpu_minimum='AVX2/FMA/F16C', native_build=native,
                startup_validation=startup, pe_verification=pe_records,
                external_runtime_requirements=['Windows 11 x64', 'Vulkan-capable GPU driver with vulkan-1.dll'],
                models_included=False, model_inference_tested=False, performance_tested=False,
                default_prefill=dict(enabled=True, file='astudio_template/template.txt',
                                     bytes=len(template_data), sha256=digest(template_data),
                                     newlines='lf', input_format='rendered_prompt',
                                     tokenization=dict(add_special=False, parse_special=True)),
                files=[record(p, payload) for p in sorted(payload.rglob('*')) if p.is_file()])
            (payload / 'version.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
            archive = release / (args.version + '.zip')
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for path in sorted(payload.rglob('*')):
                    if path.is_file():
                        z.write(path, path.relative_to(release).as_posix())
            verify_archive(archive, args.version, manifest)
            archive_hash = digest(archive.read_bytes())
            (release / 'SHA256SUMS.txt').write_text(archive_hash + '  ' + archive.name + '\n')
            shutil.copyfile(payload / 'version.json', release / 'version.json')
            shutil.copyfile(TOOLS / 'README.zh-CN.md', release / 'README.zh-CN.md')
            shutil.copyfile(TOOLS / 'PACKAGING.zh-CN.md', release / 'PACKAGING.zh-CN.md')
            for name in BRANDING_FILES:
                (release / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(TOOLS / name, release / name)
            shutil.copyfile(run / 'RELEASE-REPORT.md', release / 'RELEASE-REPORT.md')
            (run / 'package-verification.json').write_text(json.dumps(dict(
                archive=archive.name, sha256=archive_hash, bytes=archive.stat().st_size,
                files=len(manifest['files']) + 1, pe_checks_passed=True,
                archive_contents_verified=True, default_prefill_included=True), indent=2) + '\n')
            shutil.rmtree(payload)
            release.rename(destination)
    print(destination / (args.version + '.zip'))


if __name__ == '__main__':
    main()
