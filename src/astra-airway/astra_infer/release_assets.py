"""Download public ASTRA-Net Release assets when they are needed."""

from pathlib import Path
import os
import urllib.request
from zipfile import ZipFile

from tqdm import tqdm


RELEASE_REPO = 'guhong3648/ASTRA-Net'
RELEASE_TAG = 'v0.3.0'


def download_release_asset(filename, destination):
    destination = Path(destination).expanduser()
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://github.com/{RELEASE_REPO}/releases/download/{RELEASE_TAG}/{filename}'
    temp_path = destination.with_name(destination.name+'.part')

    try:
        with urllib.request.urlopen(url) as response, temp_path.open('wb') as output:
            size = int(response.headers.get('Content-Length') or 0)
            with tqdm(total=size, unit='B', unit_scale=True, desc=f'Download {filename}') as bar:
                while True:
                    chunk = response.read(1024*1024)
                    if not chunk: break
                    output.write(chunk)
                    bar.update(len(chunk))
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return destination


def ensure_checkpoint(model_path, model_name):
    """Reuse the local checkpoint, or download only the requested model."""
    model_path = Path(model_path).expanduser()
    if model_path.is_file():
        return model_path

    return download_release_asset(f'{model_name}_Final.pt', model_path)


def download_enhanced_labels(data_root='../data'):
    """Extract our enhanced annotations into data/raw/ATM26_Enhanced."""
    data_root = Path(data_root).expanduser()
    target = data_root/'raw/ATM26_Enhanced'
    if any(target.glob('*.nrrd')):
        return target

    archive = download_release_asset('ATM26_Enhanced_Labels.zip', data_root/'release_assets/ATM26_Enhanced_Labels.zip')
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as z:
        files = [name for name in z.namelist() if name.startswith('ATM26_Enhanced/') and name.endswith('.nrrd') and '/' not in name[len('ATM26_Enhanced/'):]]
        if not files:
            raise ValueError('The Release archive has no ATM26_Enhanced NRRD labels')
        for name in files:
            (target/Path(name).name).write_bytes(z.read(name))

    print(f'ATM26 Enhanced labels: {len(files)} files -> {target}', flush=True)
    return target


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='../data')
    args = parser.parse_args()
    download_enhanced_labels(args.data_root)
