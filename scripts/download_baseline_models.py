"""Download only pinned model/tokenizer assets into the project-local cache."""
import argparse
import json
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/baseline_15000.json'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for key in ['embedding', 'generation_settings']:
        settings = config[key]
        patterns = ['*.json', 'vocab.txt', 'merges.txt', 'LICENSE*', 'README.md']
        if key == 'embedding' or settings['backend'] == 'transformers':
            patterns.append('*.safetensors')
        path = snapshot_download(settings['model'], revision=settings['revision'], cache_dir=settings['cache_dir'],
            allow_patterns=patterns,
            ignore_patterns=['onnx/*', 'openvino/*', '*.bin'])
        print(f'{key}: {path}', flush=True)


if __name__ == '__main__':
    main()
