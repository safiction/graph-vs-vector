"""Build schema-derived grammars with flexible but bounded JSON whitespace."""
import json
import sys
from pathlib import Path
import xgrammar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from graph_rag.common import digest, file_digest
from graph_rag.extract import EXTRACTION_SCHEMA
from graph_rag.index import SUMMARY_SCHEMA

config_path = ROOT / 'configs/graph_vllm_5080.json'
config = json.loads(config_path.read_text())
grammars = {}
for name, schema in [('extraction', EXTRACTION_SCHEMA), ('summary', SUMMARY_SCHEMA)]:
    path = config_path.parent / f'graph_{name}.ebnf'
    grammar = xgrammar.Grammar.from_json_schema(schema, any_whitespace=True, strict_mode=True, max_whitespace_cnt=8)
    path.write_text(str(grammar) + '\n')
    grammars[digest(schema)] = {'file': path.name, 'sha256': file_digest(path), 'max_whitespace_count': 8}
config['json_grammars'] = grammars
config['structured_outputs_config'] = {'backend': 'xgrammar'}
config['launcher_sha256'] = file_digest(ROOT / 'scripts/serve_graph_vllm.sh')
config_path.write_text(json.dumps(config, indent=2) + '\n')
