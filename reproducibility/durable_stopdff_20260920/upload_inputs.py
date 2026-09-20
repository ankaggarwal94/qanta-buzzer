"""Transfer the one frozen input archive to an authenticated temporary endpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

EXPECTED_SHA = '6afc62e1cb91d0d2ac958c251ba68aec1c83f2da2b9d0354dc4635234dbc1cf2'
EXPECTED_SIZE = 1234697660
CHUNK_SIZE = 32 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Redirect refused', headers, fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.endpoint)
    if url.scheme != 'https' or not url.hostname or not url.hostname.endswith('.modal.run') or url.query or url.fragment or url.username:
        raise SystemExit('Expected a verified HTTPS Modal endpoint without credentials or query')
    token = os.environ.get('STOPDFF_TRANSFER_TOKEN')
    if not token:
        raise SystemExit('Transfer authentication unavailable')
    if args.archive.is_symlink() or not args.archive.is_file() or args.archive.stat().st_size != EXPECTED_SIZE:
        raise SystemExit('Unexpected input archive')
    base = args.endpoint.rstrip('/')
    opener = urllib.request.build_opener(NoRedirect())

    def request(path, data=None, headers=None, method='GET'):
        request_headers = {'X-Transfer-Token': token, **(headers or {})}
        req = urllib.request.Request(base + path, data=data, headers=request_headers, method=method)
        for attempt in range(3):
            try:
                with opener.open(req, timeout=140) as response:
                    payload = response.read(65536)
                    return json.loads(payload)
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f'Transfer request failed: HTTP {exc.code}') from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise RuntimeError(f'Transfer request failed: {type(exc).__name__}') from None
            time.sleep(2 ** attempt)
        raise AssertionError('Unreachable')

    before = args.archive.stat()
    whole = hashlib.sha256()
    chunks = []
    with args.archive.open('rb') as source:
        index = 0
        while data := source.read(CHUNK_SIZE):
            whole.update(data)
            digest = hashlib.sha256(data).hexdigest()
            result = request(f'/chunk/{index}', data, {
                'Content-Type': 'application/octet-stream',
                'Content-Length': str(len(data)),
                'X-Chunk-SHA256': digest,
            }, 'PUT')
            chunks.append({'index': index, 'size': len(data), 'sha256': digest, 'result': result})
            print(json.dumps({'chunk_uploaded': index + 1, 'chunks_total': (EXPECTED_SIZE + CHUNK_SIZE - 1) // CHUNK_SIZE}), flush=True)
            index += 1
    after = args.archive.stat()
    if whole.hexdigest() != EXPECTED_SHA or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source changed or archive hash mismatch; finalization not requested')
    finalized = request('/finalize', b'', {'Content-Length': '0'}, 'POST')
    status = request('/status')
    for value in (finalized, status):
        if value.get('status') != 'complete' or value.get('sha256') != EXPECTED_SHA or value.get('size') != EXPECTED_SIZE:
            raise RuntimeError('Remote finalization did not confirm the approved archive identity')
    receipt = {'schema_version': 1, 'archive_sha256': EXPECTED_SHA, 'archive_size': EXPECTED_SIZE,
               'chunks': chunks, 'finalization': finalized, 'status': status,
               'scientific_acceptance': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x') as target:
        json.dump(receipt, target, indent=2)
        target.write('\n')
    print(json.dumps({'finalization': finalized, 'status': status}), flush=True)


if __name__ == '__main__':
    main()
