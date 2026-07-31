# StopDFF v5 identity and artifact contract

`scripts/stopdff_v5/identity.py` defines canonical JSON bytes and content IDs.
Identities allow only JSON objects, arrays, strings, integers, booleans, and
null; strings and keys are NFC-normalized; keys are sorted; scientific decimals
are strings; floats are rejected. An artifact manifest has a SHA-256 `id`
computed only from its canonical `identity`.

`scripts/stopdff_v5/manifests.py` defines every identity graph edge.
`scripts/stopdff_v5/writers.py` defines package creation, and
`scripts/stopdff_v5/checker.py` independently recomputes the bindings. Emitted
artifacts must conform to the schemas under `schemas/stopdff_*.schema.json`.

Artifacts and attempts are create-once. Resume may reuse an object only after
its complete bytes and upstream identity graph match the current run.
