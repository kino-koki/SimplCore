# Training corpus provenance

`tinyshakespeare.txt` — the character-level corpus used by every real-training
step in this directory.

- source: `karpathy/char-rnn`, `data/tinyshakespeare/input.txt` (public domain
  Shakespeare text, standard char-LM benchmark)
- bytes: 1,115,394
- sha256: `86c4e6aa9db7c042…` (full digest recorded in each result JSON)
- fetched through the GitHub Contents API with
  `Accept: application/vnd.github.raw`

Fetch it again on any machine with:

```powershell
Invoke-WebRequest `
  'https://api.github.com/repos/karpathy/char-rnn/contents/data/tinyshakespeare/input.txt?ref=master' `
  -Headers @{ Accept='application/vnd.github.raw'; 'User-Agent'='simplcore' } `
  -OutFile 'realtrain/data/tinyshakespeare.txt'
```

The corpus is **real human text**, not generated key/value bindings. The
train/validation split is chronological (first 90% / last 10%) and is never
shuffled. No synthetic oracle is injected into the training signal.
