# Run evaluation suite
python -m app.eval.run_eval_v2

if ($LASTEXITCODE -ne 0) {
  Write-Host "❌ Tests failed. Fix before committing."
  exit 1
}

Write-Host "✅ Tests passed."
