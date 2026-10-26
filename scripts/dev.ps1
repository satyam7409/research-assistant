param([switch]$Fake)
$env:FAKE_PIPELINE = if ($Fake) { 'true' } else { 'false' }
Start-Process powershell -ArgumentList '-NoExit','-Command','uv run uvicorn app.main:app --reload'
Start-Process powershell -ArgumentList '-NoExit','-Command','Set-Location frontend; npm install; npm run dev'
