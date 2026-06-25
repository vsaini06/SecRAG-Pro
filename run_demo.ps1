docker compose up -d --build

Write-Host "Waiting for backend..."
$ok = $false
for ($i=0; $i -lt 30; $i++) {
  try {
    $r = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch {}
  Start-Sleep -Seconds 2
}

if (-not $ok) {
  Write-Host "Backend did not become healthy. Showing logs..."
  docker compose logs --tail=80 backend
  exit 1
}

Start-Process "http://localhost:5173"
Write-Host "SecRAG is up: http://localhost:5173"