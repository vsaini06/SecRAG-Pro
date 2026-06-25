Write-Host "Stopping SecRAG containers..." -ForegroundColor Yellow

docker compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host "SecRAG stopped successfully." -ForegroundColor Green
} else {
    Write-Host "Something went wrong while stopping containers." -ForegroundColor Red
}