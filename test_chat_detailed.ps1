$body = @{
    message = "Hello! What is 2+2? Please answer briefly."
    user_id = "test_user"
    conversation_id = "test_001"
} | ConvertTo-Json

Write-Host "Sending chat request..."
Write-Host "Message: Hello! What is 2+2? Please answer briefly."
Write-Host ""

try {
    $response = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/chat" -ContentType "application/json" -Body $body -TimeoutSec 60
    
    Write-Host "✅ Chat Response:"
    Write-Host "Full Response:"
    $response | ConvertTo-Json -Depth 10
    Write-Host ""
    
    if ($response.answer) {
        Write-Host "Answer: $($response.answer)"
    } else {
        Write-Host "⚠️ Warning: Response has no answer field"
    }
    
    Write-Host ""
    Write-Host "✅ API test successful!"
} catch {
    Write-Host "❌ Error: $_"
    Write-Host "Status Code: $($_.Exception.Response.StatusCode.value__)"
    Write-Host "Response: $($_.ErrorDetails.Message)"
}
