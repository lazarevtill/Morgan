// Morgan Chat Interface JavaScript

let ws = null;
let userId = 'user_' + Math.random().toString(36).substr(2, 9);
let conversationId = null;
let selectedRating = 0;

function initWebSocket() {
    ws = new WebSocket(`ws://localhost:8000/ws/${userId}`);
    
    ws.onopen = function() {
        document.getElementById('connectionStatus').innerHTML = '<span class="status-connected">● Connected</span>';
    };
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
    
    ws.onclose = function() {
        document.getElementById('connectionStatus').innerHTML = '<span class="status-disconnected">● Disconnected</span>';
        // Attempt to reconnect after 3 seconds
        setTimeout(initWebSocket, 3000);
    };
    
    ws.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
}

function handleMessage(data) {
    const messagesContainer = document.getElementById('chatMessages');
    
    if (data.type === 'welcome') {
        conversationId = data.conversation_id;
        addMessage('morgan', data.message);
        
        // Show user profile info if available
        if (data.user_profile) {
            const profile = data.user_profile;
            if (profile.interaction_count > 0) {
                addSystemMessage(`Welcome back! We've had ${profile.interaction_count} conversations over ${profile.relationship_age_days} days.`);
            }
        }
    } else if (data.type === 'response') {
        addMessage('morgan', data.message, data);
        
        if (data.milestone) {
            addMilestoneCelebration(data.milestone);
        }
        
        if (data.suggestions && data.suggestions.length > 0) {
            addSuggestions(data.suggestions);
        }
    } else if (data.type === 'suggestions') {
        addSuggestions(data.suggestions);
    } else if (data.type === 'feedback_received') {
        addSystemMessage(data.message);
    } else if (data.type === 'error') {
        addSystemMessage(`Error: ${data.message}`, 'error');
    }
}

function addMessage(sender, message, metadata = null) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'user' ? 'You' : 'M';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = formatMessage(message);
    
    if (metadata && metadata.emotional_tone) {
        const indicator = document.createElement('div');
        indicator.className = 'emotional-indicator';
        indicator.textContent = `Speaking with ${metadata.emotional_tone} (empathy: ${Math.round(metadata.empathy_level * 100)}%)`;
        content.appendChild(indicator);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(content);
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addSystemMessage(message, type = 'info') {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `system-message ${type}`;
    messageDiv.innerHTML = `<small><em>${message}</em></small>`;
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addMilestoneCelebration(milestone) {
    const messagesContainer = document.getElementById('chatMessages');
    const celebrationDiv = document.createElement('div');
    celebrationDiv.className = 'milestone-celebration';
    celebrationDiv.innerHTML = `
        <h4>🎉 Milestone Achieved!</h4>
        <p>${milestone.description}</p>
        <small>Emotional significance: ${Math.round(milestone.significance * 100)}%</small>
        ${milestone.celebration_message ? `<p><em>${milestone.celebration_message}</em></p>` : ''}
    `;
    messagesContainer.appendChild(celebrationDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addSuggestions(suggestions) {
    const messagesContainer = document.getElementById('chatMessages');
    const suggestionsDiv = document.createElement('div');
    suggestionsDiv.className = 'suggestions';
    
    let html = '<h4>💡 Conversation suggestions:</h4>';
    suggestions.forEach(suggestion => {
        html += `<button class="suggestion-item" onclick="sendSuggestion('${escapeHtml(suggestion)}')">${suggestion}</button>`;
    });
    
    suggestionsDiv.innerHTML = html;
    messagesContainer.appendChild(suggestionsDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (message && ws && ws.readyState === WebSocket.OPEN) {
        addMessage('user', message);
        ws.send(JSON.stringify({
            type: 'message',
            message: message
        }));
        input.value = '';
    }
}

function sendSuggestion(suggestion) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        addMessage('user', suggestion);
        ws.send(JSON.stringify({
            type: 'message',
            message: suggestion
        }));
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function showFeedback() {
    document.getElementById('feedbackOverlay').style.display = 'block';
    document.getElementById('feedbackPopup').style.display = 'block';
}

function closeFeedback() {
    document.getElementById('feedbackOverlay').style.display = 'none';
    document.getElementById('feedbackPopup').style.display = 'none';
    selectedRating = 0;
    updateStars();
    document.getElementById('feedbackComment').value = '';
}

function submitFeedback() {
    if (selectedRating > 0 && ws && ws.readyState === WebSocket.OPEN) {
        const comment = document.getElementById('feedbackComment').value;
        ws.send(JSON.stringify({
            type: 'feedback',
            rating: selectedRating,
            comment: comment
        }));
        closeFeedback();
    } else {
        alert('Please select a rating before submitting.');
    }
}

function updateStars() {
    const stars = document.querySelectorAll('.star');
    stars.forEach((star, index) => {
        if (index < selectedRating) {
            star.classList.add('active');
        } else {
            star.classList.remove('active');
        }
    });
}

function openPreferences() {
    window.open(`/preferences/${userId}`, '_blank');
}

function openTimeline() {
    window.open(`/timeline/${userId}`, '_blank');
}

function formatMessage(message) {
    // Simple message formatting
    return message
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize rating stars
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.star').forEach(star => {
        star.addEventListener('click', function() {
            selectedRating = parseInt(this.dataset.rating);
            updateStars();
        });
    });
    
    // Initialize WebSocket connection
    initWebSocket();
});