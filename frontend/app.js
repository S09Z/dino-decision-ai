/**
 * Chrome Dino AI Dashboard
 */

const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
    console.log('Connected to server');
    document.getElementById('status').textContent = 'Connected';
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
    
    // Update UI with data
    if (data.active_agent) {
        document.getElementById('activeAgent').textContent = data.active_agent;
    }
    if (data.score !== undefined) {
        document.getElementById('score').textContent = data.score;
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    document.getElementById('status').textContent = 'Connection Error';
};

ws.onclose = () => {
    console.log('Disconnected from server');
    document.getElementById('status').textContent = 'Disconnected';
};
