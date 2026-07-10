const API = 'http://127.0.0.1:8000';
let turns = 0;
let currentUrl = '';
let conversationHistory = [];

async function loadVideo() {
    const url = document.getElementById('url-input').value.trim();
  
    if (!url) {
        document.getElementById('url-hint').textContent = 'Please enter a YouTube URL first.';
        document.getElementById('url-hint').style.color = '#ef4444';
        return;
    }

    const btn = document.getElementById('load-btn');
    btn.textContent = 'Loading...';
    btn.disabled = true;
    currentUrl = url;

    try {
        const response = await fetch(`${API}/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_url: url })
            });

        if (response.ok) {
        const data = await response.json();
        showChatScreen(data.summary, data.duration_seconds);
        } else {
        showChatScreen('Backend not connected — start FastAPI first.', null);
        }
    } catch {
        showChatScreen('Backend not running. Start with: uvicorn main:app --reload', null);
    }
    btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Load video';
    btn.disabled = false;
}

function showChatScreen(summary, duration) {
    document.getElementById('screen-landing').classList.add('hidden');
    document.getElementById('screen-chat').classList.remove('hidden');

    document.getElementById('back-btn').classList.remove('hidden');

    document.getElementById('summary-content').textContent = summary || 'No summary available.';

    if (duration) {
        document.getElementById('strip-meta').textContent = '· ' + (duration / 60).toFixed(1) + ' min';
    }

    window.scrollTo(0, 0);
}

function goHome() {
    document.getElementById('screen-chat').classList.add('hidden');
    document.getElementById('screen-landing').classList.remove('hidden');

    document.getElementById('back-btn').classList.add('hidden');

    document.getElementById('url-input').value = '';
    document.getElementById('url-hint').textContent = 'Press Enter or click Load video to continue';
    document.getElementById('url-hint').style.color = '#4a5568';

    document.getElementById('summary-content').textContent = 'Load a video to see its summary.';
    document.getElementById('strip-meta').textContent = '';

    turns = 0;
    currentUrl = '';
    conversationHistory = [];

    clearChat();

    window.scrollTo(0, 0);
}

function clearChat() {
    turns = 0;
    document.getElementById('chat-messages').innerHTML = `
    <div id="empty-chat">
        <p id="empty-title">Ready to answer</p>
        <p>Ask anything about this video below</p>
    </div>`;
    document.getElementById('turn-counter').textContent = '0 turns';
    document.getElementById('stat-decision').textContent = '—';
    document.getElementById('stat-chunks').textContent = '—';
    document.getElementById('stat-score').textContent = '—';
    document.getElementById('stat-attempts').textContent = '—';
}

document.getElementById('load-btn').addEventListener('click', loadVideo);
document.getElementById('back-btn').addEventListener('click', goHome);
document.getElementById('url-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') loadVideo();
});