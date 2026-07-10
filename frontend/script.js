const API = 'http://127.0.0.1:8000';
let turns = 0;
let currentUrl = '';
let conversationHistory = [];

function isValidYouTubeUrl(url) {
    return /youtube\.com\/watch\?v=|youtu\.be\//.test(url);
}

async function loadVideo() {
    const url = document.getElementById('url-input').value.trim();

    if (!url) {
        document.getElementById('url-hint').textContent = 'Please enter a YouTube URL first.';
        document.getElementById('url-hint').style.color = '#ef4444';
        return;
    }

    if (!isValidYouTubeUrl(url)) {
        document.getElementById('url-hint').textContent = 'That doesn\'t look like a valid YouTube URL.';
        document.getElementById('url-hint').style.color = '#ef4444';
        return;
    }

    currentUrl = url;

    const btn = document.getElementById('load-btn');
    btn.innerHTML = 'Loading<span class="loading-dot">.</span><span class="loading-dot">.</span><span class="loading-dot">.</span>';
    btn.disabled = true;

    const loadPromise = fetch(`${API}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: url })
    });

    await new Promise(resolve => setTimeout(resolve, 3000));
    showChatScreen();

    try {
        const response = await loadPromise;

        if (response.ok) {
            const data = await response.json();
            document.getElementById('summary-content').textContent = data.summary || 'No summary available.';
            if (data.duration_seconds) {
                document.getElementById('strip-meta').textContent = '· ' + (data.duration_seconds / 60).toFixed(1) + ' min';
            }
        } else {
            document.getElementById('summary-content').textContent = 'Could not generate summary.';
        }
    } catch {
        document.getElementById('summary-content').textContent = 'Could not reach the backend.';
    }

    document.getElementById('send-btn').disabled = false;
    document.getElementById('chat-input').disabled = false;
}

function showChatScreen() {
    document.getElementById('screen-landing').classList.add('hidden');
    document.getElementById('screen-chat').classList.remove('hidden');
    document.getElementById('back-btn').classList.remove('hidden');

    document.getElementById('summary-content').textContent = 'Generating summary...';

    document.getElementById('send-btn').disabled = true;
    document.getElementById('chat-input').disabled = true;

    const messages = document.getElementById('chat-messages');
    messages.innerHTML = `
        <div class="msg-row" id="greeting-typing">
            <div class="avatar ai">AI</div>
            <div class="msg-content">
                <div class="typing"><span></span><span></span><span></span></div>
            </div>
        </div>`;

    setTimeout(() => {
        const typing = document.getElementById('greeting-typing');
        if (typing) {
            typing.outerHTML = `
                <div class="msg-row">
                    <div class="avatar ai">AI</div>
                    <div class="msg-content">
                        <div class="bubble ai">Hi! Ask me any questions you have about the YouTube video.</div>
                    </div>
                </div>`;
        }
    }, 2000);

    window.scrollTo(0, 0);
}

function goHome() {
    document.getElementById('screen-chat').classList.add('hidden');
    document.getElementById('screen-landing').classList.remove('hidden');
    document.getElementById('back-btn').classList.add('hidden');

    const btn = document.getElementById('load-btn');
    btn.innerHTML = '<i class="bi bi-play-circle-fill"></i> Load video';
    btn.disabled = false;

    document.getElementById('url-input').value = '';
    document.getElementById('url-hint').textContent = 'Press Enter or click Load video to continue';
    document.getElementById('url-hint').style.color = '#4a5568';

    document.getElementById('summary-content').textContent = 'Load a video to see its summary.';
    document.getElementById('strip-meta').textContent = '';

    document.getElementById('send-btn').disabled = false;
    document.getElementById('chat-input').disabled = false;

    turns = 0;
    currentUrl = '';
    conversationHistory = [];

    clearChat();
    window.scrollTo(0, 0);
}

function clearChat() {
    turns = 0;
    document.getElementById('chat-messages').innerHTML = `
        <div class="msg-row">
            <div class="avatar ai">AI</div>
            <div class="msg-content">
                <div class="bubble ai">Hi! Ask me any questions you have about the YouTube video.</div>
            </div>
        </div>`;
    document.getElementById('turn-counter').textContent = '0 turns';
}

document.getElementById('load-btn').addEventListener('click', loadVideo);
document.getElementById('back-btn').addEventListener('click', goHome);
document.getElementById('url-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') loadVideo();
});

async function sendMessage() {
    const question = document.getElementById('chat-input').value.trim();
    if (!question) return;

    const fromVal = document.getElementById('from-min').value;
    const toVal = document.getElementById('to-min').value;
    const from_min = fromVal === '' ? null : parseFloat(fromVal);
    const to_min = toVal === '' ? null : parseFloat(toVal);

    const messages = document.getElementById('chat-messages');

    messages.innerHTML += `
        <div class="msg-row user">
            <div class="avatar user">You</div>
            <div class="msg-content">
                <div class="bubble user">${question}</div>
            </div>
        </div>`;

    document.getElementById('chat-input').value = '';

    const typingId = 'typing-' + Date.now();
    messages.innerHTML += `
        <div class="msg-row" id="${typingId}">
            <div class="avatar ai">AI</div>
            <div class="msg-content">
                <div class="typing"><span></span><span></span><span></span></div>
            </div>
        </div>`;
    messages.scrollTop = messages.scrollHeight;

    const response = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            video_url: currentUrl,
            question: question,
            from_min: from_min,
            to_min: to_min
        })
    });

    const data = await response.json();

    document.getElementById(typingId).remove();

    turns++;
    document.getElementById('turn-counter').textContent = turns + ' turn' + (turns === 1 ? '' : 's');

    messages.innerHTML += `
        <div class="msg-row">
            <div class="avatar ai">AI</div>
            <div class="msg-content">
                <div class="bubble ai">${data.answer}</div>
            </div>
        </div>`;
    messages.scrollTop = messages.scrollHeight;
}

document.getElementById('send-btn').addEventListener('click', sendMessage);
document.getElementById('chat-input').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
