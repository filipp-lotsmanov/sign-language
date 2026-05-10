// WebSocket connection
let ws = null;
let sessionId = null;
let isRecording = false;
let stream = null;
let captureInterval = null;

// DOM elements
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const targetLetterElement = document.getElementById('targetLetter');
const predictionLetterElement = document.getElementById('predictionLetter');
const confidenceElement = document.getElementById('confidence');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');
const startBtn = document.getElementById('startBtn');
const recordBtn = document.getElementById('recordBtn');
const skipBtn = document.getElementById('skipBtn');
const modeCheckbox = document.getElementById('modeCheckbox');
const languageCheckbox = document.getElementById('languageCheckbox');
const tutorialGif = document.getElementById('tutorialGif');
const correctCountElement = document.getElementById('correctCount');
const totalCountElement = document.getElementById('totalCount');
const accuracyElement = document.getElementById('accuracy');
const timeDisplay = document.getElementById('timeDisplay');
const hintMessage = document.getElementById('hintMessage');
const hintText = document.getElementById('hintText');
const successMessage = document.getElementById('successMessage');
const timeoutMessage = document.getElementById('timeoutMessage');
const recordingProgress = document.getElementById('recordingProgress');

// Sentence Mode Elements
const practiceModeSelect = document.getElementById('practiceModeSelect');
const singleLetterUI = document.getElementById('singleLetterUI');
const sentenceUI = document.getElementById('sentenceUI');
const targetSentenceInput = document.getElementById('targetSentenceInput');
const setSentenceBtn = document.getElementById('setSentenceBtn');
const targetSentenceDisplay = document.getElementById('targetSentenceDisplay');
const recognizedSentenceDisplay = document.getElementById('recognizedSentenceDisplay');
const clearSentenceBtn = document.getElementById('clearSentenceBtn');
const sentenceModeTitle = document.getElementById('sentenceModeTitle');
const modeSwitchContainer = document.getElementById('modeSwitchContainer');

// State
let currentMode = 'sequential';  // Track current mode
let currentLanguage = 'en';  // Track current language
let currentTutorialUrl = null;  // Track current tutorial URL to prevent GIF restart

// Translations
const translations = {
    en: {
        'title': 'Sign Language Learning',
        'subtitle': 'Learn NGT alphabet with real-time feedback',
        'label-letter-order': 'Letter Order',
        'label-language': 'Language',
        'mode-abc': 'ABC',
        'mode-random': 'Random',
        'lang-en': 'English',
        'lang-nl': 'Nederlands',
        'target-letter': 'Target Letter',
        'watch-example': 'Watch the example above',
        'instructions': 'Instructions:',
        'instr-1': 'Watch the example GIF',
        'instr-2': 'Position your hand in view',
        'instr-3': 'Click "Record" button',
        'instr-4': 'Hold the sign for 3 seconds',
        'btn-start': 'Start Camera',
        'btn-record': 'Record',
        'btn-stop-record': 'Stop Recording',
        'btn-skip': 'Skip Letter',
        'btn-running': 'Running',
        'status-not-connected': 'Not Connected',
        'status-connected': 'Connected',
        'status-hand-detected': 'Hand Detected',
        'recording': 'Recording',
        'prediction': 'Prediction',
        'stat-correct': 'Correct',
        'stat-attempts': 'Attempts',
        'stat-accuracy': 'Accuracy',
        'time-remaining': 'Time Remaining',
        'msg-correct-title': '✓ Correct!',
        'msg-correct-text': 'Moving to next letter...',
        'msg-timeout-title': 'Time\'s up!',
        'msg-timeout-text': 'Moving to next letter...',
        'label-practice-mode': 'Practice Mode',
        'mode-letter-practice': 'Single Letters',
        'mode-sentence-practice': 'Target Sentence',
        'mode-free-practice': 'Free Sign',
        'sentence-mode-title': 'Target Sentence',
        'btn-set': 'Set',
        'btn-clear': 'Clear Recognition'
    },
    nl: {
        'title': 'Gebarentaal Leren',
        'subtitle': 'Leer het NGT alfabet met real-time feedback',
        'label-letter-order': 'Letter Volgorde',
        'label-language': 'Taal',
        'mode-abc': 'ABC',
        'mode-random': 'Willekeurig',
        'lang-en': 'English',
        'lang-nl': 'Nederlands',
        'target-letter': 'Doelletter',
        'watch-example': 'Bekijk het voorbeeld hierboven',
        'instructions': 'Instructies:',
        'instr-1': 'Bekijk de voorbeeld GIF',
        'instr-2': 'Plaats je hand in beeld',
        'instr-3': 'Klik op "Opnemen" knop',
        'instr-4': 'Houd het teken 3 seconden vast',
        'btn-start': 'Start Camera',
        'btn-record': 'Opnemen',
        'btn-stop-record': 'Stop Opname',
        'btn-skip': 'Letter Overslaan',
        'btn-running': 'Actief',
        'status-not-connected': 'Niet Verbonden',
        'status-connected': 'Verbonden',
        'status-hand-detected': 'Hand Gedetecteerd',
        'recording': 'Opnemen',
        'prediction': 'Voorspelling',
        'stat-correct': 'Correct',
        'stat-attempts': 'Pogingen',
        'stat-accuracy': 'Nauwkeurigheid',
        'time-remaining': 'Resterende Tijd',
        'msg-correct-title': '✓ Correct!',
        'msg-correct-text': 'Naar volgende letter...',
        'msg-timeout-title': 'Tijd is op!',
        'msg-timeout-text': 'Naar volgende letter...',
        'label-practice-mode': 'Oefenmodus',
        'mode-letter-practice': 'Losse Letters',
        'mode-sentence-practice': 'Doelzin',
        'mode-free-practice': 'Vrij Gebaren',
        'sentence-mode-title': 'Doelzin',
        'btn-set': 'Stel in',
        'btn-clear': 'Wis Herkenning'
    }
};

// Initialize
function init() {
    console.log('Initializing app...');
    
    // Check if elements exist
    console.log('modeCheckbox:', modeCheckbox);
    console.log('Toggle container:', document.getElementById('modeSwitchContainer'));
    
    startBtn.addEventListener('click', startSession);
    recordBtn.addEventListener('click', toggleRecording);
    skipBtn.addEventListener('click', skipLetter);
    
    if (modeCheckbox) {
        modeCheckbox.addEventListener('change', toggleMode);
        console.log(' Mode toggle event listener attached');
    } else {
        console.error(' modeCheckbox not found!');
    }
    
    if (languageCheckbox) {
        languageCheckbox.addEventListener('change', toggleLanguage);
        console.log(' Language toggle event listener attached');
    }
    
    // Initially disable buttons
    recordBtn.disabled = true;
    if (modeCheckbox) modeCheckbox.disabled = true;
    if (practiceModeSelect) {
        practiceModeSelect.disabled = true;
        practiceModeSelect.addEventListener('change', togglePracticeMode);
    }
    if (setSentenceBtn) setSentenceBtn.addEventListener('click', setTargetSentence);
    if (clearSentenceBtn) clearSentenceBtn.addEventListener('click', clearRecognizedSentence);
    
    // Initialize displays
    updateModeToggle();
    updateUIForMode(currentMode);
    updateLanguage();
    
    console.log('App initialized');
}

// Toggle language
function toggleLanguage() {
    currentLanguage = languageCheckbox.checked ? 'nl' : 'en';
    console.log(' Language changed to:', currentLanguage);
    updateLanguage();
}

// Update all text elements to current language
function updateLanguage() {
    const lang = translations[currentLanguage];
    
    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        if (lang[key]) {
            element.textContent = lang[key];
        }
    });
    
    // Update dynamic button text if camera is running
    if (startBtn.disabled) {
        startBtn.textContent = lang['btn-running'];
    }
    
    // Update record button based on recording state
    if (isRecording) {
        recordBtn.textContent = lang['btn-stop-record'];
    } else {
        recordBtn.textContent = lang['btn-record'];
    }
}

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log(' WebSocket connected');
        updateStatus('connected', 'Connected');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerResponse(data);
    };
    
    ws.onerror = (error) => {
        console.error(' WebSocket error:', error);
        updateStatus('disconnected', 'Connection error');
    };
    
    ws.onclose = () => {
        console.log(' WebSocket closed');
        updateStatus('disconnected', 'Disconnected');
        // Auto-reconnect after delay
        setTimeout(() => {
            if (stream) {
                connectWebSocket();
            }
        }, 3000);
    };
}

// Toggle letter order mode
async function toggleMode() {
    console.log(' toggleMode called');
    console.log('Session ID:', sessionId);
    console.log('Checkbox state:', modeCheckbox.checked);
    
    if (!sessionId) {
        console.error('No session ID');
        modeCheckbox.checked = !modeCheckbox.checked; // Revert
        return;
    }
    
    // Determine new mode based on checkbox state
    const newMode = modeCheckbox.checked ? 'random' : 'sequential';
    
    try {
        const response = await fetch(`/api/session/${sessionId}/mode`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode: newMode })
        });
        
        if (response.ok) {
            const data = await response.json();
            currentMode = data.mode;
            console.log(` Mode changed to: ${currentMode}`);
        } else {
            console.error('Failed to change mode');
            // Revert checkbox on failure
            modeCheckbox.checked = !modeCheckbox.checked;
        }
    } catch (error) {
        console.error('Error changing mode:', error);
        // Revert checkbox on error
        modeCheckbox.checked = !modeCheckbox.checked;
    }
}

// Update mode toggle visual state
function updateModeToggle() {
    if (modeCheckbox) {
        modeCheckbox.checked = (currentMode === 'random');
    }
    updateUIForMode(currentMode);
}

// Sentence mode logic
async function togglePracticeMode() {
    if (!sessionId) return;
    
    let selectedMode = practiceModeSelect.value;
    // Map 'letter' to sequential or random based on checkbox
    const actualMode = selectedMode === 'letter' ? (modeCheckbox.checked ? 'random' : 'sequential') : selectedMode;
    
    try {
        const response = await fetch(`/api/session/${sessionId}/mode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: actualMode })
        });
        
        if (response.ok) {
            currentMode = actualMode;
            updateUIForMode(selectedMode);
            
            // Set sentence blank for free mode
            if (selectedMode === 'free') {
                ws.send(JSON.stringify({ type: 'set_sentence', sentence: '', session_id: sessionId }));
            }
        }
    } catch (error) {
        console.error('Error changing practice mode:', error);
    }
}

function setTargetSentence() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const val = targetSentenceInput.value.trim();
    ws.send(JSON.stringify({
        type: 'set_sentence',
        sentence: val,
        session_id: sessionId
    }));
}

function clearRecognizedSentence() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
        type: 'clear_sentence',
        session_id: sessionId
    }));
}

function updateUIForMode(modeStr) {
    if (!practiceModeSelect) return;
    
    if (modeStr === 'sequential' || modeStr === 'random' || modeStr === 'letter') {
        singleLetterUI.style.display = 'block';
        sentenceUI.style.display = 'none';
        modeSwitchContainer.style.display = 'flex';
        practiceModeSelect.value = 'letter';
    } else {
        singleLetterUI.style.display = 'none';
        sentenceUI.style.display = 'block';
        modeSwitchContainer.style.display = 'none';
        
        if (modeStr === 'free') {
            document.getElementById('sentenceInputGroup').style.display = 'none';
            targetSentenceDisplay.style.display = 'none';
            sentenceModeTitle.textContent = translations[currentLanguage]['mode-free-practice'];
            practiceModeSelect.value = 'free';
        } else {
            document.getElementById('sentenceInputGroup').style.display = 'flex';
            targetSentenceDisplay.style.display = 'block';
            sentenceModeTitle.textContent = translations[currentLanguage]['mode-sentence-practice'];
            practiceModeSelect.value = 'sentence';
        }
    }
}

// Skip current letter and move to next one
function skipLetter() {
    if (!sessionId || !ws || ws.readyState !== WebSocket.OPEN) {
        console.error('Cannot skip: no active session or WebSocket not connected');
        return;
    }
    
    console.log(' Skipping letter:', targetLetterElement.textContent);
    
    // If recording, stop it first
    if (isRecording) {
        stopRecording();
    }
    
    // Send skip command
    ws.send(JSON.stringify({
        type: 'skip',
        session_id: sessionId
    }));
}

// Start session
async function startSession() {
    try {
        // Start webcam
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        webcamElement.srcObject = stream;
        
        // Connect WebSocket
        connectWebSocket();
        
        // Start sending frames
        startCaptureLoop();
        
        // Update UI
        startBtn.disabled = true;
        recordBtn.disabled = false;
        skipBtn.disabled = false;
        if (modeCheckbox) modeCheckbox.disabled = false;
        if (practiceModeSelect) practiceModeSelect.disabled = false;
        startBtn.textContent = translations[currentLanguage]['btn-running'];
        
        console.log(' Session started');
    } catch (error) {
        console.error(' Failed to start:', error);
        alert('Failed to access camera. Please check permissions.');
    }
}

// Toggle recording
function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

// Start recording
function startRecording() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        console.error('WebSocket not ready');
        return;
    }
    
    isRecording = true;
    recordBtn.textContent = translations[currentLanguage]['btn-stop-record'];
    recordBtn.classList.add('recording');
    
    // Hide messages
    successMessage.style.display = 'none';
    timeoutMessage.style.display = 'none';
    hintMessage.style.display = 'none';
    
    // Show recording progress
    if (recordingProgress) {
        recordingProgress.style.display = 'block';
    }
    
    // Send start recording command
    ws.send(JSON.stringify({
        type: 'start_recording',
        session_id: sessionId
    }));
    
    console.log(' Recording started');
}

// Stop recording
function stopRecording() {
    isRecording = false;
    recordBtn.textContent = translations[currentLanguage]['btn-record'];
    recordBtn.classList.remove('recording');
    
    // Hide recording progress
    if (recordingProgress) {
        recordingProgress.style.display = 'none';
    }
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'stop_recording',
            session_id: sessionId
        }));
    }
    
    console.log(' Recording stopped');
}

// Capture and send frames
function startCaptureLoop() {
    captureInterval = setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            captureFrame();
        }
    }, 100); // 10 FPS
}

// Capture frame from video
function captureFrame() {
    const canvas = canvasElement;
    const context = canvas.getContext('2d');
    
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    
    context.drawImage(webcamElement, 0, 0);
    
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    
    ws.send(JSON.stringify({
        type: 'frame',
        frame: frameData,
        session_id: sessionId
    }));
}

// Handle server response
function handleServerResponse(data) {
    console.log(' Server response:', data);
    
    // Update session ID
    if (data.session_id && !sessionId) {
        sessionId = data.session_id;
        console.log('Session ID:', sessionId);
    }
    
    // Update progress
    if (data.progress) {
        const prog = data.progress;
        console.log(' Progress update:', prog);
        targetLetterElement.textContent = prog.current_letter || data.current_letter;
        correctCountElement.textContent = prog.total_correct || 0;
        totalCountElement.textContent = prog.total_attempts || 0;
        
        const accuracy = prog.accuracy || 0;
        accuracyElement.textContent = accuracy.toFixed(1) + '%';
        
        // Update timer
        if (prog.time_remaining !== undefined) {
            timeDisplay.textContent = Math.ceil(prog.time_remaining) + 's';
        }
        
        // Update tutorial GIF (only if URL changed to prevent restart)
        if (prog.tutorial_url && prog.tutorial_url !== currentTutorialUrl) {
            currentTutorialUrl = prog.tutorial_url;
            tutorialGif.src = prog.tutorial_url;
            tutorialGif.style.display = 'block';
        } else if (!prog.tutorial_url && currentTutorialUrl !== null) {
            // Hide GIF if no tutorial available for this letter
            currentTutorialUrl = null;
            tutorialGif.style.display = 'none';
        }
        
        // Update mode if provided
        if (prog.mode && prog.mode !== currentMode) {
            currentMode = prog.mode;
            updateModeToggle();
        }
        
        // Update Sentence Displays
        if (prog.target_sentence !== undefined) {
            targetSentenceDisplay.textContent = prog.target_sentence;
        }
        if (prog.recognized_sentence !== undefined) {
            const tgt = prog.target_sentence || '';
            const rec = prog.recognized_sentence || '';
            
            // Format recognizing target
            if (tgt && prog.mode === 'sentence') {
                let html = '';
                for (let i = 0; i < tgt.length; i++) {
                    if (tgt[i] === ' ') {
                        html += '<span>&nbsp;</span>';
                    } else if (i < rec.length) {
                        html += `<span class="correct-char">${tgt[i]}</span>`;
                    } else if (i === rec.length) {
                        html += `<span class="current-target-char">${tgt[i]}</span>`;
                    } else {
                        html += `<span class="pending-char">${tgt[i]}</span>`;
                    }
                }
                recognizedSentenceDisplay.innerHTML = html;
            } else {
                recognizedSentenceDisplay.textContent = rec;
            }
        }
    }
    
    // Update hand detection status
    if (data.hand_detected !== undefined) {
        if (data.hand_detected) {
            updateStatus('hand-detected', 'Hand Detected');
        } else {
            updateStatus('connected', data.message || 'No Hand');
        }
    }
    
    // Update recording status
    if (data.recording !== undefined) {
        if (data.recording) {
            const message = data.message || 'Recording...';
            statusText.textContent = message;
            
            // Show recording overlay
            if (recordingProgress) {
                recordingProgress.style.display = 'block';
            }
            
            // Update buffer progress (for dynamic letters)
            const progressPercent = document.getElementById('progressPercent');
            if (data.buffer_progress !== undefined && progressPercent) {
                const pct = (data.buffer_progress * 100).toFixed(0);
                progressPercent.textContent = ` ${pct}%`;
            } else if (progressPercent) {
                // For static letters, just show recording without percentage
                progressPercent.textContent = '';
            }
        } else {
            // Recording finished
            isRecording = false;
            recordBtn.textContent = translations[currentLanguage]['btn-record'];
            recordBtn.classList.remove('recording');
            
            if (recordingProgress) {
                recordingProgress.style.display = 'none';
            }
        }
    }
    
    // Update prediction
    if (data.prediction) {
        const pred = data.prediction;
        const predictedClass = pred.predicted_class || pred.letter || '-';
        // Replace "Nonsense" with a dash for better UI
        const displayLetter = predictedClass === 'Nonsense' ? '–' : predictedClass;
        predictionLetterElement.textContent = displayLetter;
        
        const conf = (pred.confidence * 100).toFixed(1);
        confidenceElement.textContent = `${conf}%`;
    } else if (data.current_prediction) {
        const displayLetter = data.current_prediction === 'Nonsense' ? '–' : data.current_prediction;
        predictionLetterElement.textContent = displayLetter;
        
        if (data.confidence) {
            const conf = (data.confidence * 100).toFixed(1);
            confidenceElement.textContent = `${conf}%`;
        }
    }
    
    // Handle success
    if (data.success) {
        showSuccess();
    }
    
    // Handle timeout
    if (data.timeout) {
        showTimeout();
    }
    
    // Handle skip (optional: show a brief message)
    if (data.skipped) {
        console.log(' Letter skipped:', data.message);
        // Clear prediction display when skipping
        predictionLetterElement.textContent = '-';
        confidenceElement.textContent = '-';
        // Optionally show a brief notification
        // showSkipped();
    }
    
    // Handle hints
    if (data.show_hint && data.hint_message) {
        showHint(data.hint_message);
    }
    
    // Show general message
    if (data.message && !isRecording) {
        console.log('', data.message);
    }
}

// Update status badge
function updateStatus(status, text) {
    statusBadge.className = 'status-badge ' + status;
    statusText.textContent = text;
}

// Show success message
function showSuccess() {
    successMessage.style.display = 'flex';
    setTimeout(() => {
        successMessage.style.display = 'none';
    }, 2000);
}

// Show timeout message
function showTimeout() {
    timeoutMessage.style.display = 'flex';
    setTimeout(() => {
        timeoutMessage.style.display = 'none';
    }, 2000);
}

// Show hint message
function showHint(message) {
    hintText.textContent = message;
    hintMessage.style.display = 'flex';
    setTimeout(() => {
        hintMessage.style.display = 'none';
    }, 5000);
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
