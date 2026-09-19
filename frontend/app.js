'use strict';

// Set ?debug=1 on the URL to get the verbose console output this file used to
// print unconditionally, including every server response.
const DEBUG = new URLSearchParams(window.location.search).has('debug');
const log = (...args) => { if (DEBUG) console.log(...args); };

// Frames are only needed at full rate while the server is scoring an attempt.
// Idle preview runs slower: the browser was previously encoding and uploading a
// full JPEG ten times a second just to keep a "hand detected" badge current.
const CAPTURE_FPS_RECORDING = 10;
const CAPTURE_FPS_IDLE = 2;
const JPEG_QUALITY = 0.8;

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

const NO_LETTER = '–';

// Connection state
let ws = null;
let sessionId = null;
let isRecording = false;
let stream = null;
let captureTimer = null;
let captureFps = CAPTURE_FPS_IDLE;
let reconnectAttempts = 0;
let shuttingDown = false;

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

// Sentence mode elements
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
let currentMode = 'sequential';
let currentLanguage = localStorage.getItem('language') || 'en';
let currentTutorialUrl = null;

// Translations. Keys prefixed `srv-` correspond to the `message_key` the server
// sends alongside its English `message`, so status text is localised instead of
// always being English.
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
        'status-disconnected': 'Disconnected',
        'status-error': 'Connection error',
        'status-reconnecting': 'Reconnecting...',
        'recording': 'Recording',
        'prediction': 'Prediction',
        'stat-correct': 'Correct',
        'stat-attempts': 'Attempts',
        'stat-accuracy': 'Accuracy',
        'time-remaining': 'Time Remaining',
        'msg-correct-title': '✓ Correct!',
        'msg-correct-text': 'Moving to next letter...',
        'msg-timeout-title': "Time's up!",
        'msg-timeout-text': 'Moving to next letter...',
        'label-practice-mode': 'Practice Mode',
        'mode-letter-practice': 'Single Letters',
        'mode-sentence-practice': 'Target Sentence',
        'mode-free-practice': 'Free Sign',
        'sentence-mode-title': 'Target Sentence',
        'btn-set': 'Set',
        'btn-clear': 'Clear Recognition',
        'err-camera': 'Failed to access camera. Please check permissions.',
        // Server message keys
        'srv-recording_started': 'Recording started',
        'srv-recording_in_progress': 'Recording in progress...',
        'srv-collecting_frames': 'Collecting frames... {percent}%',
        'srv-no_hand': 'No hand detected',
        'srv-no_hand_during_recording': 'No hand detected during recording',
        'srv-hand_detected_ready': 'Hand detected - click Record to start',
        'srv-low_confidence': 'Not confident enough - try again',
        'srv-wrong_letter': "Detected '{detected}' instead of '{expected}'",
        'srv-attempt_failed': 'Attempt failed',
        'srv-correct_next': 'Correct! Moving to letter {letter}',
        'srv-correct_next_sentence': 'Correct! Next letter...',
        'srv-timeout_next': 'Time up! Moving to letter {letter}',
        'srv-free_signed': 'Signed {letter}',
        'srv-skipped_letter': 'Skipped {skipped}, now showing {letter}',
        'srv-sentence_set': 'Target sentence set',
        'srv-free_mode_on': 'Free sign mode activated',
        'srv-sentence_cleared': 'Recognized sentence cleared',
        'srv-model_unavailable': 'Model not available on the server',
        'srv-detection_unavailable': 'Hand detection unavailable on the server',
        'srv-hint_1': "Need help with '{letter}'? Check the example on the side.",
        'srv-hint_2': "Keep trying! Make sure your hand matches the '{letter}' shape."
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
        'status-disconnected': 'Verbinding Verbroken',
        'status-error': 'Verbindingsfout',
        'status-reconnecting': 'Opnieuw verbinden...',
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
        'btn-clear': 'Wis Herkenning',
        'err-camera': 'Geen toegang tot de camera. Controleer de rechten.',
        // Server message keys
        'srv-recording_started': 'Opname gestart',
        'srv-recording_in_progress': 'Bezig met opnemen...',
        'srv-collecting_frames': 'Frames verzamelen... {percent}%',
        'srv-no_hand': 'Geen hand gedetecteerd',
        'srv-no_hand_during_recording': 'Geen hand gedetecteerd tijdens de opname',
        'srv-hand_detected_ready': 'Hand gedetecteerd - klik op Opnemen',
        'srv-low_confidence': 'Niet zeker genoeg - probeer opnieuw',
        'srv-wrong_letter': "'{detected}' gedetecteerd in plaats van '{expected}'",
        'srv-attempt_failed': 'Poging mislukt',
        'srv-correct_next': 'Correct! Door naar letter {letter}',
        'srv-correct_next_sentence': 'Correct! Volgende letter...',
        'srv-timeout_next': 'Tijd is op! Door naar letter {letter}',
        'srv-free_signed': '{letter} gebaard',
        'srv-skipped_letter': '{skipped} overgeslagen, nu {letter}',
        'srv-sentence_set': 'Doelzin ingesteld',
        'srv-free_mode_on': 'Vrije gebarenmodus actief',
        'srv-sentence_cleared': 'Herkenning gewist',
        'srv-model_unavailable': 'Model niet beschikbaar op de server',
        'srv-detection_unavailable': 'Handdetectie niet beschikbaar op de server',
        'srv-hint_1': "Hulp nodig bij '{letter}'? Bekijk het voorbeeld hiernaast.",
        'srv-hint_2': "Blijf proberen! Zorg dat je hand de '{letter}' vorm volgt."
    }
};

/** Look up a translation key and interpolate {placeholders}. */
function t(key, args) {
    const table = translations[currentLanguage] || translations.en;
    let text = table[key];
    if (text === undefined) text = translations.en[key];
    if (text === undefined) return null;
    if (args) {
        for (const [name, value] of Object.entries(args)) {
            text = text.replaceAll(`{${name}}`, value);
        }
    }
    return text;
}

/**
 * Resolve a server-sent message, preferring the localised key over the
 * server's English string.
 */
function serverMessage(data, fallbackKey) {
    const key = data.message_key || fallbackKey;
    if (key) {
        const translated = t(`srv-${key}`, data.message_args || {});
        if (translated !== null) return translated;
    }
    return data.message || '';
}

// Initialize
function init() {
    startBtn.addEventListener('click', startSession);
    recordBtn.addEventListener('click', toggleRecording);
    skipBtn.addEventListener('click', skipLetter);

    if (modeCheckbox) modeCheckbox.addEventListener('change', toggleMode);
    if (languageCheckbox) {
        languageCheckbox.checked = currentLanguage === 'nl';
        languageCheckbox.addEventListener('change', toggleLanguage);
    }

    recordBtn.disabled = true;
    skipBtn.disabled = true;
    if (modeCheckbox) modeCheckbox.disabled = true;
    if (practiceModeSelect) {
        practiceModeSelect.disabled = true;
        practiceModeSelect.addEventListener('change', togglePracticeMode);
    }
    if (setSentenceBtn) setSentenceBtn.addEventListener('click', setTargetSentence);
    if (clearSentenceBtn) clearSentenceBtn.addEventListener('click', clearRecognizedSentence);

    window.addEventListener('beforeunload', shutdown);

    updateModeToggle();
    updateUIForMode(currentMode);
    updateLanguage();
    log('App initialized');
}

function toggleLanguage() {
    currentLanguage = languageCheckbox.checked ? 'nl' : 'en';
    try {
        localStorage.setItem('language', currentLanguage);
    } catch (e) {
        log('Could not persist language preference', e);
    }
    updateLanguage();
}

function updateLanguage() {
    // Keep the document language in sync so screen readers and spellcheckers
    // announce the right language.
    document.documentElement.lang = currentLanguage;

    document.querySelectorAll('[data-i18n]').forEach(element => {
        const translated = t(element.getAttribute('data-i18n'));
        if (translated !== null) element.textContent = translated;
    });

    if (startBtn.disabled) startBtn.textContent = t('btn-running');
    recordBtn.textContent = isRecording ? t('btn-stop-record') : t('btn-record');
}

// WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        log('WebSocket connected');
        reconnectAttempts = 0;
        updateStatus('connected', t('status-connected'));
    };

    ws.onmessage = (event) => {
        let data;
        try {
            data = JSON.parse(event.data);
        } catch (e) {
            console.error('Malformed server message', e);
            return;
        }
        handleServerResponse(data);
    };

    ws.onerror = () => updateStatus('disconnected', t('status-error'));

    ws.onclose = () => {
        log('WebSocket closed');
        if (shuttingDown || !stream) {
            updateStatus('disconnected', t('status-disconnected'));
            return;
        }
        // Exponential backoff, so a server that stays down is not hammered
        // every three seconds forever.
        const delay = Math.min(RECONNECT_BASE_MS * 2 ** reconnectAttempts, RECONNECT_MAX_MS);
        reconnectAttempts += 1;
        updateStatus('disconnected', t('status-reconnecting'));
        setTimeout(() => { if (stream && !shuttingDown) connectWebSocket(); }, delay);
    };
}

function wsReady() {
    return ws && ws.readyState === WebSocket.OPEN;
}

function send(payload) {
    if (!wsReady()) return false;
    ws.send(JSON.stringify({ ...payload, session_id: sessionId }));
    return true;
}

// Mode handling
async function toggleMode() {
    if (!sessionId) {
        modeCheckbox.checked = !modeCheckbox.checked;
        return;
    }
    const newMode = modeCheckbox.checked ? 'random' : 'sequential';
    try {
        const response = await fetch(`/api/session/${sessionId}/mode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: newMode })
        });
        if (response.ok) {
            currentMode = (await response.json()).mode;
        } else {
            modeCheckbox.checked = !modeCheckbox.checked;
        }
    } catch (error) {
        console.error('Error changing mode:', error);
        modeCheckbox.checked = !modeCheckbox.checked;
    }
}

function updateModeToggle() {
    if (modeCheckbox) modeCheckbox.checked = (currentMode === 'random');
    updateUIForMode(currentMode);
}

async function togglePracticeMode() {
    if (!sessionId) return;
    const selectedMode = practiceModeSelect.value;
    const actualMode = selectedMode === 'letter'
        ? (modeCheckbox.checked ? 'random' : 'sequential')
        : 'sentence';

    try {
        const response = await fetch(`/api/session/${sessionId}/mode`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: actualMode })
        });
        if (response.ok) {
            currentMode = actualMode;
            updateUIForMode(selectedMode);
            if (selectedMode === 'free') send({ type: 'set_sentence', sentence: '' });
        }
    } catch (error) {
        console.error('Error changing practice mode:', error);
    }
}

function setTargetSentence() {
    send({ type: 'set_sentence', sentence: targetSentenceInput.value.trim() });
}

function clearRecognizedSentence() {
    send({ type: 'clear_sentence' });
}

function updateUIForMode(modeStr) {
    if (!practiceModeSelect) return;

    if (modeStr === 'sequential' || modeStr === 'random' || modeStr === 'letter') {
        singleLetterUI.style.display = 'block';
        sentenceUI.style.display = 'none';
        modeSwitchContainer.style.display = 'flex';
        practiceModeSelect.value = 'letter';
        return;
    }

    singleLetterUI.style.display = 'none';
    sentenceUI.style.display = 'block';
    modeSwitchContainer.style.display = 'none';

    const inputGroup = document.getElementById('sentenceInputGroup');
    if (modeStr === 'free') {
        inputGroup.style.display = 'none';
        targetSentenceDisplay.style.display = 'none';
        sentenceModeTitle.textContent = t('mode-free-practice');
        practiceModeSelect.value = 'free';
    } else {
        inputGroup.style.display = 'flex';
        targetSentenceDisplay.style.display = 'block';
        sentenceModeTitle.textContent = t('mode-sentence-practice');
        practiceModeSelect.value = 'sentence';
    }
}

function skipLetter() {
    if (!wsReady()) return;
    if (isRecording) stopRecording();
    send({ type: 'skip' });
}

// Session
async function startSession() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        webcamElement.srcObject = stream;

        shuttingDown = false;
        connectWebSocket();
        startCaptureLoop(CAPTURE_FPS_IDLE);

        startBtn.disabled = true;
        recordBtn.disabled = false;
        skipBtn.disabled = false;
        if (modeCheckbox) modeCheckbox.disabled = false;
        if (practiceModeSelect) practiceModeSelect.disabled = false;
        startBtn.textContent = t('btn-running');
    } catch (error) {
        console.error('Failed to start:', error);
        alert(t('err-camera'));
    }
}

/** Release the camera and stop the capture loop. */
function shutdown() {
    shuttingDown = true;
    if (captureTimer) { clearInterval(captureTimer); captureTimer = null; }
    if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
    if (ws) { ws.close(); ws = null; }
}

function toggleRecording() {
    if (isRecording) stopRecording(); else startRecording();
}

function startRecording() {
    if (!wsReady()) return;

    isRecording = true;
    recordBtn.textContent = t('btn-stop-record');
    recordBtn.classList.add('recording');

    successMessage.style.display = 'none';
    timeoutMessage.style.display = 'none';
    hintMessage.style.display = 'none';
    if (recordingProgress) recordingProgress.style.display = 'block';

    startCaptureLoop(CAPTURE_FPS_RECORDING);
    send({ type: 'start_recording' });
}

function stopRecording() {
    isRecording = false;
    recordBtn.textContent = t('btn-record');
    recordBtn.classList.remove('recording');
    if (recordingProgress) recordingProgress.style.display = 'none';

    startCaptureLoop(CAPTURE_FPS_IDLE);
    send({ type: 'stop_recording' });
}

// Frame capture
function startCaptureLoop(fps) {
    if (captureTimer && captureFps === fps) return;
    if (captureTimer) clearInterval(captureTimer);
    captureFps = fps;
    captureTimer = setInterval(() => { if (wsReady()) captureFrame(); }, 1000 / fps);
}

function captureFrame() {
    if (!webcamElement.videoWidth) return;

    const canvas = canvasElement;
    canvas.width = webcamElement.videoWidth;
    canvas.height = webcamElement.videoHeight;
    canvas.getContext('2d').drawImage(webcamElement, 0, 0);

    send({ type: 'frame', frame: canvas.toDataURL('image/jpeg', JPEG_QUALITY) });
}

// Server responses
function handleServerResponse(data) {
    log('Server response:', data);

    if (data.session_id && !sessionId) {
        sessionId = data.session_id;
        log('Session ID:', sessionId);
    }

    if (data.progress) updateProgress(data);

    if (data.hand_detected !== undefined) {
        if (data.hand_detected) {
            updateStatus('hand-detected', t('status-hand-detected'));
        } else {
            updateStatus('connected', serverMessage(data, 'no_hand'));
        }
    }

    if (data.recording !== undefined) updateRecordingState(data);
    if (data.prediction || data.current_prediction) updatePrediction(data);

    if (data.success) showSuccess();
    if (data.timeout) showTimeout();
    if (data.skipped) {
        predictionLetterElement.textContent = '-';
        confidenceElement.textContent = '-';
    }
    if (data.show_hint) {
        const hint = data.hint_key
            ? t(`srv-${data.hint_key}`, { letter: data.progress?.current_letter || '' })
            : null;
        showHint(hint || data.hint_message || '');
    }
}

function updateProgress(data) {
    const prog = data.progress;

    // Both can be null once a sentence is complete; rendering that directly
    // used to print the string "undefined" into the target-letter card.
    targetLetterElement.textContent = prog.current_letter || data.current_letter || NO_LETTER;
    correctCountElement.textContent = prog.total_correct ?? 0;
    totalCountElement.textContent = prog.total_attempts ?? 0;
    accuracyElement.textContent = `${(prog.accuracy ?? 0).toFixed(1)}%`;

    if (prog.time_remaining !== undefined) {
        timeDisplay.textContent = `${Math.ceil(prog.time_remaining)}s`;
    }

    if (prog.tutorial_url && prog.tutorial_url !== currentTutorialUrl) {
        currentTutorialUrl = prog.tutorial_url;
        tutorialGif.src = prog.tutorial_url;
        tutorialGif.style.display = 'block';
    } else if (!prog.tutorial_url && currentTutorialUrl !== null) {
        currentTutorialUrl = null;
        tutorialGif.style.display = 'none';
    }

    if (prog.mode && prog.mode !== currentMode) {
        currentMode = prog.mode;
        updateModeToggle();
    }

    if (prog.target_sentence !== undefined) {
        targetSentenceDisplay.textContent = prog.target_sentence;
    }
    if (prog.recognized_sentence !== undefined) {
        renderSentence(prog.target_sentence || '', prog.recognized_sentence || '', prog.mode);
    }
}

/**
 * Render sentence progress.
 *
 * Built with DOM nodes rather than an innerHTML string: the target text comes
 * from user input, round-tripped through the server.
 */
function renderSentence(target, recognized, mode) {
    recognizedSentenceDisplay.replaceChildren();

    if (!target || mode !== 'sentence') {
        recognizedSentenceDisplay.textContent = recognized;
        return;
    }

    const fragment = document.createDocumentFragment();
    for (let i = 0; i < target.length; i++) {
        const span = document.createElement('span');
        if (target[i] === ' ') {
            span.textContent = ' ';
        } else {
            span.textContent = target[i];
            if (i < recognized.length) span.className = 'correct-char';
            else if (i === recognized.length) span.className = 'current-target-char';
            else span.className = 'pending-char';
        }
        fragment.appendChild(span);
    }
    recognizedSentenceDisplay.appendChild(fragment);
}

function updateRecordingState(data) {
    if (data.recording) {
        statusText.textContent = serverMessage(data, 'recording_in_progress');
        if (recordingProgress) recordingProgress.style.display = 'block';

        const progressPercent = document.getElementById('progressPercent');
        if (progressPercent) {
            progressPercent.textContent = data.buffer_progress !== undefined
                ? ` ${(data.buffer_progress * 100).toFixed(0)}%`
                : '';
        }
        return;
    }

    isRecording = false;
    recordBtn.textContent = t('btn-record');
    recordBtn.classList.remove('recording');
    if (recordingProgress) recordingProgress.style.display = 'none';
    startCaptureLoop(CAPTURE_FPS_IDLE);
}

function updatePrediction(data) {
    const predicted = data.prediction
        ? data.prediction.predicted_class
        : data.current_prediction;
    const confidence = data.prediction ? data.prediction.confidence : data.confidence;

    predictionLetterElement.textContent =
        (!predicted || predicted === 'Nonsense' || predicted === 'Unknown') ? NO_LETTER : predicted;

    // A confidence of exactly 0 is a real value; `if (confidence)` dropped it.
    confidenceElement.textContent =
        confidence === undefined || confidence === null
            ? '-'
            : `${(confidence * 100).toFixed(1)}%`;
}

function updateStatus(status, text) {
    statusBadge.className = `status-badge ${status}`;
    statusText.textContent = text;
}

function flash(element, ms) {
    element.style.display = 'flex';
    setTimeout(() => { element.style.display = 'none'; }, ms);
}

const showSuccess = () => flash(successMessage, 2000);
const showTimeout = () => flash(timeoutMessage, 2000);

function showHint(message) {
    hintText.textContent = message;
    flash(hintMessage, 5000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
