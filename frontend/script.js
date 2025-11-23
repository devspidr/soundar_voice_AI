// frontend/script.js (REPLACE your current file with this exact content)

const recBtn = document.getElementById('record');
const youDiv = document.getElementById('you');
const botDiv = document.getElementById('bot');

const pauseBtn = document.getElementById('pauseBtn');
const resumeBtn = document.getElementById('resumeBtn');
const stopBtn = document.getElementById('stopBtn');

// Backend base URL
const BASE_URL = "http://localhost:8000";

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
if (!SpeechRecognition) {
  alert("SpeechRecognition API not available in this browser. Try Chrome desktop.");
}

const recog = new SpeechRecognition();
recog.lang = "en-IN";
recog.interimResults = false;
recog.maxAlternatives = 1;

// Keep track of the utterance currently being spoken
let currentUtterance = null;

// UI helper to set recording state
function setRecordingState(isRecording) {
  if (isRecording) {
    recBtn.innerText = "Listening...";
    recBtn.disabled = true;
    recBtn.style.opacity = "0.85";
  } else {
    recBtn.innerText = "🎙️ Record (4s)";
    recBtn.disabled = false;
    recBtn.style.opacity = "1";
  }
}

function showUserText(text) {
  youDiv.innerHTML = `<b>You:</b> ${text}`;
}

function showBotText(text) {
  botDiv.innerHTML = `<b>Soundar:</b> ${text}`;
}

// Manage playback control buttons
function enablePlaybackControls(enable) {
  pauseBtn.disabled = !enable;
  resumeBtn.disabled = !enable;
  stopBtn.disabled = !enable;
}

// Update controls when speaking or paused
function updateControlsForSpeaking() {
  pauseBtn.disabled = false;
  resumeBtn.disabled = true;
  stopBtn.disabled = false;
}

function updateControlsForPaused() {
  pauseBtn.disabled = true;
  resumeBtn.disabled = false;
  stopBtn.disabled = false;
}

function clearPlaybackControls() {
  pauseBtn.disabled = true;
  resumeBtn.disabled = true;
  stopBtn.disabled = true;
}

// Recognition events
recog.onstart = () => setRecordingState(true);
recog.onend = () => setRecordingState(false);

recog.onresult = async (e) => {
  const text = (e.results && e.results[0] && e.results[0][0] && e.results[0][0].transcript) || "";
  showUserText(text);
  console.log("Transcribed:", text);

  const payload = {
    user_id: "local_user",
    transcript: text,
    conversation_history: []
  };

  let respJson;
  try {
    const res = await fetch(`${BASE_URL}/api/voice_text`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errText = await res.text();
      console.error("Backend returned non-OK:", res.status, errText);
      showBotText(`(backend error ${res.status})`);
      return;
    }

    respJson = await res.json();
    console.log("Backend returned:", respJson);
  } catch (err) {
    console.error("Fetch error:", err);
    showBotText("(network or backend error)");
    return;
  }

  const answer = (respJson && (respJson.text || respJson.answer || respJson.message)) || "(no reply)";
  showBotText(answer);

  // Stop any existing speech and cancel previous utterance
  try {
    if (window.speechSynthesis && window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
    }
  } catch (e) {
    console.warn("Could not cancel previous speech:", e);
  }

  // Create a new utterance and speak it
  try {
    if ("speechSynthesis" in window && answer && answer !== "(no reply)") {
      const utter = new SpeechSynthesisUtterance(answer);
      utter.lang = "en-IN";
      utter.rate = 1.0;
      utter.pitch = 1.0;

      // set currentUtterance so controls can pause/resume/stop it
      currentUtterance = utter;

      // When utterance ends, clear controls
      utter.onend = function() {
        currentUtterance = null;
        clearPlaybackControls();
      };

      utter.onerror = function(e) {
        console.warn("SpeechSynthesis utterance error:", e);
        currentUtterance = null;
        clearPlaybackControls();
      };

      // start speaking and enable controls
      window.speechSynthesis.speak(utter);
      updateControlsForSpeaking();
    } else {
      clearPlaybackControls();
    }
  } catch (e) {
    console.warn("speechSynthesis failed:", e);
    clearPlaybackControls();
  }
};

// Start recognition when user clicks
recBtn.onclick = () => {
  try {
    setRecordingState(true);
    recog.start();
    // safety: stop after ~4.5s to avoid indefinite listening
    setTimeout(() => {
      try { recog.stop(); } catch(e) {}
    }, 4500);
  } catch (err) {
    console.error("Failed to start recognition:", err);
    setRecordingState(false);
  }
};

// Playback control handlers
pauseBtn.onclick = () => {
  try {
    if (window.speechSynthesis && window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
      window.speechSynthesis.pause();
      updateControlsForPaused();
    }
  } catch (e) {
    console.warn("Pause failed:", e);
  }
};

resumeBtn.onclick = () => {
  try {
    if (window.speechSynthesis && window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
      updateControlsForSpeaking();
    }
  } catch (e) {
    console.warn("Resume failed:", e);
  }
};

stopBtn.onclick = () => {
  try {
    if (window.speechSynthesis && (window.speechSynthesis.speaking || window.speechSynthesis.paused)) {
      window.speechSynthesis.cancel();
      currentUtterance = null;
      clearPlaybackControls();
    }
  } catch (e) {
    console.warn("Stop failed:", e);
  }
};

// init control states
clearPlaybackControls();
