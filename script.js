// ===============================
// Backend URL
// ===============================

const API_URL = "http://127.0.0.1:8000";

// ===============================
// Controlled Vocabulary
// ===============================

const SUPPORTED_WORDS = [
  "and", "bad", "because", "brother", "come", "day", "drink", "eat",
  "family", "father", "friend", "get", "go", "good", "happy", "have",
  "help", "home", "how", "i", "know", "language", "learn", "like",
  "make", "mother", "name", "need", "no", "not", "now", "or", "people",
  "person", "sad", "say", "school", "see", "she", "sign", "sister",
  "sleep", "study", "teach", "tell", "they", "think", "time", "today",
  "tomorrow", "want", "we", "week", "what", "when", "where", "why",
  "with", "work", "yes", "yesterday", "you"
];

const SUPPORTED_SET = new Set(SUPPORTED_WORDS);

const WORD_NORMALIZATION = {
  "me": "i",
  "my": "i",
  "i'm": "i",
  "im": "i",

  "wants": "want",
  "wanted": "want",
  "wanting": "want",

  "learns": "learn",
  "learned": "learn",
  "learning": "learn",

  "signing": "sign",
  "signed": "sign",
  "signs": "sign",

  "languages": "language",

  "helped": "help",
  "helping": "help",
  "helps": "help",

  "studies": "study",
  "studied": "study",
  "studying": "study",

  "worked": "work",
  "working": "work",
  "works": "work",

  "teachers": "teach",
  "teaching": "teach",
  "taught": "teach",

  "thinks": "think",
  "thinking": "think",
  "thought": "think",

  "goes": "go",
  "going": "go",
  "went": "go",

  "comes": "come",
  "coming": "come",

  "eating": "eat",
  "ate": "eat",
  "eats": "eat",

  "drinking": "drink",
  "drank": "drink",
  "drinks": "drink",

  "says": "say",
  "said": "say",
  "saying": "say",

  "seeing": "see",
  "saw": "see",
  "seen": "see",

  "slept": "sleep",
  "sleeping": "sleep",

  "needed": "need",
  "needing": "need",
  "needs": "need",

  "liked": "like",
  "likes": "like",
  "liking": "like",

  "made": "make",
  "making": "make",
  "makes": "make",

  "having": "have",
  "has": "have",
  "had": "have",

  "friends": "friend",
  "brothers": "brother",
  "sisters": "sister",
  "mothers": "mother",
  "fathers": "father",
  "persons": "person",
  "families": "family",

  "okay": "good",
  "ok": "good",
  "great": "good",

  "yeah": "yes",
  "yep": "yes",
  "nope": "no"
};

const FILLER_WORDS = new Set([
  "a", "an", "the", "to", "am", "is", "are", "was", "were", "be",
  "been", "being", "do", "does", "did", "very", "really", "please",
  "can", "could", "would", "should", "will", "shall", "it", "its",
  "this", "that", "these", "those", "of", "for", "in", "on", "at",
  "from", "as", "by", "about"
]);

// ===============================
// DOM Elements
// ===============================

const loadingScreen = document.getElementById("loadingScreen");
const app = document.getElementById("app");

const speechToSignTab = document.getElementById("speechToSignTab");
const signToSpeechTab = document.getElementById("signToSpeechTab");
const speechToSignView = document.getElementById("speechToSignView");
const signToSpeechView = document.getElementById("signToSpeechView");
const modeSubtitle = document.getElementById("modeSubtitle");

const micButton = document.getElementById("micButton");
const micButtonText = document.getElementById("micButtonText");

const typedInput = document.getElementById("typedInput");
const translateTypedButton = document.getElementById("translateTypedButton");

const avatarVideo = document.getElementById("avatarVideo");
const idlePlaceholder = document.getElementById("idlePlaceholder");

const transcriptBox = document.getElementById("transcriptBox");
const signSequence = document.getElementById("signSequence");
const currentSign = document.getElementById("currentSign");

const replayButton = document.getElementById("replayButton");
const clearButton = document.getElementById("clearButton");

const statusText = document.getElementById("statusText");
const statusDot = document.getElementById("statusDot");

// Sign to Speech DOM
const openCameraButton = document.getElementById("openCameraButton");
const startSigningButton = document.getElementById("startSigningButton");
const stopSigningButton = document.getElementById("stopSigningButton");
const signVideoInput = document.getElementById("signVideoInput");
const predictVideoButton = document.getElementById("predictVideoButton");
const predictedSpeechBox = document.getElementById("predictedSpeechBox");
const detectedSignsBox = document.getElementById("detectedSignsBox");
const topPredictionsBox = document.getElementById("topPredictionsBox");
const speakPredictionButton = document.getElementById("speakPredictionButton");
const clearSignButton = document.getElementById("clearSignButton");
const signStatusLabel = document.getElementById("signStatusLabel");

const cameraPreview = document.getElementById("cameraPreview");
const trackingCanvas = document.getElementById("trackingCanvas");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");

// ===============================
// App State
// ===============================

let recognition = null;
let isListening = false;
let isPlaying = false;
let lastPlayableWords = [];
let lastSignPrediction = "";

let cameraStream = null;
let handsModel = null;
let poseModel = null;

let latestHandResults = null;
let latestPoseResults = null;

let handTrackingActive = false;
let handTrackingAnimationId = null;

let mediaRecorder = null;
let recordedChunks = [];
let lastRecordedVideoBlob = null;

// ===============================
// Loading Screen
// ===============================

window.addEventListener("load", () => {
  setTimeout(() => {
    loadingScreen.classList.add("hidden");
    app.classList.remove("hidden");
  }, 1200);
});

// ===============================
// General Helpers
// ===============================

function setStatus(message, type = "ready") {
  statusText.textContent = message;

  if (type === "error") {
    statusDot.style.background = "var(--danger)";
    statusDot.style.boxShadow = "0 0 20px rgba(255, 92, 122, 0.8)";
  } else if (type === "working") {
    statusDot.style.background = "var(--accent-2)";
    statusDot.style.boxShadow = "0 0 20px rgba(25, 211, 255, 0.8)";
  } else {
    statusDot.style.background = "var(--success)";
    statusDot.style.boxShadow = "0 0 20px rgba(69, 240, 160, 0.8)";
  }
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function speakText(text) {
  if (!text) return;

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";

  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

// ===============================
// Speech to Sign Helpers
// ===============================

function cleanText(text) {
  return text
    .toLowerCase()
    .replace(/[^\w\s']/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeWord(word) {
  if (WORD_NORMALIZATION[word]) {
    return WORD_NORMALIZATION[word];
  }

  if (SUPPORTED_SET.has(word)) {
    return word;
  }

  if (word.endsWith("ing")) {
    const base = word.slice(0, -3);
    if (SUPPORTED_SET.has(base)) return base;

    const baseWithE = base + "e";
    if (SUPPORTED_SET.has(baseWithE)) return baseWithE;
  }

  if (word.endsWith("ed")) {
    const base = word.slice(0, -2);
    if (SUPPORTED_SET.has(base)) return base;
  }

  if (word.endsWith("s")) {
    const base = word.slice(0, -1);
    if (SUPPORTED_SET.has(base)) return base;
  }

  return word;
}

function sentenceToPlayableWords(sentence) {
  const cleaned = cleanText(sentence);

  if (!cleaned) {
    return [];
  }

  const rawWords = cleaned.split(" ");
  const playable = [];

  for (const rawWord of rawWords) {
    if (FILLER_WORDS.has(rawWord)) {
      continue;
    }

    const normalized = normalizeWord(rawWord);

    if (FILLER_WORDS.has(normalized)) {
      continue;
    }

    if (SUPPORTED_SET.has(normalized)) {
      playable.push(normalized);
    }
  }

  return playable;
}

function renderSignSequence(words, activeIndex = -1) {
  signSequence.innerHTML = "";

  if (!words.length) {
    signSequence.innerHTML = '<span class="empty">No signs yet.</span>';
    return;
  }

  words.forEach((word, index) => {
    const chip = document.createElement("span");
    chip.className = "sign-chip";

    if (index === activeIndex) {
      chip.classList.add("active");
    }

    chip.textContent = word;
    signSequence.appendChild(chip);
  });
}

function showAvatarVideo() {
  idlePlaceholder.style.display = "none";
  avatarVideo.style.display = "block";
}

function showIdleAvatar() {
  avatarVideo.style.display = "none";
  idlePlaceholder.style.display = "grid";
}

// ===============================
// Tab Switching
// ===============================

function switchMode(mode) {
  const isSpeechToSign = mode === "speechToSign";

  speechToSignTab.classList.toggle("active", isSpeechToSign);
  signToSpeechTab.classList.toggle("active", !isSpeechToSign);

  speechToSignView.classList.toggle("active", isSpeechToSign);
  signToSpeechView.classList.toggle("active", !isSpeechToSign);

  if (isSpeechToSign) {
    modeSubtitle.textContent = "Speech → Sign demo";
    setStatus("Speech to Sign mode ready", "ready");
  } else {
    modeSubtitle.textContent = "Sign → Speech demo";
    setStatus("Sign to Speech mode ready", "ready");
  }
}

speechToSignTab.addEventListener("click", () => {
  switchMode("speechToSign");
});

signToSpeechTab.addEventListener("click", () => {
  switchMode("signToSpeech");
});

// ===============================
// Speech to Sign Pipeline
// ===============================

function processSentence(sentence) {
  if (isPlaying) {
    setStatus("Please wait until current signing is finished.", "working");
    return;
  }

  const cleanSentence = sentence.trim();

  if (!cleanSentence) {
    transcriptBox.textContent = "No speech captured.";
    currentSign.textContent = "Waiting for speech";
    setStatus("Please enter or speak a sentence.", "error");
    return;
  }

  transcriptBox.textContent = cleanSentence;
  setStatus("Processing sentence...", "working");

  const playableWords = sentenceToPlayableWords(cleanSentence);
  lastPlayableWords = playableWords;

  renderSignSequence(playableWords);
  replayButton.disabled = playableWords.length === 0;

  if (!playableWords.length) {
    currentSign.textContent = "No supported signs found";
    setStatus("No supported signs found in this sentence.", "error");
    return;
  }

  playSignQueue(playableWords);
}

async function playSignQueue(words) {
  isPlaying = true;
  micButton.disabled = true;
  translateTypedButton.disabled = true;
  replayButton.disabled = true;

  setStatus("Signing sentence...", "working");
  showAvatarVideo();

  for (let index = 0; index < words.length; index++) {
    const word = words[index];

    renderSignSequence(words, index);
    currentSign.textContent = `Signing: ${word.toUpperCase()}`;

    try {
      await playSingleSign(word);
    } catch (error) {
      console.error(error);
      setStatus(`Missing or failed video: ${word}.mp4`, "error");
      currentSign.textContent = `Could not play: ${word}`;
      await wait(450);
    }
  }

  renderSignSequence(words);
  currentSign.textContent = "Translation complete";
  setStatus("Translation complete", "ready");

  isPlaying = false;
  micButton.disabled = false;
  translateTypedButton.disabled = false;
  replayButton.disabled = words.length === 0;
}

function playSingleSign(word) {
  return new Promise((resolve, reject) => {
    const videoPath = `videos/${word}.mp4`;

    avatarVideo.onended = () => {
      resolve();
    };

    avatarVideo.onerror = () => {
      reject(new Error(`Video not found or cannot be played: ${videoPath}`));
    };

    avatarVideo.src = videoPath;
    avatarVideo.currentTime = 0;

    const playPromise = avatarVideo.play();

    if (playPromise !== undefined) {
      playPromise.catch(reject);
    }
  });
}

// ===============================
// Speech Recognition
// ===============================

function setupSpeechRecognition() {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    micButton.disabled = true;
    micButtonText.textContent = "Mic not supported";
    setStatus("Speech recognition is not supported in this browser. Use typed input.", "error");
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.continuous = false;

  recognition.onstart = () => {
    isListening = true;
    micButton.classList.add("listening");
    micButtonText.textContent = "Listening...";
    setStatus("Listening...", "working");
  };

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    processSentence(transcript);
  };

  recognition.onerror = (event) => {
    setStatus(`Speech recognition error: ${event.error}`, "error");
  };

  recognition.onend = () => {
    isListening = false;
    micButton.classList.remove("listening");
    micButtonText.textContent = "Start Listening";
  };
}

// ===============================
// Sign to Speech: Hand Tracking
// ===============================

// ===============================
// Sign to Speech: Hand + Pose Tracking
// ===============================

async function setupHandTracking() {
  if (typeof Hands === "undefined" || typeof Pose === "undefined") {
    setStatus("MediaPipe tracking did not load. Check your internet connection.", "error");
    signStatusLabel.textContent = "MediaPipe loading error";
    return false;
  }

  if (handsModel && poseModel) {
    return true;
  }

  handsModel = new Hands({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
  });

  handsModel.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6
  });

  handsModel.onResults((results) => {
    latestHandResults = results;
    drawCombinedTracking();
  });

  poseModel = new Pose({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
    }
  });

  poseModel.setOptions({
    modelComplexity: 0,
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6
  });

  poseModel.onResults((results) => {
    latestPoseResults = results;
    drawCombinedTracking();
  });

  return true;
}

function drawCombinedTracking() {
  const canvasContext = trackingCanvas.getContext("2d");

  const videoWidth = cameraPreview.videoWidth || 1280;
  const videoHeight = cameraPreview.videoHeight || 720;

  if (trackingCanvas.width !== videoWidth || trackingCanvas.height !== videoHeight) {
    trackingCanvas.width = videoWidth;
    trackingCanvas.height = videoHeight;
  }

  canvasContext.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);

  const hasPose =
    latestPoseResults &&
    latestPoseResults.poseLandmarks;

  const hasHands =
    latestHandResults &&
    latestHandResults.multiHandLandmarks &&
    latestHandResults.multiHandLandmarks.length > 0;

  if (hasPose) {
    drawConnectors(canvasContext, latestPoseResults.poseLandmarks, POSE_CONNECTIONS, {
      color: "rgba(124, 92, 255, 0.85)",
      lineWidth: 3
    });

    drawLandmarks(canvasContext, latestPoseResults.poseLandmarks, {
      color: "#ffffff",
      lineWidth: 1,
      radius: 2
    });
  }

  if (hasHands) {
    for (const landmarks of latestHandResults.multiHandLandmarks) {
      drawConnectors(canvasContext, landmarks, HAND_CONNECTIONS, {
        color: "#19d3ff",
        lineWidth: 4
      });

      drawLandmarks(canvasContext, landmarks, {
        color: "#45f0a0",
        lineWidth: 2,
        radius: 4
      });
    }
  }

  if (hasHands && hasPose) {
    signStatusLabel.textContent = "Hand and body detected";
  } else if (hasHands) {
    signStatusLabel.textContent = "Hand detected";
  } else if (hasPose) {
    signStatusLabel.textContent = "Body detected — show your hand";
  } else {
    signStatusLabel.textContent = "Show your hand";
  }
}

function startHandTrackingLoop() {
  if (handTrackingActive) {
    return;
  }

  handTrackingActive = true;

  async function detectFrame() {
    if (!handTrackingActive) {
      return;
    }

    if (
      cameraPreview.readyState >= 2 &&
      cameraPreview.videoWidth > 0
    ) {
      if (poseModel) {
        await poseModel.send({
          image: cameraPreview
        });
      }

      if (handsModel) {
        await handsModel.send({
          image: cameraPreview
        });
      }
    }

    handTrackingAnimationId = requestAnimationFrame(detectFrame);
  }

  detectFrame();
}

function stopHandTrackingLoop() {
  handTrackingActive = false;

  if (handTrackingAnimationId) {
    cancelAnimationFrame(handTrackingAnimationId);
    handTrackingAnimationId = null;
  }

  latestHandResults = null;
  latestPoseResults = null;

  const canvasContext = trackingCanvas.getContext("2d");
  canvasContext.clearRect(0, 0, trackingCanvas.width, trackingCanvas.height);
}
// ===============================
// Sign to Speech: Camera
// ===============================

async function openCamera() {
  try {
    setStatus("Opening webcam...", "working");
    signStatusLabel.textContent = "Opening camera...";

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Camera API is not available in this browser.");
    }

    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    cameraPreview.srcObject = cameraStream;
    await cameraPreview.play();

    cameraPreview.style.display = "block";
    trackingCanvas.style.display = "block";
    cameraPlaceholder.style.display = "none";

    startSigningButton.disabled = false;
    stopSigningButton.disabled = true;

    const trackingReady = await setupHandTracking();

    if (trackingReady) {
      startHandTrackingLoop();
      signStatusLabel.textContent = "Show your hand";
      setStatus("Webcam and hand tracking active", "ready");
    } else {
      signStatusLabel.textContent = "Camera ready";
      setStatus("Webcam opened without hand tracking", "ready");
    }

  } catch (error) {
    console.error(error);
    signStatusLabel.textContent = "Camera error";
    setStatus("Could not open webcam. Please allow camera permission.", "error");
  }
}

// ===============================
// Sign to Speech: Backend Rendering
// ===============================

function renderTopPredictions(predictions) {
  topPredictionsBox.innerHTML = "";

  if (!predictions || predictions.length === 0) {
    topPredictionsBox.innerHTML = '<span class="empty">No prediction yet.</span>';
    return;
  }

  predictions.forEach(item => {
    const row = document.createElement("div");
    row.className = "prediction-row";

    const word = document.createElement("span");
    word.className = "prediction-word";
    word.textContent = item.word;

    const score = document.createElement("span");
    score.className = "prediction-score";
    score.textContent = `${(item.confidence * 100).toFixed(1)}%`;

    row.appendChild(word);
    row.appendChild(score);
    topPredictionsBox.appendChild(row);
  });
}

function renderDetectedSign(word) {
  detectedSignsBox.innerHTML = "";

  const chip = document.createElement("span");
  chip.className = "sign-chip active";
  chip.textContent = word;

  detectedSignsBox.appendChild(chip);
}

async function sendVideoToBackend(videoFile, filename) {
  const formData = new FormData();
  formData.append("file", videoFile, filename);

  try {
    setStatus("Sending video to backend...", "working");
    signStatusLabel.textContent = "Predicting...";
    predictedSpeechBox.textContent = "Processing video. Please wait...";
    topPredictionsBox.innerHTML = '<span class="empty">Waiting for backend result...</span>';

    const response = await fetch(`${API_URL}/predict-video`, {
      method: "POST",
      body: formData
    });

    const result = await response.json();

    if (!response.ok || result.error) {
      throw new Error(result.error || "Backend prediction failed.");
    }

    lastSignPrediction = result.english || result.word || "";

    predictedSpeechBox.textContent = result.english || result.word || "No prediction returned.";
    renderDetectedSign(result.word || "UNKNOWN");
    renderTopPredictions(result.top_predictions);

    speakPredictionButton.disabled = false;
    signStatusLabel.textContent = "Prediction complete";
    setStatus("Prediction received from backend", "ready");
    setTimeout(() => {
    speakText(lastSignPrediction);
    }, 300);
  } catch (error) {
    console.error(error);
    signStatusLabel.textContent = "Prediction error";
    predictedSpeechBox.textContent = "Could not get prediction from backend.";
    topPredictionsBox.innerHTML = '<span class="empty">Backend connection failed.</span>';
    setStatus("Backend error. Make sure server is running.", "error");
  }
}

// ===============================
// Sign to Speech: Recording
// ===============================

function getBestRecorderMimeType() {
  const types = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm"
  ];

  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }

  return "";
}

function startSignRecording() {
  if (!cameraStream) {
    setStatus("Please open camera first.", "error");
    signStatusLabel.textContent = "Camera not open";
    return;
  }

  recordedChunks = [];
  lastRecordedVideoBlob = null;

  const mimeType = getBestRecorderMimeType();
  const options = mimeType ? { mimeType } : undefined;

  try {
    mediaRecorder = new MediaRecorder(cameraStream, options);
  } catch (error) {
    console.error(error);
    setStatus("Could not start recording in this browser.", "error");
    signStatusLabel.textContent = "Recording error";
    return;
  }

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onstop = async () => {
    lastRecordedVideoBlob = new Blob(recordedChunks, {
      type: mimeType || "video/webm"
    });

    signStatusLabel.textContent = "Sign clip recorded";
    predictedSpeechBox.textContent = "Recorded clip is ready. Sending to backend...";
    detectedSignsBox.innerHTML = '<span class="sign-chip">CLIP READY</span>';
    topPredictionsBox.innerHTML = '<span class="empty">Sending recorded clip...</span>';

    startSigningButton.disabled = false;
    stopSigningButton.disabled = true;

    setStatus("Sign clip recorded. Sending to backend...", "working");

    await sendVideoToBackend(lastRecordedVideoBlob, "recorded_sign.webm");
  };

  mediaRecorder.start();

  startSigningButton.disabled = true;
  stopSigningButton.disabled = false;

  signStatusLabel.textContent = "Recording sign...";
  predictedSpeechBox.textContent = "Recording your sign. Click Stop when finished.";
  detectedSignsBox.innerHTML = '<span class="sign-chip active">RECORDING</span>';
  topPredictionsBox.innerHTML = '<span class="empty">Recording...</span>';

  setStatus("Recording sign clip...", "working");
}

function stopSignRecording() {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }
}

// ===============================
// Buttons: Speech to Sign
// ===============================

micButton.addEventListener("click", () => {
  if (!recognition || isPlaying) return;

  if (isListening) {
    recognition.stop();
  } else {
    recognition.start();
  }
});

translateTypedButton.addEventListener("click", () => {
  const sentence = typedInput.value.trim();
  processSentence(sentence);
});

replayButton.addEventListener("click", () => {
  if (lastPlayableWords.length && !isPlaying) {
    playSignQueue(lastPlayableWords);
  }
});

clearButton.addEventListener("click", () => {
  if (isPlaying) return;

  typedInput.value = "";
  transcriptBox.textContent = "No speech captured yet.";
  signSequence.innerHTML = '<span class="empty">No signs yet.</span>';
  currentSign.textContent = "Waiting for speech";
  replayButton.disabled = true;
  lastPlayableWords = [];
  showIdleAvatar();
  setStatus("System ready", "ready");
});

// ===============================
// Buttons: Sign to Speech
// ===============================

openCameraButton.addEventListener("click", openCamera);

startSigningButton.addEventListener("click", startSignRecording);

stopSigningButton.addEventListener("click", stopSignRecording);

predictVideoButton.addEventListener("click", async () => {
  const file = signVideoInput.files[0];

  if (!file) {
    setStatus("Please choose a video file first.", "error");
    predictedSpeechBox.textContent = "No video selected.";
    return;
  }

  await sendVideoToBackend(file, file.name);
});

speakPredictionButton.addEventListener("click", () => {
  if (lastSignPrediction) {
    speakText(lastSignPrediction);
  }
});

clearSignButton.addEventListener("click", () => {
  signVideoInput.value = "";
  predictedSpeechBox.textContent = "No sign predicted yet.";
  detectedSignsBox.innerHTML = '<span class="empty">No signs yet.</span>';
  topPredictionsBox.innerHTML = '<span class="empty">No prediction yet.</span>';
  signStatusLabel.textContent = "Waiting for sign";
  speakPredictionButton.disabled = true;
  lastSignPrediction = "";
  setStatus("Sign to Speech mode ready", "ready");
});

// ===============================
// Init
// ===============================

setupSpeechRecognition();
showIdleAvatar();
switchMode("speechToSign");