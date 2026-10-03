// MediaPipe runs entirely in the browser. The model/WASM files are downloaded,
// but image bytes and detection results never leave this page.
const tasksVisionVersion = "0.10.22-rc.20250304";
const tasksVisionBase =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${tasksVisionVersion}`;
const modelUrl =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite";

let detectorPromise;

async function createDetector() {
  const { FaceDetector, FilesetResolver } = await import(
    `${tasksVisionBase}/+esm`
  );
  const vision = await FilesetResolver.forVisionTasks(`${tasksVisionBase}/wasm`);
  const options = {
    baseOptions: { modelAssetPath: modelUrl },
    runningMode: "IMAGE",
    minDetectionConfidence: 0.55,
    minSuppressionThreshold: 0.3,
  };

  // GPU is preferred on mobile browsers; CPU remains an explicit local fallback.
  try {
    return await FaceDetector.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "GPU" },
    });
  } catch (gpuError) {
    console.warn("MediaPipe GPU delegate unavailable; using CPU.", gpuError);
    return FaceDetector.createFromOptions(vision, options);
  }
}

function getDetector() {
  detectorPromise ??= createDetector().catch((error) => {
    detectorPromise = undefined;
    throw error;
  });
  return detectorPromise;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The selected image could not be decoded."));
    image.src = url;
  });
}

window.faceluckDetectFaces = async function faceluckDetectFaces(imageUrl) {
  const image = await loadImage(imageUrl);
  const detector = await getDetector();
  const result = detector.detect(image);
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  const clamp = (value) => Math.max(0, Math.min(1, value));
  const faces = (result.detections || []).map((detection) => {
    const box = detection.boundingBox;
    const left = clamp(box.originX / width);
    const top = clamp(box.originY / height);
    const right = clamp((box.originX + box.width) / width);
    const bottom = clamp((box.originY + box.height) / height);
    return {
      left,
      top,
      width: right - left,
      height: bottom - top,
      confidence: detection.categories?.[0]?.score || 0,
    };
  });
  return JSON.stringify({ width, height, faces });
};
