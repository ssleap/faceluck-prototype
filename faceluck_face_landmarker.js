// Opt-in IMAGE PoC. Only static assets are fetched; input is a local blob URL.
const version = '0.10.22-rc.20250304';
const base = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${version}`;
const model = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
let landmarkerPromise;

function getLandmarker() {
  landmarkerPromise ??= (async () => {
    const { FaceLandmarker, FilesetResolver } = await import(`${base}/+esm`);
    const files = await FilesetResolver.forVisionTasks(`${base}/wasm`);
    return FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: model, delegate: 'CPU' },
      runningMode: 'IMAGE',
      numFaces: 10,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
  })().catch(() => {
    landmarkerPromise = undefined;
    throw new Error('landmarker_unavailable');
  });
  return landmarkerPromise;
}

window.faceluckMeasureFace = async function(imageUrl) {
  // Reject remote input even if a future caller accidentally supplies it.
  if (!imageUrl.startsWith('blob:')) return JSON.stringify({ status: 'unavailable' });
  const image = new Image();
  try {
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = imageUrl;
    });
    const landmarker = await getLandmarker();
    const result = landmarker.detect(image);
    const faces = result.faceLandmarks || [];
    // Never expose/store the full landmark array. Single-face only: detector
    // selection order is not assumed to match Landmarker face order.
    const anchors = {};
    if (faces.length === 1) {
      for (const index of [10, 152, 234, 454, 133, 362, 98, 327, 61, 291]) {
        const point = faces[0][index];
        if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
          anchors[index] = [point.x, point.y];
        }
      }
    }
    return JSON.stringify({
      status: 'ok', faceCount: faces.length,
      width: image.naturalWidth, height: image.naturalHeight, anchors,
    });
  } catch (_) {
    // No error/image/coordinate logging and no synthetic success values.
    return JSON.stringify({ status: 'unavailable' });
  } finally {
    image.onload = null;
    image.onerror = null;
    image.removeAttribute('src');
  }
};
