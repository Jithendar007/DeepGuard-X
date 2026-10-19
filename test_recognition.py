import insightface

print("Loading InsightFace...")

app = insightface.app.FaceAnalysis(
    name="buffalo_l"
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("InsightFace model loaded successfully!")