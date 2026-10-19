import insightface
import numpy as np
import os


# ============================================================
# FACE IDENTIFICATION ENGINE
# ============================================================

class FaceIdentifier:

    def __init__(self):

        # ----------------------------------------------------
        # Load InsightFace ONCE
        # ----------------------------------------------------

        print("\nLoading InsightFace...")

        self.face_app = insightface.app.FaceAnalysis(
            name="buffalo_l"
        )

        self.face_app.prepare(
            ctx_id=-1,
            det_size=(320, 320)
        )

        # ----------------------------------------------------
        # Load registered people ONCE
        # ----------------------------------------------------

        self.registered_people = {}

        DATA_FOLDER = "data"

        if not os.path.exists(DATA_FOLDER):

            print("No registered users found.")

            return

        for person_name in os.listdir(DATA_FOLDER):

            person_folder = os.path.join(
                DATA_FOLDER,
                person_name
            )

            embedding_path = os.path.join(
                person_folder,
                "embedding.npy"
            )

            if os.path.isfile(embedding_path):

                embedding = np.load(
                    embedding_path
                )

                # Normalize stored embedding
                norm = np.linalg.norm(
                    embedding
                )

                if norm > 0:

                    embedding = (
                        embedding / norm
                    )

                self.registered_people[
                    person_name
                ] = embedding

        print(
            f"Registered people loaded: "
            f"{len(self.registered_people)}"
        )

        # ----------------------------------------------------
        # Settings
        # ----------------------------------------------------

        self.THRESHOLD = 0.45

        # ----------------------------------------------------
        # Score storage
        # ----------------------------------------------------

        self.person_scores = {
            person_name: []
            for person_name
            in self.registered_people
        }

    # ========================================================
    # PROCESS ONE FRAME
    # ========================================================

    def process_frame(self, frame):

        if len(self.registered_people) == 0:

            return

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = self.face_app.get(frame)

        if len(faces) == 0:

            return

        # ----------------------------------------------------
        # Select largest face
        # ----------------------------------------------------

        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        # ----------------------------------------------------
        # Current face embedding
        # ----------------------------------------------------

        current_embedding = face.embedding

        # Normalize
        norm = np.linalg.norm(
            current_embedding
        )

        if norm == 0:

            return

        current_embedding = (
            current_embedding / norm
        )

        # ----------------------------------------------------
        # Compare with registered users
        # ----------------------------------------------------

        for (
            person_name,
            stored_embedding
        ) in self.registered_people.items():

            similarity = np.dot(
                stored_embedding,
                current_embedding
            )

            self.person_scores[
                person_name
            ].append(
                float(similarity)
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    def get_result(self):

        valid_people = {}

        for (
            person_name,
            scores
        ) in self.person_scores.items():

            if len(scores) > 0:

                mean_score = np.mean(
                    np.array(scores)
                )

                valid_people[
                    person_name
                ] = mean_score

        # ----------------------------------------------------
        # No valid samples
        # ----------------------------------------------------

        if len(valid_people) == 0:

            print("\nNo valid face samples were obtained.")

            return None, 0.0

        # ----------------------------------------------------
        # Find best identity
        # ----------------------------------------------------

        best_name = max(
            valid_people,
            key=valid_people.get
        )

        final_identity_score = (
            valid_people[best_name]
        )

        best_scores = np.array(
            self.person_scores[best_name]
        )

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print("\n======================================")
        print("       FINAL IDENTITY RESULT")
        print("======================================")

        print(
            f"Best identity : "
            f"{best_name}"
        )

        print(
            f"Values used   : "
            f"{len(best_scores)}"
        )

        print(
            "Frame scores  :",
            np.round(
                best_scores,
                4
            )
        )

        print("--------------------------------------")

        print(
            f"Mean score    : "
            f"{final_identity_score:.4f}"
        )

        print(
            f"Percentage    : "
            f"{final_identity_score * 100:.2f}%"
        )

        print(
            f"Threshold     : "
            f"{self.THRESHOLD:.2f}"
        )

        print("--------------------------------------")

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if final_identity_score >= self.THRESHOLD:

            print(
                "FINAL DECISION: MATCH"
            )

            print(
                "======================================"
            )

            return (
                best_name,
                float(final_identity_score)
            )

        else:

            print(
                "FINAL DECISION: UNKNOWN"
            )

            print(
                "======================================"
            )

            return (
                None,
                float(final_identity_score)
            )


# ============================================================
# OPTIONAL COMPATIBILITY FUNCTION
# ============================================================

def recognize_user(selected_frames):

    identifier = FaceIdentifier()

    for item in selected_frames:

        frame = item[0]

        identifier.process_frame(frame)

    return identifier.get_result()