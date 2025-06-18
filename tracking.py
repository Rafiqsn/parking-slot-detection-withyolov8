import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from cnn_resnet18 import extract_feature_from_bbox


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, feature, kf=None):
        # bbox: [x1, y1, x2, y2]
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        self.kf.statePre[:4, 0] = np.array(bbox, dtype=np.float32)
        self.kf.statePre[4:, 0] = 0

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.feature = feature

    def update(self, bbox, feature):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.feature = feature
        self.kf.correct(np.array(bbox, dtype=np.float32))
        self.history = []

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.statePost[:4, 0])
        return self.kf.statePost[:4, 0]

    def get_state(self):
        return self.kf.statePost[:4, 0]


def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        - wh
        + 1e-6
    )
    return o


class Tracker:
    def __init__(
        self,
        feature_model,
        max_age=10,
        min_hits=2,
        iou_threshold=0.2,
        cos_threshold=0.7,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.cos_threshold = cos_threshold
        self.trackers = []
        self.frame_count = 0
        self.feature_model = feature_model

    def update(self, frame, detections):
        """
        detections: list of [x1, y1, x2, y2]
        """
        self.frame_count += 1

        # Predict new locations for all trackers
        trks = []
        for t in self.trackers:
            pos = t.predict()
            trks.append(pos)

        # Extract features for detections
        features = []
        for det in detections:
            feat = extract_feature_from_bbox(frame, det, self.feature_model)
            features.append(feat)

        # Compute cost matrix (cosine distance + IOU)
        cost_matrix = np.zeros((len(trks), len(detections)), dtype=np.float32)
        for t, tracker in enumerate(self.trackers):
            for d, det in enumerate(detections):
                iou_score = iou(trks[t], det)
                cos_dist = cosine(tracker.feature, features[d])
                # Lower cost for better match
                cost_matrix[t, d] = 0.5 * (1 - iou_score) + 0.5 * cos_dist

        matched, unmatched_trks, unmatched_dets = [], [], []
        if len(trks) > 0 and len(detections) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_trks = set()
            assigned_dets = set()
            for r, c in zip(row_ind, col_ind):
                if (
                    r < len(self.trackers)
                    and c < len(detections)
                    and cost_matrix[r, c] < self.cos_threshold
                ):
                    matched.append((r, c))
                    assigned_trks.add(r)
                    assigned_dets.add(c)
            unmatched_trks = [i for i in range(len(trks)) if i not in assigned_trks]
            unmatched_dets = [i for i in range(len(detections)) if i not in assigned_dets]
        else:
            unmatched_trks = list(range(len(trks)))
            unmatched_dets = list(range(len(detections)))

        # Update matched trackers (with index checking)
        for t, d in matched:
            if (
                t < len(self.trackers)
                and d < len(detections)
                and d < len(features)
            ):
                self.trackers[t].update(detections[d], features[d])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            if i < len(detections) and i < len(features):
                self.trackers.append(KalmanBoxTracker(detections[i], features[i]))

        # Remove dead trackers
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        # Return list of active tracks: [id, bbox]
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = t.get_state()
                bbox = bbox.tolist() if hasattr(bbox, "tolist") else bbox
                results.append((t.id, bbox))
        return results
