from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np

from panodeon.generators.base import MaskOptions
from panodeon.inference.providers import preload_dlls_for_providers, resolve_execution_providers, session_provider_options

CLASS_NAME_TO_ID = {
    "body": 0,
    "body_with_wheelchair": 5,
    "body_with_crutches": 6,
    "head": 7,
    "face": 16,
    "eye": 17,
    "nose": 18,
    "mouth": 19,
    "ear": 20,
    "hand": 32,
    "hand_left": 33,
    "hand_right": 34,
    "foot": 45,
    "foot_left": 46,
    "foot_right": 47,
}

ONNX_MASK_PROB_EPS = 1e-6


class DeimWholebodySegmenter:
    """Minimal ONNX adapter for PINTO DEIMv2 Wholebody49 instance masks."""

    def __init__(self, model_path: Path, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for DEIMv2 inference") from exc
        self.model_path = model_path
        self._ort = ort
        self.requested_providers = providers or resolve_execution_providers("CPUExecutionProvider")
        self.session = self._create_session(self.model_path, self.requested_providers)
        self.providers = self.session.get_providers()
        self.inputs = self.session.get_inputs()
        self.input_names = {item.name for item in self.inputs}
        self.output_names = {item.name for item in self.session.get_outputs()}
        self.image_input_name = self.inputs[0].name
        image_shape = self.inputs[0].shape
        self.image_size = (
            (int(image_shape[2]), int(image_shape[3]))
            if len(image_shape) >= 4 and all(isinstance(v, int) for v in image_shape[2:4])
            else None
        )
        self.normalize = infer_onnx_normalize_from_model_path(model_path)
        self.detect_session = None
        if "_masks" in model_path.name:
            detect_model_name = model_path.name.replace("_masks", "")
            detect_model_path = model_path.parent / detect_model_name
            if detect_model_path.exists():
                self.detect_session = self._create_session(detect_model_path, self.requested_providers)

    def predict_mask(self, image: np.ndarray, options: MaskOptions) -> np.ndarray:
        image = np.nan_to_num(image, nan=0, posinf=0, neginf=0).astype(np.uint8)
        input_tensor = self._preprocess(image)
        detect_sess = getattr(self, "detect_session", None)
        if detect_sess is not None:
            detect_inputs = detect_sess.get_inputs()
            detect_feed = {detect_inputs[0].name: input_tensor}
            detect_outputs = detect_sess.run(["label_xyxy_score"], detect_feed)
            labels_boxes_scores = detect_outputs[0][0]
            scores = labels_boxes_scores[:, 5]
            if scores.max() < options.score_threshold:
                return np.zeros(image.shape[:2], dtype=np.uint8)
        orig_target_sizes = np.array([[image.shape[1], image.shape[0]]], dtype=np.float32)
        feed = {self.image_input_name: input_tensor}
        if "orig_target_sizes" in self.input_names:
            feed["orig_target_sizes"] = orig_target_sizes

        requested_outputs = ["label_xyxy_score"]
        if "masks" in self.output_names:
            requested_outputs.append("masks")
        outputs = self.session.run(requested_outputs, feed)
        output_map = dict(zip(requested_outputs, outputs))
        labels_boxes_scores = np.nan_to_num(output_map["label_xyxy_score"], nan=0.0, posinf=0.0, neginf=0.0)
        if "masks" not in output_map:
            return self._boxes_to_mask(labels_boxes_scores, image.shape[:2], options)
        probability_masks = np.nan_to_num(output_map["masks"], nan=0.0, posinf=0.0, neginf=0.0)
        return self._masks_to_mask(
            labels_boxes_scores=labels_boxes_scores,
            probability_masks=probability_masks,
            output_shape=image.shape[:2],
            options=options,
        )

    def predict_masks(self, images: list[np.ndarray], options: MaskOptions) -> list[np.ndarray]:
        sanitized_images = [np.nan_to_num(img, nan=0, posinf=0, neginf=0).astype(np.uint8) for img in images]
        first_dim = self.inputs[0].shape[0]
        is_batch_supported = not isinstance(first_dim, int) or first_dim < 0 or first_dim > 1

        if not is_batch_supported:
            return [self.predict_mask(img, options) for img in sanitized_images]

        N = len(sanitized_images)
        tensors = [self._preprocess(img) for img in sanitized_images]
        input_tensor = np.concatenate(tensors, axis=0)

        detect_sess = getattr(self, "detect_session", None)
        valid_indices = list(range(N))

        if detect_sess is not None:
            detect_first_dim = detect_sess.get_inputs()[0].shape[0]
            is_detect_batch_supported = not isinstance(detect_first_dim, int) or detect_first_dim < 0 or detect_first_dim > 1

            if is_detect_batch_supported:
                detect_inputs = detect_sess.get_inputs()
                detect_feed = {detect_inputs[0].name: input_tensor}
                detect_outputs = detect_sess.run(["label_xyxy_score"], detect_feed)
                batch_labels_boxes_scores = detect_outputs[0]
            else:
                detect_inputs = detect_sess.get_inputs()
                batch_labels_boxes_scores = []
                for i in range(N):
                    detect_feed = {detect_inputs[0].name: tensors[i]}
                    detect_outputs = detect_sess.run(["label_xyxy_score"], detect_feed)
                    batch_labels_boxes_scores.append(detect_outputs[0][0])

            valid_indices = []
            for i in range(N):
                scores = batch_labels_boxes_scores[i][:, 5]
                if scores.max() >= options.score_threshold:
                    valid_indices.append(i)

        output_masks = [np.zeros(img.shape[:2], dtype=np.uint8) for img in sanitized_images]
        if not valid_indices:
            return output_masks

        active_tensors = [tensors[i] for i in valid_indices]
        active_input_tensor = np.concatenate(active_tensors, axis=0)

        orig_target_sizes = np.array(
            [[sanitized_images[i].shape[1], sanitized_images[i].shape[0]] for i in valid_indices],
            dtype=np.float32
        )

        feed = {self.image_input_name: active_input_tensor}
        if "orig_target_sizes" in self.input_names:
            feed["orig_target_sizes"] = orig_target_sizes

        requested_outputs = ["label_xyxy_score"]
        if "masks" in self.output_names:
            requested_outputs.append("masks")

        outputs = self.session.run(requested_outputs, feed)
        output_map = dict(zip(requested_outputs, outputs))

        labels_boxes_scores = np.nan_to_num(output_map["label_xyxy_score"], nan=0.0, posinf=0.0, neginf=0.0)
        has_masks = "masks" in output_map

        if has_masks:
            probability_masks = np.nan_to_num(output_map["masks"], nan=0.0, posinf=0.0, neginf=0.0)

        for idx, orig_idx in enumerate(valid_indices):
            img = sanitized_images[orig_idx]
            img_shape = img.shape[:2]
            single_labels_boxes_scores = labels_boxes_scores[idx][None, :, :]

            if not has_masks:
                mask = self._boxes_to_mask(single_labels_boxes_scores, img_shape, options)
            else:
                single_probability_masks = probability_masks[idx][None, :, :, :]
                mask = self._masks_to_mask(
                    labels_boxes_scores=single_labels_boxes_scores,
                    probability_masks=single_probability_masks,
                    output_shape=img_shape,
                    options=options,
                )
            output_masks[orig_idx] = mask

        return output_masks

    def _create_session(self, model_path: Path, providers: list[str]):
        preload_dlls_for_providers(providers)
        return self._ort.InferenceSession(
            str(model_path),
            providers=providers,
            provider_options=session_provider_options(providers),
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.image_size is not None:
            input_h, input_w = self.image_size
            image = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        tensor = image.astype(np.float32) / 255.0
        if self.normalize:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            tensor = (tensor - mean) / std
        tensor = tensor.transpose(2, 0, 1)[None, :, :, :]
        return np.ascontiguousarray(tensor, dtype=np.float32)

    def _masks_to_mask(
        self,
        labels_boxes_scores: np.ndarray,
        probability_masks: np.ndarray,
        output_shape: tuple[int, int],
        options: MaskOptions,
    ) -> np.ndarray:
        selected_class_ids = class_ids_from_names(options.selected_classes)
        merged = np.zeros(output_shape, dtype=np.uint8)
        batch_pred = labels_boxes_scores[0]
        batch_masks = probability_masks[0]
        for idx, pred in enumerate(batch_pred):
            class_id = int(pred[0])
            score = float(pred[5])
            if class_id not in selected_class_ids or score < options.score_threshold:
                continue
            if idx >= len(batch_masks):
                continue
            mask_probs = resize_probability_mask(batch_masks[idx], output_shape)
            binary = mask_probs >= options.mask_threshold
            binary = clip_mask_to_box(binary, pred[1:5], output_shape)
            merged[binary] = 255
        return merged

    def _boxes_to_mask(
        self,
        labels_boxes_scores: np.ndarray,
        output_shape: tuple[int, int],
        options: MaskOptions,
    ) -> np.ndarray:
        selected_class_ids = class_ids_from_names(options.selected_classes)
        merged = np.zeros(output_shape, dtype=np.uint8)
        height, width = output_shape
        for pred in labels_boxes_scores[0]:
            class_id = int(pred[0])
            score = float(pred[5])
            if class_id not in selected_class_ids or score < options.score_threshold:
                continue
            x1, y1, x2, y2 = denormalize_box(pred[1:5], width, height)
            merged[y1:y2, x1:x2] = 255
        return merged


def infer_onnx_normalize_from_model_path(model_path: Path) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", model_path.stem.lower()) if token]
    return not any(token in {"atto", "femto", "pico", "n"} for token in tokens)


def class_ids_from_names(names: tuple[str, ...]) -> set[int]:
    return {CLASS_NAME_TO_ID[name] for name in names if name in CLASS_NAME_TO_ID}


def resize_probability_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[0]
    mask = np.asarray(mask, dtype=np.float32)
    logits = np.log(np.clip(mask, ONNX_MASK_PROB_EPS, 1.0 - ONNX_MASK_PROB_EPS))
    logits -= np.log1p(-np.clip(mask, ONNX_MASK_PROB_EPS, 1.0 - ONNX_MASK_PROB_EPS))
    resized_logits = cv2.resize(
        logits,
        (output_shape[1], output_shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return 1.0 / (1.0 + np.exp(-resized_logits))


def denormalize_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    return (
        max(0, min(int(round(x1)), width - 1)),
        max(0, min(int(round(y1)), height - 1)),
        max(0, min(int(round(x2)), width)),
        max(0, min(int(round(y2)), height)),
    )


def clip_mask_to_box(mask: np.ndarray, box: np.ndarray, output_shape: tuple[int, int], padding: int = 1) -> np.ndarray:
    height, width = output_shape
    x1, y1, x2, y2 = denormalize_box(box, width, height)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    clipped = np.zeros_like(mask, dtype=bool)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped
