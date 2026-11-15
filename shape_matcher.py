# shape_matcher.py
import cv2
import numpy as np

# 假设 shape_model_builder.py 里已有 preprocess 和 ShapeModel
from shape_model_builder import preprocess, ShapeModel

class ShapeMatcher:
    """
    基于梯度方向内积的模板匹配 + 图像金字塔分层加速 + 多尺度搜索。
    修复点:
      ① 顶层位置步长固定为小值, 避免粗搜误差超出细化窗口
      ② 顶层用放宽阈值, 避免模糊层正确候选被筛掉
      ③ 顶层候选按尺度分组保留, 避免全局 top-K 挤掉正确候选
      ④ 细化层候选数 K 放宽, 保证有潜力的候选走完全程
    """

    def __init__(self, model, max_levels=5):
        self.model = model
        self.max_levels = max_levels

    # ---------- 金字塔构建 ----------
    def _build_image_pyramid(self, gray, num_levels):
        pyr = [gray]
        for _ in range(num_levels - 1):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    def _build_model_pyramid(self, num_levels):
        pyr = []
        for lvl in range(num_levels):
            scale = 2.0 ** lvl
            pyr.append({
                'points': (self.model.points / scale).astype(np.float32),
                'angles': self.model.angles.copy(),
            })
        return pyr

    def _can_search_at_level(self, img, model_pts):
        H, W = img.shape
        r_max = float(np.sqrt((model_pts ** 2).sum(axis=1)).max())
        margin = int(np.ceil(r_max)) + 2
        return (H > 2 * margin + 4) and (W > 2 * margin + 4)

    # ---------- 图像归一化梯度 ----------
    @staticmethod
    def _gradients(img):
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy) + 1e-6
        return (gx / mag).astype(np.float32), (gy / mag).astype(np.float32)

    # ---------- 批量打分 ----------
    def _score_positions(self, ngx, ngy, cy_arr, cx_arr, theta,
                         points, angles, scale=1.0):
        H, W = ngx.shape
        P = len(cy_arr)
        N = len(points)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        pts_s = points * scale
        rx = pts_s[:, 0] * cos_t - pts_s[:, 1] * sin_t
        ry = pts_s[:, 0] * sin_t + pts_s[:, 1] * cos_t

        rot_ang = angles + theta
        wx = np.cos(rot_ang).astype(np.float32)
        wy = np.sin(rot_ang).astype(np.float32)

        score = np.zeros(P, dtype=np.float32)
        for i in range(N):
            sy = cy_arr + ry[i]
            sx = cx_arr + rx[i]
            sy_i = np.round(sy).astype(np.int32)
            sx_i = np.round(sx).astype(np.int32)
            valid = (sy_i >= 0) & (sy_i < H) & (sx_i >= 0) & (sx_i < W)
            if not valid.any():
                continue
            gxi = np.zeros(P, dtype=np.float32)
            gyi = np.zeros(P, dtype=np.float32)
            gxi[valid] = ngx[sy_i[valid], sx_i[valid]]
            gyi[valid] = ngy[sy_i[valid], sx_i[valid]]
            score += gxi * wx[i] + gyi * wy[i]
        return score / N

    # ---------- 顶层粗筛 ----------
    def _top_level_search(self, ngx, ngy, model_pts, model_angs,
                          angle_start, angle_extent, angle_step,
                          scales, min_score, step_px):
        H, W = ngx.shape
        r_max = float(np.sqrt((model_pts ** 2).sum(axis=1)).max())
        margin = int(np.ceil(r_max * max(scales))) + 2
        if margin * 2 >= min(H, W):
            return []

        ys = np.arange(margin, H - margin, step_px)
        xs = np.arange(margin, W - margin, step_px)
        if len(ys) == 0 or len(xs) == 0:
            return []
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        cy_arr = yy.ravel().astype(np.float32)
        cx_arr = xx.ravel().astype(np.float32)

        angles_list = np.arange(angle_start,
                                angle_start + angle_extent + angle_step / 2,
                                angle_step)

        candidates = []
        for s in scales:
            for theta in angles_list:
                scores = self._score_positions(ngx, ngy, cy_arr, cx_arr,
                                               theta, model_pts, model_angs,
                                               scale=s)
                above = scores >= min_score
                for idx in np.where(above)[0]:
                    candidates.append((float(cy_arr[idx]), float(cx_arr[idx]),
                                       float(theta), float(s),
                                       float(scores[idx])))
        candidates.sort(key=lambda x: -x[4])
        return candidates

    # ---------- 逐层细化 ----------
    def _refine_at_level(self, ngx, ngy, model_pts, model_angs, candidates,
                         angle_prev_step, angle_new_step,
                         scale_prev_step, scale_new_step,
                         min_score, window=2):
        H, W = ngx.shape
        refined = []
        for (cy, cx, theta, scale, _) in candidates:
            y_range = np.arange(int(np.floor(cy - window)),
                                int(np.ceil(cy + window)) + 1)
            x_range = np.arange(int(np.floor(cx - window)),
                                int(np.ceil(cx + window)) + 1)
            y_range = y_range[(y_range >= 0) & (y_range < H)]
            x_range = x_range[(x_range >= 0) & (x_range < W)]
            if len(y_range) == 0 or len(x_range) == 0:
                continue
            yy, xx = np.meshgrid(y_range, x_range, indexing='ij')
            cy_arr = yy.ravel().astype(np.float32)
            cx_arr = xx.ravel().astype(np.float32)

            theta_lo = theta - angle_prev_step / 2
            theta_hi = theta + angle_prev_step / 2
            angles_list = np.arange(theta_lo, theta_hi + angle_new_step / 2,
                                    angle_new_step)

            if scale_prev_step > 0:
                # 放宽到 ±0.75 倍步长, 容忍真值落在两格点之间
                s_lo = scale - scale_prev_step * 0.75
                s_hi = scale + scale_prev_step * 0.75
                scales_list = np.arange(s_lo, s_hi + scale_new_step / 2,
                                        scale_new_step)
                scales_list = scales_list[scales_list > 0]
            else:
                scales_list = [scale]

            best = None
            for s in scales_list:
                for t in angles_list:
                    scores = self._score_positions(
                        ngx, ngy, cy_arr, cx_arr, t,
                        model_pts, model_angs, scale=s)
                    idx = int(np.argmax(scores))
                    sc = float(scores[idx])
                    if best is None or sc > best[4]:
                        best = (float(cy_arr[idx]), float(cx_arr[idx]),
                                float(t), float(s), sc)

            if best is not None and best[4] >= min_score:
                refined.append(best)

        refined.sort(key=lambda x: -x[4])
        return refined

    # ---------- NMS 去重 ----------
    def _select_non_overlapping(self, candidates, num_matches,
                                max_overlap, base_radius):
        """
        注意: 本参数的语义与 Halcon 的 MaxOverlap 相反。
              值越大 → 抑制越狠；值越小 → 抑制越松。
              单目标 (num_matches=1) 时影响可忽略；多目标时要留意。
        """
        selected = []
        for cand in candidates:
            r, c, _, s, _ = cand
            overlap = False
            for (r2, c2, _, s2, _) in selected:
                if np.hypot(r - r2, c - c2) < max_overlap * (base_radius * (s + s2)):
                    overlap = True
                    break
            if not overlap:
                selected.append(cand)
                if len(selected) >= num_matches:
                    break
        return selected

    # ---------- 主入口 ----------
    def match(self, image,
              angle_start=0.0, angle_extent=2 * np.pi,
              angle_step=np.radians(1.0),
              scale_range=(1.0, 1.0),
              scale_step=0.02,
              min_score=0.5, num_matches=1, max_overlap=0.5,
              num_levels=None, step_pixels_top=None,
              refine_window=2):
        """
        返回: [(Row, Column, Angle, Scale, Score), ...]
        """
        gray = preprocess(image)

        # === 第 1 段: 建金字塔 + 自动降层 ===
        levels = self.max_levels if num_levels is None else min(num_levels, self.max_levels)
        levels = max(levels, 1)
        img_pyr = self._build_image_pyramid(gray, levels)
        model_pyr = self._build_model_pyramid(levels)

        while levels > 1 and not self._can_search_at_level(
                img_pyr[levels - 1], model_pyr[levels - 1]['points']):
            levels -= 1
            img_pyr = img_pyr[:levels]
            model_pyr = model_pyr[:levels]

        top = levels - 1

        # === 修复 ①: 顶层步长固定为小值, 与层数无关 ===
        if step_pixels_top is None:
            step_pixels_top = 2 if top <= 3 else 3
        step_top = angle_step * (2 ** top)

        # 尺度列表
        s_min, s_max = scale_range
        if s_max <= s_min + 1e-9:
            scales_top = [s_min]
            scale_prev_step = 0.0
        else:
            scales_top = np.arange(s_min, s_max + scale_step / 2, scale_step).tolist()
            scale_prev_step = scale_step

        # === 第 2 段: 顶层粗筛 ===
        # === 修复 ②: 顶层用放宽阈值, 避免模糊层正确候选被筛掉 ===
        min_score_top = min_score * 0.6 if min_score > 0 else min_score

        ngx, ngy = self._gradients(img_pyr[top])
        candidates = self._top_level_search(
            ngx, ngy,
            model_pyr[top]['points'], model_pyr[top]['angles'],
            angle_start, angle_extent, step_top,
            scales_top, min_score_top, step_pixels_top,
        )
        if not candidates:
            return []

        # === 修复 ③: 顶层候选保留策略 ===
        if len(scales_top) == 1:
            # 单尺度: 退化为全局 top-K
            K = max(num_matches * 10, 20)
            candidates = candidates[:K]
        else:
            # 多尺度: 每个尺度各自保留 top-K, 避免某个尺度的假阳性挤掉正确候选
            K_per_scale = max(num_matches * 3, 5)
            by_scale = {}
            for c in candidates:
                key = round(c[3], 4)
                by_scale.setdefault(key, []).append(c)
            kept = []
            for lst in by_scale.values():
                kept.extend(lst[:K_per_scale])
            kept.sort(key=lambda x: -x[4])
            candidates = kept

        # === 第 3 段: 逐层细化 ===
        current_step = step_top
        for lvl in range(top - 1, -1, -1):
            ngx, ngy = self._gradients(img_pyr[lvl])
            mapped = [(cy * 2.0, cx * 2.0, t, s, sc)
                      for (cy, cx, t, s, sc) in candidates]
            next_step = current_step / 2.0
            next_s_step = scale_prev_step / 2.0 if scale_prev_step > 0 else 0.0
            candidates = self._refine_at_level(
                ngx, ngy,
                model_pyr[lvl]['points'], model_pyr[lvl]['angles'],
                mapped, current_step, next_step,
                scale_prev_step, next_s_step if next_s_step > 0 else scale_step,
                min_score, window=refine_window,
            )
            if not candidates:
                return []
            # === 修复 ④: 细化层 K 放宽 ===
            K = max(num_matches * 15, 40)
            candidates = candidates[:K]
            current_step = next_step
            if scale_prev_step > 0:
                scale_prev_step = next_s_step

        # === 第 4 段: NMS 去重 ===
        return self._select_non_overlapping(
            candidates, num_matches, max_overlap, self.model.radius)


# ---------- 可视化 ----------
def visualize_match(bgr, results, radius):
    vis = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    for (row, col, angle, score) in results:
        cv2.circle(vis, (int(col), int(row)), int(radius), (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(col), int(row)), (0, 0, 255),
                       cv2.MARKER_CROSS, 30, 2)
        text = f"S={score:.3f} A={np.degrees(angle):.2f}"
        cv2.putText(vis, text, (int(col) + 15, int(row) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis
