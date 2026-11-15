# shape_model_builder.py
import cv2
import numpy as np
from dataclasses import dataclass
import pickle
import sys
import os

# ---- Windows 高 DPI：让 OpenCV 窗口使用物理像素，避免窗口过小 ----
if os.name == 'nt':
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # per-monitor DPI aware
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()         # 系统级 DPI aware（fallback）
    except Exception:
        pass

# ============================================================
# 1. 预处理（建模和匹配共用）
# ============================================================
def preprocess(image, blur_ksize=5, blur_sigma=1.5):
    """
    输入: 任意 BGR / Gray 图像
    输出: 单通道灰度 + 高斯平滑后的图像
    注意: 模板匹配阶段必须使用完全相同的参数，保证特征一致
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
    return blurred


# ============================================================
# 2. 交互式圆形 ROI
# ============================================================
class InteractiveCircleROI:
    """
    操作:
      - 左键在红色圆心附近按下并拖动 → 平移整个圆
      - 左键在蓝色半径控制点附近按下并拖动 → 改变半径
      - Enter / Space → 确认
      - ESC → 取消
      - 窗口可自由拖动边角调整大小，图像自动等比适应窗口
    """

    HIT_THRESHOLD = 12          # 判定"点中控制柄"的像素距离（原图坐标）

    def __init__(self, gray, init_center, init_radius, win_name="ROI Selector"):
        # 内部统一 3 通道便于彩色绘制
        self.image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
        self.H, self.W = self.image.shape[:2]

        self.center = [float(init_center[0]), float(init_center[1])]
        self.radius = float(min(init_radius, 0.48 * min(self.H, self.W)))

        self.win_name = win_name
        self.dragging = None
        self.confirmed = False
        self.cancelled = False

        # 显示状态（由 _fit_window 动态更新）
        self.scale = 1.0
        self._off_x = 0
        self._off_y = 0
        self._win_w = self.W
        self._win_h = self.H
        self._img_scaled = None
        self._cache_key = None
        self._last_win_size = None

    # ---------- 屏幕尺寸 ----------
    @staticmethod
    def _get_screen_size():
        try:
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return sw, sh
        except Exception:
            return 1920, 1080

    # ---------- 根据窗口大小重新 fit 图像 ----------
    def _fit_window(self):
        """
        返回一个 canvas（尺寸 == 当前窗口内容区），图像等比居中放置。
        同时更新 self.scale / self._off_x / self._off_y
        """
        win_w = win_h = 0
        try:
            rect = cv2.getWindowImageRect(self.win_name)
            win_w, win_h = rect[2], rect[3]
        except Exception:
            pass
        if win_w <= 0 or win_h <= 0:
            win_w, win_h = self.W, self.H

        scale = min(win_w / self.W, win_h / self.H)
        disp_w = max(1, int(round(self.W * scale)))
        disp_h = max(1, int(round(self.H * scale)))

        # 只在尺寸变化时重算缩放图（避免每帧重新插值）
        if self._cache_key != (disp_w, disp_h):
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            self._img_scaled = cv2.resize(self.image, (disp_w, disp_h), interpolation=interp)
            self._cache_key = (disp_w, disp_h)

        off_x = (win_w - disp_w) // 2
        off_y = (win_h - disp_h) // 2

        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        canvas[off_y:off_y + disp_h, off_x:off_x + disp_w] = self._img_scaled

        self.scale = scale
        self._off_x, self._off_y = off_x, off_y
        self._win_w, self._win_h = win_w, win_h
        return canvas

    # ---------- 显示坐标 → 原图坐标 ----------
    def _img_coords(self, x, y):
        if self.scale <= 0:
            return -1.0, -1.0
        return (x - self._off_x) / self.scale, (y - self._off_y) / self.scale

    # ---------- 半径控制点（原图坐标） ----------
    def _radius_handle(self):
        return (self.center[0] + self.radius, self.center[1])

    # ---------- 重绘 ----------
    def _redraw(self):
        canvas = self._fit_window()
        s = self.scale

        def to_canvas(px, py):
            return (int(round(px * s + self._off_x)),
                    int(round(py * s + self._off_y)))

        c_disp = to_canvas(*self.center)
        r_disp = max(1, int(round(self.radius * s)))
        rh_disp = to_canvas(*self._radius_handle())

        cv2.circle(canvas, c_disp, r_disp, (0, 255, 0), 2)      # 绿圆
        cv2.circle(canvas, c_disp, 6, (0, 0, 255), -1)          # 红点：圆心
        cv2.circle(canvas, rh_disp, 6, (255, 0, 0), -1)         # 蓝点：半径控制

        info = (f"center=({self.center[0]:.0f},{self.center[1]:.0f})  "
                f"r={self.radius:.1f}  win={self._win_w}x{self._win_h}  "
                f"[Enter]=OK  [ESC]=Cancel")
        cv2.putText(canvas, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

        cv2.imshow(self.win_name, canvas)

    # ---------- 鼠标回调 ----------
    def _on_mouse(self, event, x, y, flags, param):
        ox, oy = self._img_coords(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            cx, cy = self.center
            hx, hy = self._radius_handle()
            if (ox - cx) ** 2 + (oy - cy) ** 2 <= self.HIT_THRESHOLD ** 2:
                self.dragging = 'center'
            elif (ox - hx) ** 2 + (oy - hy) ** 2 <= self.HIT_THRESHOLD ** 2:
                self.dragging = 'radius'

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            if self.dragging == 'center':
                self.center[0] = ox
                self.center[1] = oy
            elif self.dragging == 'radius':
                self.radius = max(5.0, float(np.hypot(ox - self.center[0],
                                                     oy - self.center[1])))
            self._redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    # ---------- 主循环 ----------
    def run(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        sw, sh = self._get_screen_size()

        # 初始窗口：屏幕的 85%，基本铺满，用户可再自己拖大/拖小
        cv2.resizeWindow(self.win_name, int(sw * 0.85), int(sh * 0.85))

        cv2.setMouseCallback(self.win_name, self._on_mouse)
        self._redraw()
        self._last_win_size = (self._win_w, self._win_h)

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):          # Enter / Space
                self.confirmed = True
                break
            if key == 27:                # ESC
                self.cancelled = True
                break
            try:
                if cv2.getWindowProperty(self.win_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.cancelled = True
                    break
            except cv2.error:
                self.cancelled = True
                break

            # 检测窗口尺寸是否变化 → 变则重绘
            try:
                rect = cv2.getWindowImageRect(self.win_name)
                size = (rect[2], rect[3])
            except Exception:
                size = None
            if size != self._last_win_size:
                self._redraw()
                self._last_win_size = size

        cv2.destroyWindow(self.win_name)
        if self.confirmed:
            return ((self.center[0], self.center[1]), self.radius)
        return (None, None)


# ============================================================
# 3. 模板数据结构
# ============================================================

@dataclass
class ShapeModel:
    points: np.ndarray       # (N, 2)  float32  (dx, dy) 相对于圆心
    angles: np.ndarray       # (N,)    float32  梯度方向 rad
    magnitudes: np.ndarray   # (N,)    float32  梯度幅值
    radial_dists: np.ndarray # (N,)    float32  NEW: 每个点到圆心的距离
    polarities: np.ndarray   # (N,)    int8     NEW: +1 径向向外, -1 径向向内
    center: tuple            # 原图中的圆心 (x, y)
    radius: float            # 原图中的焊缝外圆半径
    image_size: tuple

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================
# 4. 从圆形 ROI 提取 ShapeModel
# ============================================================
def extract_shape_model(gray, center, radius,
                        canny_low=80, canny_high=200,
                        min_magnitude=10.0,
                        radial_tol=0.15,
                        polarity=+1,
                        polarity_threshold=0.5,
                        angular_bins=0,
                        max_points=None):
    """
    从圆形 ROI 中提取"环形 + 径向梯度"双重约束的圆边缘点集。

    参数:
      gray               : 已预处理的灰度图
      center, radius     : 交互时圈定的焊缝外圆近似圆心与半径
      canny_low/high     : Canny 阈值。焊缝对比度通常较高，建议 80/200 起步
      min_magnitude      : Sobel 幅值下限，过滤弱噪声
      radial_tol         : 半径容差 (ratio)。环带 = radius × (1 ± radial_tol)
                           - 0.10 ~ 0.15: 焊缝外圆半径稳定时用
                           - 0.20 ~ 0.25: 装配间隙大、外圆本身有波动时用
      polarity           : 极性约束
                           +1 : 只保留"梯度指向径向向外"的点（外边缘，暗→亮）
                           -1 : 只保留"梯度指向径向向内"的点（内边缘，亮→暗）
                            0 : 不约束极性（不推荐，容易混入间隙内壁点）
      polarity_threshold : |cos(梯度方向 - 径向)| 下限，0~1
                           - 0.5  ~ 夹角 ≤ 60°，较严
                           - 0.3  ~ 夹角 ≤ 72°，较松
                           - 0.7  ~ 夹角 ≤ 45°，很严
      angular_bins       : 角度分箱数。0 = 不分箱
                           >0 时：把圆周按 360°/angular_bins 均分，
                                  每个角度只保留"最符合 radius 的点"
                           - 圆上每方向只有一个理想点，用 360 或 720 合适
                           - 焊缝宽、内外两条边都会被检出时，必须 >0
      max_points         : 最终下采样上限

    返回: ShapeModel（含 points / angles / radial_dists / polarities）
    """
    cx, cy = int(round(center[0])), int(round(center[1]))
    r = int(round(radius))

    # ---- 1) Canny 边缘（全图） ----
    edges = cv2.Canny(gray, canny_low, canny_high)

    # ---- 2) Sobel 梯度（与 Canny 同图，方向可靠） ----
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)

    # ---- 3) 收集边缘点 ----
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        raise RuntimeError(
            "全图无边缘点：请降低 canny_low/canny_high")

    m = mag[ys, xs]
    angs = ang[ys, xs]

    # ---- 4) 幅值过滤 ----
    keep = m >= min_magnitude
    xs, ys, m, angs = xs[keep], ys[keep], m[keep], angs[keep]
    if len(xs) == 0:
        raise RuntimeError(
            f"幅值过滤后无剩余点：所有边缘点幅值 < min_magnitude={min_magnitude}")

    # ---- 5) 径向距离过滤（环带） ----
    dx_all = xs.astype(np.float32) - cx
    dy_all = ys.astype(np.float32) - cy
    dist = np.sqrt(dx_all * dx_all + dy_all * dy_all)

    r_lo = radius * (1.0 - radial_tol)
    r_hi = radius * (1.0 + radial_tol)
    band_ok = (dist >= r_lo) & (dist <= r_hi)
    dx_all, dy_all, dist = dx_all[band_ok], dy_all[band_ok], dist[band_ok]
    xs, ys, m, angs = xs[band_ok], ys[band_ok], m[band_ok], angs[band_ok]
    if len(xs) == 0:
        raise RuntimeError(
            f"环带过滤后无剩余点：radial_tol={radial_tol} 太窄，"
            f"或交互时圈定的 radius={radius:.1f} 与实际偏差过大")

    # ---- 6) 极性过滤（梯度方向必须沿径向） ----
    radial_ang = np.arctan2(dy_all, dx_all)  # 该点相对圆心的"径向向外"方向
    cos_diff = np.cos(angs - radial_ang)     # ∈ [-1, 1]

    if polarity == +1:
        keep_pol = cos_diff >= polarity_threshold
    elif polarity == -1:
        keep_pol = cos_diff <= -polarity_threshold
    elif polarity == 0:
        keep_pol = np.ones_like(cos_diff, dtype=bool)
    else:
        raise ValueError(f"polarity 只能是 -1 / 0 / +1，收到 {polarity}")

    xs, ys, m, angs, dist, cos_diff, radial_ang = (
        xs[keep_pol], ys[keep_pol], m[keep_pol],
        angs[keep_pol], dist[keep_pol], cos_diff[keep_pol],
        radial_ang[keep_pol],
    )
    if len(xs) == 0:
        raise RuntimeError(
            f"极性过滤后无剩余点：polarity={polarity}, "
            f"polarity_threshold={polarity_threshold} 过严；"
            f"若焊缝是"暗→亮"向外则用 +1，反之用 -1")

    # ---- 7) 角度分箱（可选：每个方向只保留一个最符合 radius 的点） ----
    if angular_bins > 0 and len(xs) > 0:
        # 把 radial_ang ∈ [-π, π] 映射到 [0, angular_bins)
        bin_idx = ((radial_ang + np.pi) / (2.0 * np.pi) * angular_bins).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, angular_bins - 1)

        radial_err = np.abs(dist - radius)
        keep_idx = []
        for b in range(angular_bins):
            mask_b = bin_idx == b
            if mask_b.any():
                idx_in_bin = np.where(mask_b)[0]
                best = idx_in_bin[np.argmin(radial_err[idx_in_bin])]
                keep_idx.append(best)
        keep_idx = np.array(keep_idx, dtype=int)

        xs, ys, m, angs, dist, cos_diff = (
            xs[keep_idx], ys[keep_idx], m[keep_idx],
            angs[keep_idx], dist[keep_idx], cos_diff[keep_idx],
        )

    # ---- 8) 可选下采样（一般角度分箱后点数已很少） ----
    n = len(xs)
    if max_points is not None and n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        xs, ys, m, angs, dist, cos_diff = (
            xs[idx], ys[idx], m[idx],
            angs[idx], dist[idx], cos_diff[idx],
        )

    # ---- 9) 相对坐标 + 极性标签 ----
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    pts = np.stack([dx, dy], axis=1).astype(np.float32)

    polarities = np.where(cos_diff >= 0, 1, -1).astype(np.int8)

    return ShapeModel(
        points=pts,
        angles=angs.astype(np.float32),
        magnitudes=m.astype(np.float32),
        radial_dists=dist.astype(np.float32),
        polarities=polarities,
        center=(cx, cy),
        radius=float(radius),
        image_size=gray.shape,
    )


# ============================================================
# 5. 可视化：把特征点 + 梯度方向画在原图上
# ============================================================
def visualize_model(bgr, model, max_disp=1200, arrow_len=10,
                    arrow_step=1, radial_tol=0.15):
    """
    arrow_step: 每隔几个点画一个箭头（点太多时用 5~10）
    radial_tol: 与提取时保持一致，用于画环带边界
    """
    vis = bgr.copy()
    H, W = vis.shape[:2]
    s = min(1.0, max_disp / max(H, W))

    cx, cy = model.center
    r = model.radius

    # 环带边界（灰色虚线圆）
    r_lo = int(round(r * (1 - radial_tol)))
    r_hi = int(round(r * (1 + radial_tol)))
    cv2.circle(vis, (int(cx), int(cy)), r_lo, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.circle(vis, (int(cx), int(cy)), r_hi, (150, 150, 150), 1, cv2.LINE_AA)
    # 名义半径（绿色实线）
    cv2.circle(vis, (int(cx), int(cy)), int(round(r)), (0, 255, 0), 1, cv2.LINE_AA)

    for i, ((dx, dy), a, pol) in enumerate(
            zip(model.points, model.angles, model.polarities)):
        px = cx + dx
        py = cy + dy
        # +1 → 红，-1 → 蓝
        color = (0, 0, 255) if pol > 0 else (255, 0, 0)
        cv2.circle(vis, (int(px), int(py)), 1, (0, 255, 255), -1)
        if arrow_step > 0 and i % arrow_step == 0:
            ex = px + arrow_len * np.cos(a)
            ey = py + arrow_len * np.sin(a)
            cv2.line(vis, (int(px), int(py)), (int(ex), int(ey)), color, 1)

    n_pos = int((model.polarities > 0).sum())
    n_neg = int((model.polarities < 0).sum())
    info = f"N={len(model.points)}  pol+={n_pos}  pol-={n_neg}"
    cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    if s < 1.0:
        vis = cv2.resize(vis, (int(W * s), int(H * s)),
                         interpolation=cv2.INTER_AREA)

    cv2.imshow("Extracted Shape Model", vis)
    print(f"模板点数={len(model.points)}  (+1 极性={n_pos}, -1 极性={n_neg})")
    print("按任意键关闭预览...")
    cv2.waitKey(0)
    cv2.destroyWindow("Extracted Shape Model")

# ============================================================
# 6. 主流程
# ============================================================
def main_create_template(img_path, out_path="shape_model.pkl",
                         canny_low=80, canny_high=200, min_magnitude=10.0,
                         radial_tol=0.15, polarity=+1,
                         polarity_threshold=0.5, angular_bins=360,
                         max_points=None):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    print(f"[Load] {img_path}  size={bgr.shape}")

    gray = preprocess(bgr)

    H, W = gray.shape
    default_center = (W // 2, H // 2)
    default_radius = min(W, H) // 4

    roi = InteractiveCircleROI(gray, default_center, default_radius)
    center, radius = roi.run()
    if center is None:
        print("[Cancel] 用户取消。")
        return None
    print(f"[ROI] center=({center[0]:.1f},{center[1]:.1f})  radius={radius:.1f}")

    model = extract_shape_model(
        gray, center, radius,
        canny_low=canny_low, canny_high=canny_high,
        min_magnitude=min_magnitude,
        radial_tol=radial_tol,
        polarity=polarity,
        polarity_threshold=polarity_threshold,
        angular_bins=angular_bins,
        max_points=max_points,
    )
    print(f"[Model] N={len(model.points)}  "
          f"center={model.center}  radius={model.radius:.1f}")

    visualize_model(bgr, model, arrow_step=10, radial_tol=radial_tol)

    model.save(out_path)
    print(f"[Save] {out_path}")
    return model


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "test.png"
    main_create_template(img)
