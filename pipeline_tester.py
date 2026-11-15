import cv2
import numpy as np
import glob
import time
import csv
import json
import os
from dataclasses import dataclass, asdict, field
from scipy.signal import find_peaks

# 假设你已经有了以下两个模块
from shape_model_builder import ShapeModel, preprocess
from shape_matcher import ShapeMatcher

# ============================================================
# 1. 卡尺精定位模块 (从上一轮代码整合，略做封装)
# ============================================================
class Caliper:
    def __init__(self, img_gray):
        self.img = img_gray

    def extract_profile(self, center, angle_rad, r_min, r_max, width_px):
        cx, cy = center
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        length = int(r_max - r_min)
        if length <= 0: return None
        
        p1 = (cx + r_min * dx, cy + r_min * dy)
        p2 = (cx + r_max * dx, cy + r_max * dy)
        tx, ty = -dy, dx
        half_w = width_px / 2.0
        
        src_tri = np.float32([
            [p1[0] + tx * half_w, p1[1] + ty * half_w],
            [p1[0] - tx * half_w, p1[1] - ty * half_w],
            [p2[0] + tx * half_w, p2[1] + ty * half_w]
        ])
        dst_tri = np.float32([[0, 0], [0, width_px], [length, 0]])
        
        M = cv2.getAffineTransform(src_tri, dst_tri)
        warped = cv2.warpAffine(self.img, M, (length, width_px), flags=cv2.INTER_LINEAR)
        
        weights = cv2.getGaussianKernel(width_px, -1).flatten()
        profile = np.sum(warped * weights[:, np.newaxis], axis=0) / np.sum(weights)
        return profile

    def find_edges(self, profile, min_contrast=5.0):
        if profile is None or len(profile) < 3: return []
        smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), sigmaX=1.0).flatten()
        grad = np.gradient(smoothed)
        g_search = np.abs(grad)
        peaks, _ = find_peaks(g_search, height=min_contrast, distance=2)
        
        edges = []
        for idx in peaks:
            y0, y1, y2 = g_search[idx-1], g_search[idx], g_search[idx+1]
            denom = y0 - 2*y1 + y2
            offset = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-6 else 0.0
            edges.append((idx + offset, 1 if grad[idx] > 0 else -1, g_search[idx]))
        return edges

class CircleFitter:
    @staticmethod
    def fit(points_xy, ransac_thresh=1.5, max_iters=100):
        n = len(points_xy)
        if n < 3: return None, None, None, None
        best_inliers = 0
        best_mask = None
        for _ in range(max_iters):
            idx = np.random.choice(n, 3, replace=False)
            pts = points_xy[idx]
            A = np.array([[2*pts[0][0], 2*pts[0][1], 1],
                          [2*pts[1][0], 2*pts[1][1], 1],
                          [2*pts[2][0], 2*pts[2][1], 1]])
            B = np.array([pts[0][0]**2 + pts[0][1]**2,
                          pts[1][0]**2 + pts[1][1]**2,
                          pts[2][0]**2 + pts[2][1]**2])
            try:
                sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                cx, cy, c = sol
                r2 = c + cx**2 + cy**2
                if r2 <= 0: continue
                r = np.sqrt(r2)
            except: continue
            
            dists = np.abs(np.hypot(points_xy[:,0]-cx, points_xy[:,1]-cy) - r)
            inliers = dists < ransac_thresh
            if np.sum(inliers) > best_inliers:
                best_inliers = np.sum(inliers)
                best_mask = inliers
                
        if best_mask is not None and best_inliers >= 3:
            final_pts = points_xy[best_mask]
            A = np.c_[2*final_pts[:,0], 2*final_pts[:,1], np.ones(len(final_pts))]
            B = final_pts[:,0]**2 + final_pts[:,1]**2
            sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            cx, cy, c = sol
            return cx, cy, np.sqrt(c + cx**2 + cy**2), best_mask
        return None, None, None, None

class WeldLocator:
    def __init__(self, num_calipers=360, r_search=20, width_px=7):
        self.num_calipers = num_calipers
        self.r_search = r_search
        self.width_px = width_px

    def locate(self, img_bgr, cx_init, cy_init, r_init):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        caliper = Caliper(img_gray)
        mid_points = []
        angles = np.linspace(0, 2*np.pi, self.num_calipers, endpoint=False)
        r_min = max(5, r_init - self.r_search)
        r_max = r_init + self.r_search
        
        for angle in angles:
            profile = caliper.extract_profile((cx_init, cy_init), angle, r_min, r_max, self.width_px)
            edges = caliper.find_edges(profile, min_contrast=8.0)
            if len(edges) >= 2:
                edges.sort(key=lambda x: x[0])
                best_pair = None
                min_dist = 15.0
                for i in range(len(edges)-1):
                    p1, p2 = edges[i], edges[i+1]
                    if p1[1] != p2[1]:
                        dist = abs(p1[0] - p2[0])
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (p1, p2)
                if best_pair is not None:
                    mid_r = r_min + (best_pair[0][0] + best_pair[1][0]) / 2.0
                    px = cx_init + mid_r * np.cos(angle)
                    py = cy_init + mid_r * np.sin(angle)
                    mid_points.append([px, py])
                    
        if len(mid_points) < 30:
            return None, "Too few edge points"
            
        cx_fit, cy_fit, r_fit, inliers = CircleFitter.fit(np.array(mid_points), ransac_thresh=1.5)
        if cx_fit is None:
            return None, "RANSAC failed"
            
        inlier_ratio = np.sum(inliers) / len(mid_points)
        if inlier_ratio < 0.4:
            return None, f"Low inlier ratio: {inlier_ratio:.2f}"
            
        return {"cx": cx_fit, "cy": cy_fit, "radius": r_fit, "inlier_ratio": inlier_ratio}, "OK"


# ============================================================
# 2. 测试数据结构与日志定义
# ============================================================
@dataclass
class PipelineRecord:
    filename: str
    # 粗定位结果
    match_success: bool
    match_score: float
    match_time_ms: float
    match_cx: float
    match_cy: float
    match_r: float
    # 精定位结果
    fine_success: bool
    fine_time_ms: float
    fine_cx: float
    fine_cy: float
    fine_r: float
    # 修正量
    dx: float
    dy: float
    dr: float
    # 整体
    total_time_ms: float
    error_msg: str

def save_pipeline_csv(records, path):
    fields = ['filename', 'match_success', 'match_score', 'match_time_ms', 'match_cx', 'match_cy', 'match_r',
              'fine_success', 'fine_time_ms', 'fine_cx', 'fine_cy', 'fine_r',
              'dx', 'dy', 'dr', 'total_time_ms', 'error_msg']
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            d = asdict(r)
            # 格式化浮点数
            for k in ['match_score', 'match_cx', 'match_cy', 'match_r', 
                      'fine_cx', 'fine_cy', 'fine_r', 'dx', 'dy', 'dr']:
                if d[k] is not None and d[k] != -1.0:
                    d[k] = round(d[k], 3)
            for k in ['match_time_ms', 'fine_time_ms', 'total_time_ms']:
                d[k] = round(d[k], 1)
            w.writerow(d)
    print(f"[CSV] 日志已保存至: {path}")

def print_pipeline_summary(records):
    total = len(records)
    match_ok = sum(1 for r in records if r.match_success)
    fine_ok = sum(1 for r in records if r.match_success and r.fine_success)
    
    # 耗时统计（只统计成功的）
    match_times = [r.match_time_ms for r in records if r.match_success]
    fine_times = [r.fine_time_ms for r in records if r.fine_success]
    
    # 修正量统计
    dxs = [r.dx for r in records if r.fine_success]
    dys = [r.dy for r in records if r.fine_success]
    drs = [r.dr for r in records if r.fine_success]
    
    print("\n" + "="*64)
    print(" 全流程批量测试汇总 (Match + Fine Locator)")
    print("="*64)
    print(f"  总数            : {total}")
    print(f"  模板匹配成功数  : {match_ok} ({match_ok/total*100:.1f}%)")
    print(f"  精定位成功数    : {fine_ok} ({fine_ok/total*100:.1f}%)")
    print("-" * 64)
    print(f"  模板匹配平均耗时: {np.mean(match_times):.1f} ms")
    print(f"  精定位平均耗时  : {np.mean(fine_times):.1f} ms")
    print(f"  全流程平均耗时  : {np.mean(match_times) + np.mean(fine_times):.1f} ms")
    print("-" * 64)
    if dxs:
        print(f"  粗 -> 精 修正量 (均值 ± 标准差):")
        print(f"    ΔX = {np.mean(dxs):+.2f} ± {np.std(dxs):.2f} px")
        print(f"    ΔY = {np.mean(dys):+.2f} ± {np.std(dys):.2f} px")
        print(f"    ΔR = {np.mean(drs):+.2f} ± {np.std(drs):.2f} px")
    print("="*64 + "\n")


# ============================================================
# 3. 自动生成 Policy
# ============================================================
def auto_build_policy(sample_image_path, model_radius):
    """
    利用 HoughCircles 从一张样本图估计合理范围
    如果失败，返回一个极宽松的默认值
    """
    print(f"[Auto Policy] 正在分析样本图 {sample_image_path} ...")
    img = cv2.imread(sample_image_path)
    if img is None:
        print("[WARN] 样本图读取失败，使用默认宽松 Policy")
        return (0, img.shape[0]), (0, img.shape[1]), (0.8, 1.2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2.0)
    h, w = gray.shape
    
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min(h,w)//3,
        param1=100, param2=30,
        minRadius=int(model_radius*0.8), maxRadius=int(model_radius*1.2)
    )
    
    if circles is not None and len(circles[0]) > 0:
        cx, cy, r = circles[0][0]
        margin = max(100, int(r * 0.5))
        print(f"[Auto Policy] 估计圆心 ({cx:.0f}, {cy:.0f}), 半径 {r:.0f}")
        return (cy - margin, cy + margin), (cx - margin, cx + margin), (0.95, 1.05)
    else:
        print("[WARN] Hough 未检测到圆，使用全图宽松 Policy")
        return (0, h), (0, w), (0.8, 1.2)


# ============================================================
# 4. 主测试流水线
# ============================================================
def main():
    # --- 配置 ---
    model_path = "shape_model.pkl"
    test_images_dir = "./test_images/*.png"
    out_csv = "pipeline_result.csv"
    
    # 1. 加载模型
    model = ShapeModel.load(model_path)
    matcher = ShapeMatcher(model, max_levels=4)
    locator = WeldLocator(num_calipers=360, r_search=20, width_px=7)
    
    image_paths = sorted(glob.glob(test_images_dir))
    if not image_paths:
        print(f"[ERROR] 未找到测试图片: {test_images_dir}")
        return
        
    # 2. 自动生成 Policy (拿第一张图去猜)
    row_range, col_range, scale_range = auto_build_policy(image_paths[0], model.radius)
    
    records = []
    t_start_all = time.perf_counter()
    
    # 3. 遍历测试
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        print(f"[{i+1}/{len(image_paths)}] {filename}", end=" ", flush=True)
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            records.append(PipelineRecord(filename, False, -1, 0, -1, -1, -1, False, 0, -1, -1, -1, 0, 0, 0, 0, "Load failed"))
            print("-> Load Failed")
            continue
            
        # --- 阶段 A: 模板匹配 ---
        t0 = time.perf_counter()
        try:
            match_res, diag = matcher.match(
                img, angle_extent=0, 
                scale_range=scale_range, scale_step=0.02,
                min_score=0.4, num_matches=1, num_levels=4, debug=True
            )
            t_match = (time.perf_counter() - t0) * 1000
            
            if not match_res:
                records.append(PipelineRecord(filename, False, -1, t_match, -1, -1, -1, False, 0, -1, -1, -1, 0, 0, 0, t_match, f"Match fail: {diag.get('failed_at')}"))
                print("-> Match Failed")
                continue
                
            row, col, angle, scale, score = match_res[0]
            cx_rough, cy_rough, r_rough = col, row, model.radius * scale
            
            # 校验匹配得分是否达到工艺门槛
            if score < 0.6:
                records.append(PipelineRecord(filename, False, score, t_match, cx_rough, cy_rough, r_rough, False, 0, -1, -1, -1, 0, 0, 0, t_match, "Score too low"))
                print(f"-> Low Score ({score:.2f})")
                continue
                
        except Exception as e:
            t_match = (time.perf_counter() - t0) * 1000
            records.append(PipelineRecord(filename, False, -1, t_match, -1, -1, -1, False, 0, -1, -1, -1, 0, 0, 0, t_match, f"Match Error: {e}"))
            print("-> Match Exception")
            continue

        # --- 阶段 B: 卡尺精定位 ---
        t1 = time.perf_counter()
        fine_res, fine_status = locator.locate(img, cx_rough, cy_rough, r_rough)
        t_fine = (time.perf_counter() - t1) * 1000
        
        if fine_res is None:
            records.append(PipelineRecord(
                filename, True, score, t_match, cx_rough, cy_rough, r_rough, 
                False, t_fine, -1, -1, -1, 0, 0, 0, t_match + t_fine, fine_status
            ))
            print(f"-> Match OK, Fine Failed ({fine_status})")
            continue
            
        # --- 成功 ---
        cx_f, cy_f, r_f = fine_res["cx"], fine_res["cy"], fine_res["radius"]
        dx = cx_f - cx_rough
        dy = cy_f - cy_rough
        dr = r_f - r_rough
        
        records.append(PipelineRecord(
            filename, True, score, t_match, cx_rough, cy_rough, r_rough,
            True, t_fine, cx_f, cy_f, r_f, dx, dy, dr, t_match + t_fine, "OK"
        ))
        print(f"-> OK (Score={score:.2f}, dX={dx:+.1f}, dY={dy:+.1f}, dR={dr:+.1f})")

    # 4. 保存日志与分析
    save_pipeline_csv(records, out_csv)
    print_pipeline_summary(records)


if __name__ == "__main__":
    main()
