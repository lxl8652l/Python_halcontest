import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

class Caliper:
    """单个卡尺：负责提取 1D 剖面并找亚像素边缘"""
    def __init__(self, img_gray):
        self.img = img_gray

    def extract_profile(self, center, angle_rad, r_min, r_max, width_px):
        """
        沿径向提取灰度剖面。
        center: (cx, cy)
        angle_rad: 卡尺的角度
        r_min, r_max: 沿径向的起点和终点半径
        width_px: 卡尺的切向宽度
        """
        # 计算起点和终点
        cx, cy = center
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        p1 = (int(round(cx + r_min * dx)), int(round(cy + r_min * dy)))
        p2 = (int(round(cx + r_max * dx)), int(round(cy + r_max * dy)))
        
        # 构造旋转矩形 ROI (利用仿射变换拉直)
        length = int(r_max - r_min)
        # 映射目标：长度为 length，宽度为 width_px
        src_pts = np.float32([[p1[0], p1[1]], [p2[0], p2[1]]])
        
        # 切线向量 (用于宽度方向的平移)
        tx, ty = -dy, dx
        
        # 四个角点
        half_w = width_px / 2.0
        pt_A = (p1[0] + tx * half_w, p1[1] + ty * half_w)  # 起点+切向偏移
        pt_B = (p1[0] - tx * half_w, p1[1] - ty * half_w)  # 起点-切向偏移
        pt_C = (p2[0] + tx * half_w, p2[1] + ty * half_w)  # 终点+切向偏移
        
        src_tri = np.float32([pt_A, pt_B, pt_C])
        dst_tri = np.float32([[0, 0], [0, width_px], [length, 0]])
        
        # 仿射变换拉直
        M = cv2.getAffineTransform(src_tri, dst_tri)
        warped = cv2.warpAffine(self.img, M, (length, width_px), flags=cv2.INTER_LINEAR)
        
        # 切向高斯加权平均 (压平宽度方向，得到1D)
        weights = cv2.getGaussianKernel(width_px, -1).flatten()
        profile = np.sum(warped * weights[:, np.newaxis], axis=0) / np.sum(weights)
        
        return profile

    def find_edges(self, profile, polarity=0, min_contrast=5.0):
        """
        一阶导数找边缘，抛物线插值求亚像素
        polarity: 0=任意, 1=由暗到亮(正峰), -1=由亮到暗(负峰)
        """
        if profile is None or len(profile) < 3:
            return []

        # 1. 高斯平滑 (减少高频噪声)
        smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), sigmaX=1.0).flatten()
        
        # 2. 一阶导数 (中心差分)
        grad = np.gradient(smoothed)
        
        # 3. 根据极性过滤
        if polarity == 1:
            g_search = grad.copy()
        elif polarity == -1:
            g_search = -grad.copy()
        else:
            g_search = np.abs(grad)
            
        # 4. 找极值点 (峰值)
        # prominence 强制要求局部突出度，height 强制要求绝对高度
        peaks, properties = find_peaks(g_search, height=min_contrast, distance=2)
        
        edges = []
        for idx in peaks:
            # 5. 亚像素抛物线插值 (三点二次插值)
            y0, y1, y2 = g_search[idx-1], g_search[idx], g_search[idx+1]
            denom = y0 - 2*y1 + y2
            
            if abs(denom) > 1e-6:
                # 偏移量介于 -0.5 到 0.5 之间
                offset = 0.5 * (y0 - y2) / denom
            else:
                offset = 0.0
                
            subpix_pos = idx + offset
            edge_strength = g_search[idx] # 用峰值作为置信度
            
            # 记录: (径向坐标, 极性, 幅值)
            # grad[idx] > 0 说明灰度从内向外增加 (暗到亮)
            edge_pol = 1 if grad[idx] > 0 else -1
            edges.append((subpix_pos, edge_pol, edge_strength))
            
        return edges


class CircleFitter:
    """RANSAC 鲁棒拟合圆"""
    @staticmethod
    def fit(points_xy, ransac_thresh=2.0, max_iters=200):
        """
        输入: Nx2 的 numpy 数组 (x, y)
        返回: (cx, cy, radius, inlier_mask)
        """
        n_points = len(points_xy)
        if n_points < 3:
            return None, None, None, None

        best_cx, best_cy, best_r = 0, 0, 0
        best_inliers = 0
        best_mask = None

        for _ in range(max_iters):
            # 随机选3个点
            idx = np.random.choice(n_points, 3, replace=False)
            pts = points_xy[idx]
            
            # 求解圆方程 (x-a)^2 + (y-b)^2 = r^2
            # 化简得: 2x*a + 2y*b + c = x^2 + y^2, 其中 c = r^2 - a^2 - b^2
            A = np.array([
                [2*pts[0][0], 2*pts[0][1], 1],
                [2*pts[1][0], 2*pts[1][1], 1],
                [2*pts[2][0], 2*pts[2][1], 1]
            ])
            B = np.array([pts[0][0]**2 + pts[0][1]**2,
                          pts[1][0]**2 + pts[1][1]**2,
                          pts[2][0]**2 + pts[2][1]**2])
            
            try:
                sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            except np.linalg.LinAlgError:
                continue
                
            cx = sol[0]
            cy = sol[1]
            r2 = sol[2] + cx**2 + cy**2
            
            if r2 <= 0:
                continue
            r = np.sqrt(r2)
            
            # 计算所有点到这个圆的距离
            dists = np.abs(np.hypot(points_xy[:,0]-cx, points_xy[:,1]-cy) - r)
            inliers = dists < ransac_thresh
            
            n_inliers = np.sum(inliers)
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_cx, best_cy, best_r = cx, cy, r
                best_mask = inliers

        # 用所有内点进行最小二乘最终拟合
        if best_mask is not None and best_inliers >= 3:
            final_pts = points_xy[best_mask]
            # 代数法最小二乘
            A = np.c_[2*final_pts[:,0], 2*final_pts[:,1], np.ones(len(final_pts))]
            B = final_pts[:,0]**2 + final_pts[:,1]**2
            sol, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            cx = sol[0]
            cy = sol[1]
            r = np.sqrt(sol[2] + cx**2 + cy**2)
            return cx, cy, r, best_mask
            
        return None, None, None, None


class WeldLocator:
    def __init__(self, num_calipers=360, r_search=20, width_px=7):
        self.num_calipers = num_calipers
        self.r_search = r_search  # 卡尺在预估半径上下搜多少像素
        self.width_px = width_px

    def locate(self, img_bgr, cx_init, cy_init, r_init):
        """
        主流程: 输入图像和粗定位参数，输出精定位结果
        """
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        caliper = Caliper(img_gray)
        
        # 1. 沿 360° 提取焊缝中点点集
        mid_points = []
        angles = np.linspace(0, 2*np.pi, self.num_calipers, endpoint=False)
        
        r_min = max(5, r_init - self.r_search)
        r_max = r_init + self.r_search
        
        for angle in angles:
            # 提取 1D 剖面
            profile = caliper.extract_profile((cx_init, cy_init), angle, r_min, r_max, self.width_px)
            
            # 提取边缘。焊缝表现为由暗到亮(正峰)和由亮到暗(负峰)的一对极值
            edges = caliper.find_edges(profile, polarity=0, min_contrast=8.0)
            
            if len(edges) >= 2:
                # 筛选出一正一负且靠得近的边缘对
                # 按位置排序
                edges.sort(key=lambda x: x[0])
                
                # 寻找间距最小的异常边缘（间隙的宽度通常在 2~10 像素）
                best_pair = None
                min_dist = 15.0  # 假设间隙最大 15 像素
                
                for i in range(len(edges)-1):
                    p1 = edges[i]
                    p2 = edges[i+1]
                    # 检查极性是不是一正一负
                    if p1[1] != p2[1]:
                        dist = abs(p1[0] - p2[0])
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (p1, p2)
                
                if best_pair is not None:
                    # 求出亚像素中点对应的真实半径
                    mid_r = r_min + (best_pair[0][0] + best_pair[1][0]) / 2.0
                    # 转为 x, y 坐标
                    px = cx_init + mid_r * np.cos(angle)
                    py = cy_init + mid_r * np.sin(angle)
                    mid_points.append([px, py])

        # 2. 收集点并用 RANSAC 拟合圆
        if len(mid_points) < 20:
            return None, "Too few points found"
            
        mid_points = np.array(mid_points)
        
        # 最终进行 RANSAC 拟合
        cx_fit, cy_fit, r_fit, inliers = CircleFitter.fit(mid_points, ransac_thresh=1.5)
        
        if cx_fit is None:
            return None, "RANSAC failed"
            
        inlier_ratio = np.sum(inliers) / len(mid_points)
        
        if inlier_ratio < 0.5:
            return None, f"Low inlier ratio: {inlier_ratio:.2f}"

        # 返回高精度结果
        return {
            "cx": cx_fit,
            "cy": cy_fit,
            "radius": r_fit,
            "inlier_ratio": inlier_ratio,
            "valid_points": mid_points[inliers]
        }, "OK"


# ==========================================
# 调试与可视化
# ==========================================
def visualize_comparison(img_bgr, cx_init, cy_init, r_init, result):
    """
    绘制粗定位与精定位对比图
    """
    # 转成 RGB 用于 matplotlib 显示
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ================= 左图：粗定位 =================
    axes[0].imshow(img_rgb)
    # 画粗定位圆（黄色虚线）
    circle_rough = plt.Circle((cx_init, cy_init), r_init, 
                              color='yellow', fill=False, linestyle='--', linewidth=2, label='Rough (Match)')
    axes[0].add_patch(circle_rough)
    # 画圆心
    axes[0].plot(cx_init, cy_init, 'y+', markersize=12)
    axes[0].set_title(f"Rough Localization (Match Result)\nCenter=({cx_init:.1f}, {cy_init:.1f}), R={r_init:.1f}", fontsize=12)
    axes[0].axis('off')

    # ================= 右图：亚像素精定位 =================
    axes[1].imshow(img_rgb)
    if result is None:
        axes[1].text(0.5, 0.5, "Locate Failed", color='red', fontsize=20, ha='center', va='center')
    else:
        cx_fit, cy_fit, r_fit = result["cx"], result["cy"], result["radius"]
        
        # 画精定位圆（绿色实线）
        circle_fine = plt.Circle((cx_fit, cy_fit), r_fit, 
                                 color='lime', fill=False, linewidth=2, label='Fine (Subpixel)')
        axes[1].add_patch(circle_fine)
        
        # 画提取到的亚像素边缘点（红点）
        pts = result["valid_points"]
        axes[1].plot(pts[:, 0], pts[:, 1], 'r.', markersize=2, label='Inliers')
        
        # 画精定位圆心
        axes[1].plot(cx_fit, cy_fit, 'g+', markersize=12)
        
        # 计算偏差
        dx = cx_fit - cx_init
        dy = cy_fit - cy_init
        dr = r_fit - r_init
        
        info_text = (f"Fine Localization\n"
                     f"Center=({cx_fit:.1f}, {cy_fit:.1f}), R={r_fit:.1f}\n"
                     f"Delta: dX={dx:+.2f} px, dY={dy:+.2f} px, dR={dr:+.2f} px\n"
                     f"Inlier Ratio: {result['inlier_ratio']*100:.1f}%")
        
        axes[1].set_title(info_text, fontsize=12, color='darkgreen')
        axes[1].legend(loc='upper right')

    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 测试代码
    img = cv2.imread("test.png")
    if img is not None:
        # 假设模板匹配给出的粗定位为: cx, cy, r (你可以换成实际匹配结果)
        rough_cx, rough_cy, rough_r = 640, 480, 200  
        
        # 如果为了让对比更明显，你可以手动给粗定位加点随机偏差：
        # rough_cx += 3.5
        # rough_cy -= 2.1
        
        locator = WeldLocator(num_calipers=360, r_search=20, width_px=7)
        result, status = locator.locate(img, rough_cx, rough_cy, rough_r)
        
        print("Status:", status)
        if result:
            print(f"Refined: cx={result['cx']:.2f}, cy={result['cy']:.2f}, r={result['radius']:.2f}")
            
        # 调用对比可视化
        visualize_comparison(img, rough_cx, rough_cy, rough_r, result)
    else:
        print("Image not found.")
