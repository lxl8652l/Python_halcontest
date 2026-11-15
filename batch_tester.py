# batch_tester.py
import os
import time
import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np


@dataclass
class MatchPolicy:
    """外部判据：什么算'匹配成功'"""
    min_score_final: float = 0.6
    row_range: tuple = None        # (row_min, row_max)，None 表示不约束
    col_range: tuple = None        # (col_min, col_max)
    scale_range_ok: tuple = None   # (s_min, s_max)，如 (0.95, 1.05)
    # Ground Truth（可选）
    gt_center_tol: float = None    # 像素容差，None 表示没有 GT


@dataclass
class MatchRecord:
    filename: str
    success: bool
    failure_reason: str
    score: float
    row: float
    col: float
    angle_deg: float
    scale: float
    elapsed_ms: float
    diag: dict = field(default_factory=dict)


class BatchTester:
    def __init__(self, matcher, policy=None, save_vis_dir=None):
        """
        matcher     : ShapeMatcher 实例
        policy      : MatchPolicy，判据
        save_vis_dir: 若给定，保存每张图的叠加可视化（成功/失败分目录）
        """
        self.matcher = matcher
        self.policy = policy or MatchPolicy()
        self.save_vis_dir = save_vis_dir
        if save_vis_dir:
            os.makedirs(os.path.join(save_vis_dir, "success"), exist_ok=True)
            os.makedirs(os.path.join(save_vis_dir, "failed"), exist_ok=True)

    # ---------- 单张图的判据评估 ----------
    def _evaluate(self, results, diag):
        """
        返回 (success, failure_reason, score, row, col, angle_deg, scale)
        """
        if not results:
            return (False, f"NO_RESULT_{diag.get('failed_at', 'UNKNOWN')}",
                    -1.0, -1.0, -1.0, 0.0, -1.0)

        # 取最高分候选
        row, col, angle, scale, score = results[0]

        if score < self.policy.min_score_final:
            return (False, "LOW_SCORE",
                    score, row, col, np.degrees(angle), scale)

        if self.policy.row_range is not None:
            r_lo, r_hi = self.policy.row_range
            if not (r_lo <= row <= r_hi):
                return (False, "ROW_OUT_OF_RANGE",
                        score, row, col, np.degrees(angle), scale)

        if self.policy.col_range is not None:
            c_lo, c_hi = self.policy.col_range
            if not (c_lo <= col <= c_hi):
                return (False, "COL_OUT_OF_RANGE",
                        score, row, col, np.degrees(angle), scale)

        if self.policy.scale_range_ok is not None:
            s_lo, s_hi = self.policy.scale_range_ok
            if not (s_lo <= scale <= s_hi):
                return (False, "SCALE_OUT_OF_RANGE",
                        score, row, col, np.degrees(angle), scale)

        return (True, "OK", score, row, col, np.degrees(angle), scale)

    # ---------- 保存可视化 ----------
    def _save_vis(self, bgr, results, record):
        if not self.save_vis_dir:
            return
        vis = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        color = (0, 200, 0) if record.success else (0, 0, 255)
        radius = self.matcher.model.radius

        if results:
            row, col, angle, scale, score = results[0]
            r_draw = max(1, int(round(radius * scale)))
            cv2.circle(vis, (int(col), int(row)), r_draw, color, 3)
            cv2.drawMarker(vis, (int(col), int(row)), (0, 0, 255),
                           cv2.MARKER_CROSS, 40, 3)
            text = (f"{'OK' if record.success else 'FAIL'} "
                    f"S={score:.3f} A={np.degrees(angle):.1f} Sc={scale:.2f}")
        else:
            text = f"FAIL: {record.failure_reason}"

        cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, color, 3)
        sub = "success" if record.success else "failed"
        name = os.path.basename(record.filename)
        out = os.path.join(self.save_vis_dir, sub, name)
        cv2.imwrite(out, vis)

    # ---------- 批量执行 ----------
    def run(self, image_paths, match_kwargs, verbose=True):
        """
        image_paths : 图片路径列表
        match_kwargs: 传给 matcher.match() 的参数（不含 image）
        返回: (records, summary)
        """
        records = []
        t_start_all = time.perf_counter()

        for i, path in enumerate(image_paths, 1):
            rec = self._run_one(path, match_kwargs)
            records.append(rec)

            if verbose:
                tag = "OK  " if rec.success else "FAIL"
                print(f"[{i:>4}/{len(image_paths)}] {tag} "
                      f"{os.path.basename(path):<40} "
                      f"S={rec.score:.3f} ({rec.elapsed_ms:.0f}ms) "
                      f"{'' if rec.success else rec.failure_reason}")

        total_elapsed = time.perf_counter() - t_start_all
        summary = self._build_summary(records, total_elapsed)
        return records, summary

    def _run_one(self, path, match_kwargs):
        t0 = time.perf_counter()
        try:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                return MatchRecord(path, False, "LOAD_FAILED",
                                   -1, -1, -1, 0, -1, 0.0, {})

            results, diag = self.matcher.match(bgr, debug=True, **match_kwargs)
            elapsed = (time.perf_counter() - t0) * 1000.0

            success, reason, score, row, col, angle_deg, scale = \
                self._evaluate(results, diag)

            rec = MatchRecord(path, success, reason, score, row, col,
                              angle_deg, scale, elapsed, diag)
            self._save_vis(bgr, results, rec)
            return rec

        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return MatchRecord(path, False, f"EXCEPTION:{type(e).__name__}",
                               -1, -1, -1, 0, -1, elapsed, {"error": str(e)})

    def _build_summary(self, records, total_elapsed):
        total = len(records)
        success = sum(1 for r in records if r.success)
        failed = total - success

        # 失败原因细分
        breakdown = {}
        for r in records:
            if not r.success:
                breakdown[r.failure_reason] = breakdown.get(r.failure_reason, 0) + 1

        # 耗时统计
        times = [r.elapsed_ms for r in records if r.elapsed_ms > 0]
        avg_ms = sum(times) / len(times) if times else 0
        max_ms = max(times) if times else 0
        min_ms = min(times) if times else 0

        # 成功样本的得分统计
        scores_ok = [r.score for r in records if r.success]
        avg_score = sum(scores_ok) / len(scores_ok) if scores_ok else 0

        # 失败样本的 diag 分布（死在哪个环节）
        fail_stage = {}
        for r in records:
            if not r.success:
                stage = r.diag.get('failed_at', 'UNKNOWN') or 'UNKNOWN'
                fail_stage[stage] = fail_stage.get(stage, 0) + 1

        return {
            'total': total,
            'success': success,
            'failed': failed,
            'success_rate': success / total if total else 0,
            'total_elapsed_s': total_elapsed,
            'avg_elapsed_ms': avg_ms,
            'max_elapsed_ms': max_ms,
            'min_elapsed_ms': min_ms,
            'avg_score_ok': avg_score,
            'failure_breakdown': breakdown,
            'fail_stage_breakdown': fail_stage,
            'timestamp': datetime.now().isoformat(),
        }


# ---------- 日志输出 ----------
def save_csv(records, path):
    fields = ['filename', 'success', 'failure_reason', 'score',
              'row', 'col', 'angle_deg', 'scale', 'elapsed_ms',
              'top_raw', 'top_best', 'after_group', 'final_nms', 'failed_at']
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({
                'filename': os.path.basename(r.filename),
                'success': r.success,
                'failure_reason': r.failure_reason,
                'score': round(r.score, 4),
                'row': round(r.row, 2),
                'col': round(r.col, 2),
                'angle_deg': round(r.angle_deg, 2),
                'scale': round(r.scale, 4),
                'elapsed_ms': round(r.elapsed_ms, 1),
                'top_raw': r.diag.get('top_raw', 0),
                'top_best': round(r.diag.get('top_best', -1), 4),
                'after_group': r.diag.get('after_group', 0),
                'final_nms': r.diag.get('final_nms', 0),
                'failed_at': r.diag.get('failed_at', ''),
            })
    print(f"[CSV] {path}")


def save_json(records, summary, path):
    data = {
        'summary': summary,
        'records': [
            {
                'filename': os.path.basename(r.filename),
                'success': r.success,
                'failure_reason': r.failure_reason,
                'score': round(r.score, 4),
                'row': round(r.row, 2),
                'col': round(r.col, 2),
                'angle_deg': round(r.angle_deg, 2),
                'scale': round(r.scale, 4),
                'elapsed_ms': round(r.elapsed_ms, 1),
                'diag': {k: (v if not isinstance(v, np.generic) else float(v))
                         for k, v in r.diag.items()},
            }
            for r in records
        ]
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[JSON] {path}")


def print_summary(summary):
    print("\n" + "=" * 64)
    print("批量测试汇总")
    print("=" * 64)
    print(f"  总数        : {summary['total']}")
    print(f"  成功        : {summary['success']}  ({summary['success_rate']*100:.1f}%)")
    print(f"  失败        : {summary['failed']}  ({(1-summary['success_rate'])*100:.1f}%)")
    print(f"  平均耗时    : {summary['avg_elapsed_ms']:.1f} ms")
    print(f"  最大/最小   : {summary['max_elapsed_ms']:.1f} / {summary['min_elapsed_ms']:.1f} ms")
    print(f"  成功均分    : {summary['avg_score_ok']:.4f}")
    print("-" * 64)
    print("  失败原因细分:")
    for reason, cnt in summary['failure_breakdown'].items():
        print(f"    {reason:<28} : {cnt}")
    print("-" * 64)
    print("  死在哪一环节:")
    for stage, cnt in summary['fail_stage_breakdown'].items():
        print(f"    {stage:<28} : {cnt}")
    print("=" * 64)
