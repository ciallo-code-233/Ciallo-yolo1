import cv2
from ultralytics import YOLO
import argparse
import time
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="YOLO推理脚本，支持pt/onnx，统计平均置信度")
    parser.add_argument("--model", type=str, required=True, help="模型文件路径，best.pt / best.onnx")
    parser.add_argument("--video", type=str, default="test.mp4", help="输入原始视频")
    parser.add_argument("--output", type=str, default="result.mp4", help="输出带标注的视频")
    args = parser.parse_args()

    # 加载模型（pt、onnx都能加载）
    print(f"✅ 加载模型：{args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("❌ 视频打开失败，请检查视频路径！")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"视频分辨率：{w} × {h}，帧率 {fps}")

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 统计变量
    frame_count = 0
    total_infer_time = 0.0
    conf_list = []

    print("\n开始推理......")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 只统计模型推理耗时
        start = time.time()
        results = model(frame, verbose=False)
        end = time.time()
        total_infer_time += end - start

        for res in results:
            if res.boxes is not None:
                for box, kpt in zip(res.boxes, res.keypoints):
                    conf = float(box.conf[0])
                    conf_list.append(conf)
                    cls_name = model.names[int(box.cls[0])]
                    # 绘制四点闭合四边形
                    kpt_np = kpt.xy.cpu().numpy()[0]
                    pts = kpt_np.astype(int).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=(0,255,0), thickness=2)
                    # 类别+置信文字
                    x_text, y_text = int(kpt_np[0][0]), int(kpt_np[0][1])
                    cv2.putText(frame, f"{cls_name} {conf:.2f}",
                                (x_text, y_text - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)
                    # 关键点红点
                    for kx, ky in kpt_np:
                        cv2.circle(frame, (int(kx), int(ky)), 4, (0, 0, 255), -1)
        out.write(frame)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 计算统计结果
    avg_infer_time = total_infer_time / frame_count
    avg_conf = np.mean(conf_list) if len(conf_list) > 0 else 0.0
    total_target_num = len(conf_list)

    print("="*60)
    print(f"总帧数：{frame_count}")
    print(f"单帧平均推理耗时：{avg_infer_time:.4f} s")
    print(f"所有目标平均置信度：{avg_conf:.4f}")
    print(f"视频总共检出目标数量：{total_target_num}")
    print(f"标注视频保存路径：{args.output}")
    print("="*60)

if __name__ == "__main__":
    main()

