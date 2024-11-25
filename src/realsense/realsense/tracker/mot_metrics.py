import motmetrics as mm
import numpy as np
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='/clusterhome/bsanthanam/yolov5_InsSeg/yolox/YOLOX/ground_truth/combined_mot_output.txt', help='ground truth file')
    parser.add_argument('--dt', type=str, default='/clusterhome/bsanthanam/yolov5_InsSeg/yolox/YOLOX/YOLOX_outputs/yolox_m/track_vis/2024_10_22_12_32_36.txt', help='prediction file')
    return parser.parse_args()

def load_mot_data(file_path):
    # Load the data
    data = np.loadtxt(file_path, delimiter=',')
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    
    # Convert frame numbers to int
    df['frame'] = df['frame'].astype(int)
    
    # Group by frame
    return df.groupby('frame')

def main(args):

    # Load your data
    gt_data = load_mot_data(args.gt)
    pred_data = load_mot_data(args.dt)

    # Initialize accumulator
    acc = mm.MOTAccumulator(auto_id=True)

    # Process each frame
    for frame in range(1, 13):  # 12 frames, assuming they're numbered 1-12
        # Get ground truth for this frame
        gt_frame = gt_data.get_group(frame) if frame in gt_data.groups else pd.DataFrame()
        
        # Get predictions for this frame
        pred_frame = pred_data.get_group(frame) if frame in pred_data.groups else pd.DataFrame()
        
        # Extract object IDs
        gt_ids = gt_frame['id'].values if not gt_frame.empty else []
        pred_ids = pred_frame['id'].values if not pred_frame.empty else []
        
        # Extract bounding boxes
        gt_bboxes = gt_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values if not gt_frame.empty else []
        pred_bboxes = pred_frame[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values if not pred_frame.empty else []
        
        # Compute distances
        distances = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5) if len(gt_bboxes) and len(pred_bboxes) else []
        
        # Update accumulator
        acc.update(gt_ids, pred_ids, distances)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1'], name='acc')

    print(summary)

if __name__ == "__main__":
    args = parse_args()
    main(args)
